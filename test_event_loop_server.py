"""Tests for server.py

Run with:  python3 -m unittest -v test_event_loop_server

Two groups:
  - Unit tests: call the server functions directly (parser, commands, and the
    event-loop helpers on a socketpair). No server process.
  - Integration tests: start the real server in a subprocess and talk to it over TCP.

Wire format (inside the outer 4-byte length frame):
  request:  nstr | len | str1 | len | str2 | ...
  response: res (uint32) | data...
"""
import os
import select
import socket
import struct
import subprocess
import sys
import time
import unittest
from unittest import mock

import server as srv
from server import (Conn, STATE_READ, STATE_WRITE, STATE_DELETE, MAX_MSG_SIZE, MAX_ARGS,
                    RES_OK, RES_ERR, RES_NX)

HERE = os.path.dirname(os.path.abspath(__file__))


def frame(payload: bytes) -> bytes:
    # 4-byte little-endian length header + payload
    return struct.pack('<I', len(payload)) + payload


def req_payload(*args: bytes) -> bytes:
    # Inner request payload: nstr, then len + bytes for each argument
    out = struct.pack('<I', len(args))
    for a in args:
        out += struct.pack('<I', len(a)) + a
    return out


def req(*args: bytes) -> bytes:
    # Full request as sent on the wire
    return frame(req_payload(*args))


def reply(code: int, data: bytes = b"") -> bytes:
    # Full response as sent on the wire
    return frame(struct.pack('<I', code) + data)


OK = reply(RES_OK)
NX = reply(RES_NX)
BAD_REQ = reply(RES_ERR, b"bad req")
UNKNOWN = reply(RES_ERR, b"Unknown cmd")


def recv_exact(sock: socket.socket, n: int, timeout: float = 3.0) -> bytes:
    sock.settimeout(timeout)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def read_reply(sock: socket.socket) -> tuple[int, bytes]:
    (length,) = struct.unpack('<I', recv_exact(sock, 4))
    body = recv_exact(sock, length)
    (code,) = struct.unpack('<I', body[:4])
    return code, body[4:]


def make_conn():
    # server side is non-blocking, exactly like accept_new_connection() sets it up
    server_side, client_side = socket.socketpair()
    server_side.setblocking(False)
    return Conn(server_side), client_side


def quiet(fn, *args):
    # The server prints on malformed input; keep test output readable
    with open(os.devnull, 'w') as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            return fn(*args)
        finally:
            sys.stdout = old


# ---------------------------------------------------------------------------
# Unit tests: request parsing
# ---------------------------------------------------------------------------

class TestParseReq(unittest.TestCase):
    def parse(self, data):
        return quiet(srv.parse_req, data)

    def test_set_command(self):
        self.assertEqual(self.parse(req_payload(b"set", b"k", b"v")), [b"set", b"k", b"v"])

    def test_zero_args(self):
        self.assertEqual(self.parse(req_payload()), [])

    def test_empty_string_argument(self):
        self.assertEqual(self.parse(req_payload(b"get", b"")), [b"get", b""])

    def test_binary_argument(self):
        blob = bytes(range(256))
        self.assertEqual(self.parse(req_payload(b"set", b"k", blob)), [b"set", b"k", blob])

    def test_payload_shorter_than_nstr(self):
        self.assertIsNone(self.parse(b""))
        self.assertIsNone(self.parse(b"\x01\x00\x00"))

    def test_too_many_args(self):
        # Well-formed in every other way, so only the MAX_ARGS check can reject it
        self.assertIsNone(self.parse(req_payload(*[b"x"] * (MAX_ARGS + 1))))

    def test_max_args_accepted(self):
        args = [b"x"] * MAX_ARGS
        self.assertEqual(self.parse(req_payload(*args)), args)

    def test_nstr_larger_than_args_present(self):
        # Claims 3 args but only carries 1
        data = struct.pack('<I', 3) + struct.pack('<I', 3) + b"get"
        self.assertIsNone(self.parse(data))

    def test_truncated_length_field(self):
        data = struct.pack('<I', 1) + b"\x03\x00"
        self.assertIsNone(self.parse(data))

    def test_length_points_past_end(self):
        data = struct.pack('<I', 1) + struct.pack('<I', 10) + b"get"
        self.assertIsNone(self.parse(data))

    def test_trailing_bytes_rejected(self):
        self.assertIsNone(self.parse(req_payload(b"get", b"k") + b"x"))

    def test_raw_text_rejected(self):
        self.assertIsNone(self.parse(b"Hello, server!"))


# ---------------------------------------------------------------------------
# Unit tests: command execution against g_map
# ---------------------------------------------------------------------------

class TestProcessCommand(unittest.TestCase):
    def setUp(self):
        srv.g_map.clear()

    def run_cmd(self, *args):
        return quiet(srv.process_command, req_payload(*args))

    def test_set_then_get(self):
        self.assertEqual(self.run_cmd(b"set", b"k", b"v"), (RES_OK, b""))
        self.assertEqual(self.run_cmd(b"get", b"k"), (RES_OK, b"v"))

    def test_set_overwrites(self):
        self.run_cmd(b"set", b"k", b"v1")
        self.run_cmd(b"set", b"k", b"v2")
        self.assertEqual(self.run_cmd(b"get", b"k"), (RES_OK, b"v2"))

    def test_get_missing_key(self):
        self.assertEqual(self.run_cmd(b"get", b"nope"), (RES_NX, b""))

    def test_empty_value_is_not_missing(self):
        self.run_cmd(b"set", b"k", b"")
        self.assertEqual(self.run_cmd(b"get", b"k"), (RES_OK, b""))

    def test_del_existing_key(self):
        self.run_cmd(b"set", b"k", b"v")
        self.assertEqual(self.run_cmd(b"del", b"k"), (RES_OK, b""))
        self.assertEqual(self.run_cmd(b"get", b"k"), (RES_NX, b""))

    def test_del_missing_key_is_ok(self):
        self.assertEqual(self.run_cmd(b"del", b"nope"), (RES_OK, b""))

    def test_command_name_case_insensitive(self):
        self.run_cmd(b"SET", b"k", b"v")
        self.assertEqual(self.run_cmd(b"GeT", b"k"), (RES_OK, b"v"))

    def test_keys_are_case_sensitive(self):
        self.run_cmd(b"set", b"K", b"v")
        self.assertEqual(self.run_cmd(b"get", b"k"), (RES_NX, b""))

    def test_unknown_command(self):
        self.assertEqual(self.run_cmd(b"foo"), (RES_ERR, b"Unknown cmd"))

    def test_wrong_arg_count(self):
        for args in [(b"get",), (b"get", b"a", b"b"), (b"set", b"k"),
                     (b"set", b"k", b"v", b"x"), (b"del",), (b"del", b"a", b"b")]:
            with self.subTest(args=args):
                self.assertEqual(self.run_cmd(*args), (RES_ERR, b"Unknown cmd"))

    def test_zero_args_is_bad_request(self):
        self.assertEqual(self.run_cmd(), (RES_ERR, b"bad req"))

    def test_malformed_payload_is_bad_request(self):
        self.assertEqual(quiet(srv.process_command, b"Hello, server!"), (RES_ERR, b"bad req"))


# ---------------------------------------------------------------------------
# Unit tests: event-loop helpers
# ---------------------------------------------------------------------------

class TestProcessReadBuffer(unittest.TestCase):
    def setUp(self):
        srv.g_map.clear()
        self.conn, self.peer = make_conn()

    def tearDown(self):
        self.conn.con_socket.close()
        self.peer.close()

    def process(self):
        quiet(srv.process_read_buffer, self.conn, None, None)

    def test_empty_buffer_does_nothing(self):
        self.process()
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(self.conn.wbuf, b"")

    def test_incomplete_header_waits(self):
        self.conn.rbuf.extend(b"\x07\x00\x00")  # 3 of 4 header bytes
        self.process()
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(self.conn.rbuf, b"\x07\x00\x00")  # kept, not dropped
        self.assertEqual(self.conn.wbuf, b"")

    def test_incomplete_body_waits(self):
        data = req(b"set", b"k", b"v")
        self.conn.rbuf.extend(data[:10])  # header + part of the body
        self.process()
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(len(self.conn.rbuf), 10)
        self.assertEqual(self.conn.wbuf, b"")

    def test_one_complete_request(self):
        self.conn.rbuf.extend(req(b"set", b"k", b"v"))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.rbuf, b"")
        self.assertEqual(self.conn.wbuf, OK)
        self.assertEqual(srv.g_map, {b"k": b"v"})

    def test_split_3_then_rest(self):
        data = req(b"set", b"k", b"v")
        self.conn.rbuf.extend(data[:3])
        self.process()
        self.assertEqual(self.conn.state, STATE_READ)
        self.conn.rbuf.extend(data[3:])
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.wbuf, OK)

    def test_pipelined_requests_reply_in_order(self):
        self.conn.rbuf.extend(req(b"set", b"k", b"v") + req(b"get", b"k") + req(b"get", b"nope"))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.rbuf, b"")
        self.assertEqual(self.conn.wbuf, OK + reply(RES_OK, b"v") + NX)

    def test_complete_request_then_partial(self):
        partial = req(b"get", b"k")[:6]
        self.conn.rbuf.extend(req(b"set", b"k", b"v") + partial)
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)  # reply to first is ready
        self.assertEqual(self.conn.wbuf, OK)
        self.assertEqual(self.conn.rbuf, partial)  # second kept for later

    def test_zero_length_frame_is_bad_request(self):
        self.conn.rbuf.extend(frame(b""))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.wbuf, BAD_REQ)

    def test_malformed_payload_keeps_connection(self):
        self.conn.rbuf.extend(frame(b"Hello, server!") + req(b"get", b"k"))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.wbuf, BAD_REQ + NX)

    def test_max_size_request_is_accepted(self):
        # nstr + (len + "set") + (len + "k") + (len + value) == MAX_MSG_SIZE
        value = b"x" * (MAX_MSG_SIZE - 4 - 7 - 5 - 4)
        data = req(b"set", b"k", value)
        self.assertEqual(len(data), 4 + MAX_MSG_SIZE)
        self.conn.rbuf.extend(data)
        self.process()
        self.assertEqual(self.conn.wbuf, OK)
        self.assertEqual(srv.g_map[b"k"], value)

    def test_oversized_message_marks_delete(self):
        self.conn.rbuf.extend(struct.pack('<I', MAX_MSG_SIZE + 1))
        self.process()
        self.assertEqual(self.conn.state, STATE_DELETE)
        self.assertEqual(self.conn.wbuf, b"")

    def test_command_exception_becomes_error_reply(self):
        with mock.patch.object(srv, "process_command", side_effect=RuntimeError("boom")):
            self.conn.rbuf.extend(req(b"get", b"k"))
            self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.rbuf, b"")
        self.assertEqual(self.conn.wbuf, reply(RES_ERR, b"internal error"))


class TestFillBuffer(unittest.TestCase):
    def setUp(self):
        self.conn, self.peer = make_conn()

    def tearDown(self):
        self.conn.con_socket.close()
        self.peer.close()

    def test_no_data_returns_true_without_blocking(self):
        self.assertTrue(srv.fill_buffer(self.conn))
        self.assertEqual(self.conn.rbuf, b"")
        self.assertEqual(self.conn.state, STATE_READ)

    def test_reads_everything_available(self):
        # more than one recv(4096), but small enough to fit the socketpair buffer (~8 KB on macOS)
        data = os.urandom(6000)
        self.peer.sendall(data)
        self.assertTrue(srv.fill_buffer(self.conn))
        self.assertEqual(bytes(self.conn.rbuf), data)

    def test_appends_to_existing_buffer(self):
        data = req(b"set", b"k", b"v")
        self.conn.rbuf.extend(data[:3])
        self.peer.sendall(data[3:])
        srv.fill_buffer(self.conn)
        self.assertEqual(bytes(self.conn.rbuf), data)

    def test_eof_marks_delete(self):
        self.peer.close()
        self.assertFalse(srv.fill_buffer(self.conn))
        self.assertEqual(self.conn.state, STATE_DELETE)

    def test_data_then_eof_keeps_data(self):
        self.peer.sendall(b"abc")
        self.peer.close()
        self.assertFalse(srv.fill_buffer(self.conn))
        self.assertEqual(self.conn.rbuf, b"abc")
        self.assertEqual(self.conn.state, STATE_DELETE)


class TestFlushBuffer(unittest.TestCase):
    def setUp(self):
        self.conn, self.peer = make_conn()

    def tearDown(self):
        self.conn.con_socket.close()
        self.peer.close()

    def test_full_send_returns_to_read(self):
        self.conn.wbuf.extend(OK)
        self.conn.state = STATE_WRITE
        srv.send_response(self.conn, None, None)
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(self.conn.wbuf, b"")
        self.assertEqual(self.conn.wbuf_sent, 0)
        self.assertEqual(recv_exact(self.peer, len(OK)), OK)

    def test_empty_wbuf_returns_to_read(self):
        self.conn.state = STATE_WRITE
        self.assertFalse(srv.flush_buffer(self.conn))
        self.assertEqual(self.conn.state, STATE_READ)

    def test_partial_send_resumes_without_duplicating_bytes(self):
        # Bigger than the kernel send buffer, so send() must stop partway
        data = os.urandom(8 * 1024 * 1024)
        self.conn.wbuf.extend(data)
        self.conn.state = STATE_WRITE

        srv.send_response(self.conn, None, None)
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertGreater(self.conn.wbuf_sent, 0)
        self.assertLess(self.conn.wbuf_sent, len(data))

        # Drain the peer and keep flushing, like repeated POLLOUT events would
        received = bytearray()
        self.peer.setblocking(False)
        deadline = time.time() + 10
        while len(received) < len(data) and time.time() < deadline:
            try:
                received.extend(self.peer.recv(1 << 20))
            except BlockingIOError:
                pass
            if self.conn.state == STATE_WRITE:
                srv.send_response(self.conn, None, None)

        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(bytes(received), data)

    def test_send_to_closed_peer_raises_oserror(self):
        # connection_io relies on this being an OSError so it can drop the client
        self.peer.close()
        self.conn.wbuf.extend(OK)
        self.conn.state = STATE_WRITE
        with self.assertRaises(OSError):
            srv.send_response(self.conn, None, None)


class TestConnectionIO(unittest.TestCase):
    def setUp(self):
        srv.g_map.clear()
        self.conn, self.peer = make_conn()
        self.poller = select.poll()
        self.poller.register(self.conn.fd, select.POLLIN)
        self.connections = {self.conn.fd: self.conn}

    def tearDown(self):
        self.conn.con_socket.close()
        self.peer.close()

    def io(self, event):
        quiet(srv.connection_io, self.conn, event, self.poller, self.connections)

    def test_request_reply_cycle(self):
        self.peer.sendall(req(b"set", b"k", b"v"))
        self.io(select.POLLIN)
        # Small reply: process_and_flush sends it right away
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(recv_exact(self.peer, len(OK)), OK)
        self.assertIn(self.conn.fd, self.connections)

    def test_get_reply_carries_value(self):
        self.peer.sendall(req(b"set", b"k", b"hello") + req(b"get", b"k"))
        self.io(select.POLLIN)
        self.assertEqual(read_reply(self.peer), (RES_OK, b""))
        self.assertEqual(read_reply(self.peer), (RES_OK, b"hello"))

    def test_pollerr_removes_connection(self):
        fd = self.conn.fd
        self.io(select.POLLERR)
        self.assertNotIn(fd, self.connections)
        self.assertEqual(self.conn.con_socket.fileno(), -1)  # closed

    def test_eof_removes_connection(self):
        fd = self.conn.fd
        self.peer.close()
        self.io(select.POLLIN)
        self.assertNotIn(fd, self.connections)

    def test_oversized_message_removes_connection(self):
        fd = self.conn.fd
        self.peer.sendall(struct.pack('<I', MAX_MSG_SIZE + 1))
        self.io(select.POLLIN)
        self.assertNotIn(fd, self.connections)

    def test_write_to_closed_peer_removes_connection(self):
        fd = self.conn.fd
        self.conn.wbuf.extend(OK)
        self.conn.state = STATE_WRITE
        self.peer.close()
        self.io(select.POLLOUT)
        self.assertNotIn(fd, self.connections)


# ---------------------------------------------------------------------------
# Integration tests: real server process over TCP
# ---------------------------------------------------------------------------

def free_port() -> int:
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class TestServerIntegration(unittest.TestCase):
    # g_map lives for the whole server process, so each test uses its own keys

    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [sys.executable, '-c', f'import server as s; s.run_server(port={cls.port})'],
            cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                socket.create_connection(('127.0.0.1', cls.port), timeout=0.2).close()
                return
            except OSError:
                time.sleep(0.05)
        cls.proc.kill()
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=5)
        cls.proc.stderr.close()

    def setUp(self):
        self.socks = []

    def tearDown(self):
        for s in self.socks:
            s.close()
        # Every test must leave the server running
        time.sleep(0.05)
        if self.proc.poll() is not None:
            self.fail("server process died: " + self.proc.stderr.read().decode())

    def client(self) -> socket.socket:
        s = socket.create_connection(('127.0.0.1', self.port), timeout=3)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socks.append(s)
        return s

    def call(self, s, *args) -> tuple[int, bytes]:
        s.sendall(req(*args))
        return read_reply(s)

    def assert_closed_by_server(self, s):
        s.settimeout(3)
        try:
            self.assertEqual(s.recv(100), b"")
        except ConnectionResetError:
            pass  # also counts as closed

    def test_set_get_del(self):
        s = self.client()
        self.assertEqual(self.call(s, b"set", b"sgd", b"v"), (RES_OK, b""))
        self.assertEqual(self.call(s, b"get", b"sgd"), (RES_OK, b"v"))
        self.assertEqual(self.call(s, b"del", b"sgd"), (RES_OK, b""))
        self.assertEqual(self.call(s, b"get", b"sgd"), (RES_NX, b""))

    def test_del_twice(self):
        # Used to crash the whole server (cmd[6] IndexError)
        s = self.client()
        self.assertEqual(self.call(s, b"del", b"twice"), (RES_OK, b""))
        self.assertEqual(self.call(s, b"del", b"twice"), (RES_OK, b""))

    def test_data_shared_across_connections(self):
        a, b = self.client(), self.client()
        self.call(a, b"set", b"shared", b"from-a")
        self.assertEqual(self.call(b, b"get", b"shared"), (RES_OK, b"from-a"))

    def test_binary_key_and_value(self):
        s = self.client()
        key, value = b"\x00\xffkey", bytes(range(256))
        self.call(s, b"set", key, value)
        self.assertEqual(self.call(s, b"get", key), (RES_OK, value))

    def test_errors_keep_connection_open(self):
        s = self.client()
        self.assertEqual(self.call(s, b"foo"), (RES_ERR, b"Unknown cmd"))
        s.sendall(frame(b"Hello, server!"))
        self.assertEqual(read_reply(s), (RES_ERR, b"bad req"))
        self.assertEqual(self.call(s, b"get", b"errs-nope"), (RES_NX, b""))

    def test_message_split_3_then_rest(self):
        s = self.client()
        data = req(b"set", b"split", b"v")
        s.sendall(data[:3])
        time.sleep(0.1)
        s.sendall(data[3:])
        self.assertEqual(read_reply(s), (RES_OK, b""))

    def test_one_byte_at_a_time(self):
        s = self.client()
        for b in req(b"get", b"slow-nope"):
            s.sendall(bytes([b]))
            time.sleep(0.005)
        self.assertEqual(read_reply(s), (RES_NX, b""))

    def test_pipelined_requests(self):
        s = self.client()
        s.sendall(b"".join(req(b"set", b"pipe%d" % i, b"%d" % i) for i in range(100)))
        for _ in range(100):
            self.assertEqual(read_reply(s), (RES_OK, b""))
        s.sendall(b"".join(req(b"get", b"pipe%d" % i) for i in range(100)))
        for i in range(100):
            self.assertEqual(read_reply(s), (RES_OK, b"%d" % i))

    def test_complete_then_partial_still_gets_reply(self):
        s = self.client()
        second = req(b"get", b"cp")
        s.sendall(req(b"set", b"cp", b"v") + second[:6])
        self.assertEqual(read_reply(s), (RES_OK, b""))
        s.sendall(second[6:])
        self.assertEqual(read_reply(s), (RES_OK, b"v"))

    def test_max_size_request(self):
        s = self.client()
        value = b"x" * (MAX_MSG_SIZE - 4 - 7 - 7 - 4)  # key "big" is 3 bytes
        self.assertEqual(self.call(s, b"set", b"big", value), (RES_OK, b""))
        self.assertEqual(self.call(s, b"get", b"big"), (RES_OK, value))

    def test_oversized_message_closes_connection(self):
        s = self.client()
        s.sendall(struct.pack('<I', MAX_MSG_SIZE + 1))
        self.assert_closed_by_server(s)

    def test_half_closed_client_still_gets_reply(self):
        s = self.client()
        s.sendall(req(b"get", b"half-nope"))
        s.shutdown(socket.SHUT_WR)
        self.assertEqual(read_reply(s), (RES_NX, b""))

    def test_slow_client_does_not_block_others(self):
        slow = self.client()
        data = req(b"get", b"slow")
        slow.sendall(data[:5])  # stuck mid-message
        fast = self.client()
        self.assertEqual(self.call(fast, b"set", b"slow", b"v"), (RES_OK, b""))
        slow.sendall(data[5:])
        self.assertEqual(read_reply(slow), (RES_OK, b"v"))

    def test_many_concurrent_clients(self):
        clients = [self.client() for _ in range(50)]
        for i, s in enumerate(clients):
            s.sendall(req(b"set", b"client%d" % i, b"%d" % i))
        for s in clients:
            self.assertEqual(read_reply(s), (RES_OK, b""))
        for i, s in enumerate(clients):
            self.assertEqual(self.call(s, b"get", b"client%d" % i), (RES_OK, b"%d" % i))

    def test_client_disconnect_mid_message(self):
        s = self.client()
        s.sendall(req(b"set", b"never", b"finished")[:7])
        s.close()
        # server must still work for others
        self.assertEqual(self.call(self.client(), b"get", b"never"), (RES_NX, b""))

    def test_client_reset_does_not_kill_server(self):
        for _ in range(10):
            s = self.client()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            s.sendall(req(b"get", b"bye"))
            s.close()  # linger 0 -> RST instead of FIN
        self.assertEqual(self.call(self.client(), b"get", b"bye"), (RES_NX, b""))

    def test_client_that_never_reads_replies(self):
        s = self.client()
        s.sendall(b"".join(req(b"get", b"x") for _ in range(1000)))
        s.close()
        self.assertEqual(self.call(self.client(), b"get", b"x"), (RES_NX, b""))


if __name__ == '__main__':
    unittest.main()
