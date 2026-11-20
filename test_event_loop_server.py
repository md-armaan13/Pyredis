"""Tests for event_loop_server.py

Run with:  python3 -m unittest -v test_event_loop_server

Two groups:
  - Unit tests: call the server functions directly on a socketpair (no server process).
  - Integration tests: start the real server in a subprocess and talk to it over TCP.
"""
import os
import select
import socket
import struct
import subprocess
import sys
import time
import unittest

import server as srv
from server import Conn, STATE_READ, STATE_WRITE, STATE_DELETE, MAX_MSG_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
REPLY_BODY = b"Hello, client!"


def frame(payload: bytes) -> bytes:
    # 4-byte little-endian length header + payload
    return struct.pack('<I', len(payload)) + payload


REPLY = frame(REPLY_BODY)


def recv_exact(sock: socket.socket, n: int, timeout: float = 3.0) -> bytes:
    sock.settimeout(timeout)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def make_conn():
    # server side is non-blocking, exactly like accept_new_connection() sets it up
    server_side, client_side = socket.socketpair()
    server_side.setblocking(False)
    return Conn(server_side), client_side


def quiet(fn, *args):
    # The server prints every payload; keep test output readable
    with open(os.devnull, 'w') as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            return fn(*args)
        finally:
            sys.stdout = old


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestProcessReadBuffer(unittest.TestCase):
    def setUp(self):
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
        self.conn.rbuf.extend(frame(b"set k v")[:8])  # header + 4 of 7 body bytes
        self.process()
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(len(self.conn.rbuf), 8)
        self.assertEqual(self.conn.wbuf, b"")

    def test_one_complete_message(self):
        self.conn.rbuf.extend(frame(b"set k v"))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.rbuf, b"")
        self.assertEqual(self.conn.wbuf, REPLY)

    def test_split_3_then_8_bytes(self):
        # The scenario walked through earlier: 3 bytes arrive, then 8 more
        data = frame(b"set k v")
        self.conn.rbuf.extend(data[:3])
        self.process()
        self.assertEqual(self.conn.state, STATE_READ)
        self.conn.rbuf.extend(data[3:])
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.wbuf, REPLY)

    def test_pipelined_messages_get_one_reply_each(self):
        self.conn.rbuf.extend(frame(b"a") + frame(b"b") + frame(b"c"))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.rbuf, b"")
        self.assertEqual(self.conn.wbuf, REPLY * 3)

    def test_complete_message_then_partial(self):
        partial = frame(b"hello")[:6]
        self.conn.rbuf.extend(frame(b"one") + partial)
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)  # reply to first is ready
        self.assertEqual(self.conn.wbuf, REPLY)
        self.assertEqual(self.conn.rbuf, partial)  # second kept for later

    def test_zero_length_message(self):
        self.conn.rbuf.extend(frame(b""))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.assertEqual(self.conn.wbuf, REPLY)

    def test_max_size_message_is_accepted(self):
        self.conn.rbuf.extend(frame(b"x" * MAX_MSG_SIZE))
        self.process()
        self.assertEqual(self.conn.state, STATE_WRITE)

    def test_oversized_message_marks_delete(self):
        self.conn.rbuf.extend(struct.pack('<I', MAX_MSG_SIZE + 1))
        self.process()
        self.assertEqual(self.conn.state, STATE_DELETE)
        self.assertEqual(self.conn.wbuf, b"")


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
        self.conn.rbuf.extend(b"\x07\x00\x00")
        self.peer.sendall(b"\x00set k v")
        srv.fill_buffer(self.conn)
        self.assertEqual(bytes(self.conn.rbuf), frame(b"set k v"))

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
        self.conn.wbuf.extend(REPLY)
        self.conn.state = STATE_WRITE
        srv.send_response(self.conn, None, None)
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(self.conn.wbuf, b"")
        self.assertEqual(self.conn.wbuf_sent, 0)
        self.assertEqual(recv_exact(self.peer, len(REPLY)), REPLY)

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
        self.conn.wbuf.extend(REPLY)
        self.conn.state = STATE_WRITE
        with self.assertRaises(OSError):
            srv.send_response(self.conn, None, None)


class TestConnectionIO(unittest.TestCase):
    def setUp(self):
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
        self.peer.sendall(frame(b"ping"))
        self.io(select.POLLIN)
        self.assertEqual(self.conn.state, STATE_WRITE)
        self.io(select.POLLOUT)
        self.assertEqual(self.conn.state, STATE_READ)
        self.assertEqual(recv_exact(self.peer, len(REPLY)), REPLY)
        self.assertIn(self.conn.fd, self.connections)

    def test_poller_watches_pollout_while_writing(self):
        self.peer.sendall(frame(b"ping"))
        self.io(select.POLLIN)
        events = dict(self.poller.poll(1000))
        self.assertTrue(events.get(self.conn.fd, 0) & select.POLLOUT)

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
        self.peer.sendall(frame(b"ping"))
        self.io(select.POLLIN)
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
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [sys.executable, '-c', f'import event_loop_server as s; s.run_server(port={cls.port})'],
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

    def assert_reply(self, s, count=1):
        self.assertEqual(recv_exact(s, len(REPLY) * count), REPLY * count)

    def assert_closed_by_server(self, s):
        s.settimeout(3)
        try:
            self.assertEqual(s.recv(100), b"")
        except ConnectionResetError:
            pass  # also counts as closed

    def test_single_request(self):
        s = self.client()
        s.sendall(frame(b"Hello, server!"))
        self.assert_reply(s)

    def test_sequential_requests_same_connection(self):
        s = self.client()
        for msg in [b"Hello, server!", b"Another message", b"Final message"]:
            s.sendall(frame(msg))
            self.assert_reply(s)

    def test_message_split_3_then_8_bytes(self):
        s = self.client()
        data = frame(b"set k v")
        s.sendall(data[:3])
        time.sleep(0.1)
        s.sendall(data[3:])
        self.assert_reply(s)

    def test_one_byte_at_a_time(self):
        s = self.client()
        for b in frame(b"slow message"):
            s.sendall(bytes([b]))
            time.sleep(0.005)
        self.assert_reply(s)

    def test_pipelined_requests(self):
        s = self.client()
        s.sendall(b"".join(frame(b"msg %d" % i) for i in range(100)))
        self.assert_reply(s, count=100)

    def test_complete_then_partial_still_gets_reply(self):
        s = self.client()
        s.sendall(frame(b"one") + frame(b"hello")[:6])
        self.assert_reply(s)
        s.sendall(frame(b"hello")[6:])
        self.assert_reply(s)

    def test_max_size_message(self):
        s = self.client()
        s.sendall(frame(b"x" * MAX_MSG_SIZE))
        self.assert_reply(s)

    def test_oversized_message_closes_connection(self):
        s = self.client()
        s.sendall(struct.pack('<I', MAX_MSG_SIZE + 1))
        self.assert_closed_by_server(s)

    def test_half_closed_client_still_gets_reply(self):
        s = self.client()
        s.sendall(frame(b"last request"))
        s.shutdown(socket.SHUT_WR)
        self.assert_reply(s)

    def test_slow_client_does_not_block_others(self):
        slow = self.client()
        slow.sendall(frame(b"partial")[:5])  # stuck mid-message
        fast = self.client()
        fast.sendall(frame(b"quick"))
        self.assert_reply(fast)
        slow.sendall(frame(b"partial")[5:])
        self.assert_reply(slow)

    def test_many_concurrent_clients(self):
        clients = [self.client() for _ in range(50)]
        for i, s in enumerate(clients):
            s.sendall(frame(b"client %d" % i))
        for s in clients:
            self.assert_reply(s)

    def test_client_disconnect_mid_message(self):
        s = self.client()
        s.sendall(frame(b"never finished")[:7])
        s.close()
        # server must still work for others
        self.test_single_request()

    def test_client_reset_does_not_kill_server(self):
        for _ in range(10):
            s = self.client()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            s.sendall(frame(b"bye"))
            s.close()  # linger 0 -> RST instead of FIN
        self.test_single_request()

    def test_client_that_never_reads_replies(self):
        s = self.client()
        s.sendall(b"".join(frame(b"x") for _ in range(1000)))
        s.close()
        self.test_single_request()


if __name__ == '__main__':
    unittest.main()
