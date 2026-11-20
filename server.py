import struct
import socket
import select


MAX_MSG_SIZE = 4096
MAX_WBUF = 64 * 1024  # 64 KiB

STATE_READ = 0
STATE_WRITE = 1
STATE_DELETE = 2
class Conn:
    def __init__(self, con_socket: socket.socket):
        self.con_socket = con_socket
        self.fd = con_socket.fileno()
        self.state = STATE_READ
        self.rbuf = bytearray()
        self.wbuf = bytearray()
        self.wbuf_sent = 0

def fill_buffer(conn: Conn) -> bool:
    while True:
        try:
            chunk  = conn.con_socket.recv(4096)
            if not chunk:
                # Connection closed by client
                conn.state = STATE_DELETE
                return False

            conn.rbuf.extend(chunk)
        except BlockingIOError:
            #stop reading when no more data is available
            break

    return True

def accept_new_connection(server_socket: socket.socket, poller: select.poll, connections: dict[int, Conn]):
    try:
        con_socket, _ = server_socket.accept()
        con_socket.setblocking(False)
        conn = Conn(con_socket)
        connections[conn.fd] = conn

        poller.register(conn.fd, select.POLLIN)

    except OSError as e:
        # BlockingIOError (no pending connection) or ConnectionAbortedError (client gave up before accept)
        if not isinstance(e, BlockingIOError):
            print("Accept error:", e)

def flush_buffer(conn: Conn) -> bool:
    remaining = len(conn.wbuf) - conn.wbuf_sent
    if remaining == 0:
        conn.state = STATE_READ
        conn.wbuf_sent = 0
        conn.wbuf.clear()
        return False

    try:
        # memoryview slice sends from the offset without copying the remaining bytes.
        # The view is released before wbuf.clear() below, otherwise the resize would raise BufferError
        with memoryview(conn.wbuf) as view:
            sent = conn.con_socket.send(view[conn.wbuf_sent:])
        # Update the number of bytes sent
        conn.wbuf_sent += sent
        if conn.wbuf_sent == len(conn.wbuf):
            conn.state = STATE_READ
            conn.wbuf_sent = 0
            conn.wbuf.clear()
            return False

        #Data still remains to be sent, keep the state as write
        return True
    except BlockingIOError:
        # Socket not ready for writing, will try again later
        return False

def send_response(conn: Conn, poller: select.poll, connections: dict[int, Conn]):
    while flush_buffer(conn):
        pass

def process_read_buffer(conn: Conn, poller: select.poll, connections: dict[int, Conn]):
    if len(conn.rbuf) < 4:
        # Not enough data to read the length header
        return

    while len(conn.rbuf) >= 4:
        # Unpack length (little-endian uint32)
        if len(conn.wbuf) - conn.wbuf_sent > MAX_WBUF:
            print("Write buffer exceeds maximum allowed size")
            break

        length = struct.unpack('<I', conn.rbuf[:4])[0]

        if length > MAX_MSG_SIZE:
            print("Error: Message length exceeds maximum allowed size")
            conn.state = STATE_DELETE
            return

        if len(conn.rbuf) < 4 + length:
            # Not enough data yet, but replies for earlier messages may already be in wbuf
            break

        payload = conn.rbuf[4:4 + length]
        print("Received payload:", payload.decode('utf-8', errors='replace'))
        del conn.rbuf[:4 + length]  # Remove processed data from rbuf

        # Construct the reply message
        reply = "Hello, client!".encode('utf-8')
        reply_length = len(reply)
        reply_header = struct.pack('<I', reply_length)
        reply = reply_header + reply
        conn.wbuf.extend(reply)  # Store the reply in the write buffer

    # All replies for this batch are buffered, switch to write if there is anything to send
    if conn.wbuf:
        conn.state = STATE_WRITE

def process_and_flush(conn: Conn, poller: select.poll, connections: dict[int, Conn]):
    while True:
        process_read_buffer(conn, poller, connections)
        if conn.state != STATE_WRITE:
            return  # need more data, or connection marked for delete
        send_response(conn, poller, connections)
        if conn.state != STATE_READ:
            return  # kernel send buffer full, wait for POLLOUT
        # wbuf fully drained: rbuf may still hold requests held back by MAX_WBUF,
        # and no POLLIN will come for them, so loop and process them now

def connection_io(conn: Conn, event: int, poller: select.poll, connections: dict[int, Conn]):
    if event & (select.POLLERR | select.POLLNVAL):
        # Socket error or invalid fd, nothing more can be done with this connection
        conn.state = STATE_DELETE
    elif conn.state == STATE_READ:
        try:
            fill_buffer(conn)
            process_and_flush(conn, poller, connections)
        except OSError as e:
            # ConnectionResetError, etc. Drop this client only, keep the server running
            print("Connection error:", e)
            conn.state = STATE_DELETE
    elif conn.state == STATE_WRITE:
        try:
            send_response(conn, poller, connections)
            if conn.state == STATE_READ:
                # wbuf drained: handle requests left in rbuf by backpressure
                process_and_flush(conn, poller, connections)
        except OSError as e:
            # BrokenPipeError, ConnectionResetError, etc.
            print("Connection error:", e)
            conn.state = STATE_DELETE

    if conn.state == STATE_READ:
        poller.modify(conn.fd, select.POLLIN)
    elif conn.state == STATE_WRITE:
        poller.modify(conn.fd, select.POLLOUT)
    elif conn.state == STATE_DELETE:
        poller.unregister(conn.fd)
        conn.con_socket.close()
        del connections[conn.fd]

def run_server(host: str = '127.0.0.1', port: int = 5050):

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((host, port))
    server_socket.listen(1)

    server_socket.setblocking(False)


    poller = select.poll()

    poller.register(server_socket, select.POLLIN)

    #Connection dictionary to keep track of active connections
    connections: dict[int, Conn] = {}

    while True:
        #Polling for events
        events = poller.poll(1000)  # Timeout in milliseconds

        for fd, event in events:
            #new connection
            if fd == server_socket.fileno():
                accept_new_connection(server_socket, poller, connections)
            elif fd in connections:
                conn = connections[fd]

                connection_io(conn, event, poller, connections)

def __main__():
    try:
        print("Starting server on 127.0.0.1:5050")
        run_server()
    except KeyboardInterrupt:
        print("Server shutting down...")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    __main__()