import socket
import struct

MAX_MSG_SIZE = 4096

def read_full(con_socket: socket.socket, n: int) -> bytes:
    # recv() may return fewer bytes than asked, so loop until we have n
    buf = bytearray()
    while len(buf) < n:
        chunk = con_socket.recv(n - len(buf))
        if not chunk:
            # Peer closed the connection; return what we got (may be short)
            break
        buf.extend(chunk)
    return bytes(buf)


def query(client_socket: socket.socket, msg: str) -> bool:
    payload = msg.encode('utf-8', errors='replace')
    length = len(payload)
    if length > MAX_MSG_SIZE:
        print("Error: Message length exceeds maximum allowed size")
        return False

    header = struct.pack('<I', length)
    client_socket.sendall(header + payload)

    #Read response from the server
    response_header = read_full(client_socket, 4)
    if not response_header or len(response_header) < 4:
        print("Error: Failed to read response header")
        return False
    (response_length,) = struct.unpack('<I', response_header)

    response = read_full(client_socket, response_length)
    if response is None or len(response) < response_length:
        print("Error: unexpected end of stream while reading response")
        return False

    print("Received response:", response.decode('utf-8', errors='replace'))
    return True


def __main__():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    #connec to the server
    sock.connect(('127.0.0.1', 5050))

    for msg in ["Hello, server!", "Another message", "Final message"]:
        # Send a message
        if not query(client_socket=sock, msg=msg):
            print("Error sending message:", msg)
            break


if __name__ == "__main__":
    __main__()