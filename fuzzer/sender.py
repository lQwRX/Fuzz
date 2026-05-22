# fuzzer/sender.py
import socket

class ModbusSender:
    def __init__(self, ip, port, timeout=2):
        self.ip = ip
        self.port = port
        self.timeout = timeout

    def send_hex(self, hex_msg):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.ip, self.port))
            sock.send(bytes.fromhex(hex_msg))
            response = sock.recv(1024)
            sock.close()
            return True, response.hex()
        except Exception as e:
            return False, str(e)