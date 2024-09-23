import socket
# import struct
import time


class TcpClient:
    def __init__(self, ipaddress="127.0.0.1", port=8080):
        self.ip = ipaddress  # "127.0.0.1"
        self.port = port  # 40333
        self.tcp_client = socket.socket()
        self.tcp_client.connect((self.ip, self.port))

    def get_message(self):
        data, addr = self.tcp_client.recvfrom(1000)
        # int_len = struct.unpack("<I", data)[0]  # <I 大端字节序
        # data, addr = self.tcp_client.recvfrom(int_len)
        return data.decode()

    # def send_message(self, message):
    #     self.tcp_client.send(message.encode(encoding='utf-8'))
    #     print(f'Client> {message}')

    def send_message(self, message):
        # self.tcp_client.send(struct.pack("<I", len(message)))
        self.tcp_client.send(message.encode(encoding='utf-8'))
        print(f'Client> {message}')

    def close_client(self):
        self.tcp_client.close()
        print("client close")


if __name__ == '__main__':
    client = TcpClient()

    while True:
        file_path = client.get_message()
        if file_path == "Finished!":
            break
        client.send_message(file_path + " result: test")  # 初始化成功
        time.sleep(0.5)
