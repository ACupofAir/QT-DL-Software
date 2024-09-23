import socket  			 				#导入socket模块
import os
host = '127.0.0.1'						#主机IP
port = 8080								#端口
web = socket.socket()					#创建TCP/IP套接字
web.bind((host, port))					#绑定端口
web.listen(1)							#设置最多连接数
#创建一个死循环
data_path = "D:/Datasets/shipsEar_22050_test"

file_dirs = os.listdir(data_path)
files = []
for file_dir in file_dirs:
    files += os.listdir(os.path.join(data_path, file_dir))
    print(len(files))

print("服务器等待客户端连接...")

conn, addr = web.accept()  # 建立客户端连接
print(conn, addr)
# file_index = 0
while True:
    # conn.sendall(b'HTTP/1.1 200 OK\r\n\r\nHello World')
    file_dirs = os.listdir(data_path)
    files = []
    for file_dir in file_dirs:
        for file in os.listdir(os.path.join(data_path, file_dir)):

            conn.send(os.path.join(data_path, file_dir, file).encode(encoding='utf-8'))
            data = conn.recv(1024)				#获取客户端请求的数据
            print(data)							#打印出接收到的数据
    # file_index += 1
    # if file_index == len(files):
    conn.send(b'Finished!')
    break

    #向客户端发送数据
    # conn.close()	#关闭连接
