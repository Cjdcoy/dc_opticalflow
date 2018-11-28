from threading import Thread
import socket
import struct
import cv2
import numpy as np
import time

ADDRESS = ("", 10000)


class Receiving(Thread):

    def __init__(self):
        Thread.__init__(self)

    def __listen_client(self, sc):
        running = True
        nb_loop = 0
        one_fps = 1
        start_time = time.time()
        while running:
            # receive size
            len_str = sc.recv(4)
            size = struct.unpack('!i', len_str)[0]
            #print('size:', size)

            img_str = b''
            while size > 0:
                if size >= 4096:
                    data = sc.recv(4096)
                else:
                    data = sc.recv(size)

                if not data:
                    break

                size -= len(data)
                img_str += data

            img = cv2.imdecode(np.fromstring(img_str, dtype=np.uint8), cv2.IMREAD_COLOR)
            if time.time() - start_time > one_fps:
                fps = nb_loop / (time.time() - start_time)
                start_time = time.time()
                print(fps)
                nb_loop = 0
            nb_loop += 1
            #cv2.imshow('i', img)
            #cv2.waitKey(1)

            #print('len:', len(img_str))

    def run(self):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(ADDRESS)
        s.listen(1)
        # s = socket.socket()
        # s.connect(ADDRESS)

        try:
            while True:
                try:
                    sc, info = s.accept()
                    self.__listen_client(sc)

                except struct.error as e:
                    cv2.destroyAllWindows()

        except Exception as e:
            pass

        finally:
            print("Closing socket and exit")
            s.close()


if __name__ == "__main__":
	Receiving().run()
