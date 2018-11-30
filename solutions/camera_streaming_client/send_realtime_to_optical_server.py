import cv2
import numpy as np
from threading import Thread
import socket
import struct
import sys
import cPickle as pickle

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2

    def __del__(self):
        self.video.release()

    def get_frame(self):
        while True:
            success, image = self.video.read()
            if success:
                image = cv2.resize(image, (int(self.width), int(self.height)))
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()



ADDRESS = ("localhost", 10000)


class Streaming(Thread):

    def __init__(self):
        #Thread.__init__(self)
        self.cap = VideoCamera()

    def run(self):
        s = socket.socket()
        s.connect(ADDRESS)

        print("Wait for connection")
        try:
            print("Video client connected:, info")

            while True:
                # get image surface
                image = self.cap.get_frame()
                #print(' Buffer size is %s', sc.buffer_size)
                s.send(struct.pack('!i', len(image)))
                s.send(image)

                len_str = s.recv(4)
                size = struct.unpack('!i', len_str)[0]
                # print('size:', size)

                img_str = b''
                while size > 0:
                    if size >= 4096:
                        data = s.recv(4096)
                    else:
                        data = s.recv(size)

                    if not data:
                        break

                    size -= len(data)
                    img_str += data
                if sys.version_info.major < 3:
                    unserialized_input = pickle.loads(img_str)
                else:
                    unserialized_input = pickle.loads(img_str, encoding='bytes')
                cv2.imshow('opticalflow received', unserialized_input)
                if cv2.waitKey(1) == 27:
                    sys.exit(0)


        except Exception as e:
            print(e)
            s.close()
        finally:
            print("Closing socket and exit")
            s.close()


if __name__ == "__main__":
    Streaming().run()

