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
                return image



ADDRESS = ("localhost", 10000)


class Streaming(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.cap = VideoCamera()

    def send_image(self, s):
        image = self.cap.get_frame()
        serialized_data = pickle.dumps(image, protocol=2)
        s.send(struct.pack('!i', len(serialized_data)))
        s.send(serialized_data)

    def receive_image(self, s):
        len_str = s.recv(4)
        size = struct.unpack('!i', len_str)[0]
        blob = b''
        while size > 0:
            if size >= 4096:
                data = s.recv(4096)
            else:
                data = s.recv(size)
            if not data:
                break
            size -= len(data)
            blob += data

        if sys.version_info.major < 3:
            unserialized_blob = pickle.loads(blob)
        else:
            unserialized_blob = pickle.loads(blob, encoding='bytes')
        return unserialized_blob

    def run(self):
        s = socket.socket()
        s.connect(ADDRESS)
        print("Connected")
        while True:
            self.send_image(s)
            flow = self.receive_image(s)

            cv2.imshow('opticalflow received', flow)
            if cv2.waitKey(1) == 27:
                break
        print("Closing socket and exit")
        s.close()


if __name__ == "__main__":
    try:
        Streaming().run()
    except Exception as e:
        print(e)

