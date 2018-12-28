from __future__ import print_function
from vision_module import ComputeImage
from threading import Thread
import socket
import struct
import cv2
import time
import argparse
import cPickle as pickle
import zlib


class Receiving(Thread):

    def __init__(self):
        #Thread.__init__(self)
        self.computeImage = ComputeImage() # COMPUTE IMAGE IS THE MODULE YOU HAVE TO LOAD IN ORDER TO SELECT YOUR ALOORITHM
        self.chrono = time.time()
        self.z = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS | 16)
        self.zd = zlib.decompressobj()

    def compress(self, o):
        p = pickle.dumps(o, pickle.HIGHEST_PROTOCOL)
        return p

    def decompress(self, s):
        p = pickle.loads(s)
        return p

    def receive_image(self, sc):
        len_str = sc.recv(4)
        size = struct.unpack('!i', len_str)[0]

        blob = b''
        while size > 0:
            if size >= 4096:
                data = sc.recv(4096)
            else:
                data = sc.recv(size)

            if not data:
                break

            size -= len(data)
            blob += data

        unserialized_blob = self.decompress(blob)
        return unserialized_blob

    def send_image(self, sc, flow):
        serialized_data = self.compress(flow)
        sc.send(struct.pack('!i', len(serialized_data)))
        sc.send(serialized_data)

    def fps_counter(self, nb_loop):
        if time.time() - self.chrono > 1:
            fps = nb_loop / (time.time() - self.chrono)
            self.chrono = time.time()
            print(fps)
            return 0
        return nb_loop

    def __listen_client(self, sc, args):
        first_loop = True
        running = True
        nb_loop = 0
        while running:
            actual_img = self.receive_image(sc)
            if first_loop:
                first_loop = False
                prev_img = actual_img
            if args.fps == 1:
                nb_loop = self.fps_counter(nb_loop)
                nb_loop += 1
            flow = self.computeImage.run(prev_img, actual_img, args)
            self.send_image(sc, flow)
            prev_img = actual_img


    def run(self, args):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.ip, args.port))
        s.listen(1)
        try:
            while True:
                try:
                    print("Waiting for client...")
                    sc, info = s.accept()
                    print("Client accepted, computing...")
                    self.__listen_client(sc, args)

                except struct.error as e:
                    cv2.destroyAllWindows()
        except Exception as e:
            print(e)
            pass

        finally:
            print("\nClosing socket.")
            s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", help="default = any ip is allowed else only the ip specified is allowed", type=str, default="")
    parser.add_argument("-p", "--port", type=int, default=10000)
    parser.add_argument("-c", "--caffemodel", help='path to model', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_weights.caffemodel.h5")
    parser.add_argument("-d", "--deployproto",  help='path to deploy prototxt template', type=str, default="../../flownet2/models/FlowNet2-SD/FlowNet2-SD_deploy.prototxt.template")
    parser.add_argument("-f", "--fps",  help='1 to print fps', type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
    try:
        Receiving().run(args)
    except KeyboardInterrupt:
        print("exit key pressed, closing.")
