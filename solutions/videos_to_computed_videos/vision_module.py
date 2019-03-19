import cv2
import os

#Computeimage is a generic class that allows us easily change the algorithm within the testing environment
#you can load the properties you want placing a file .csv
#csv format : type,value. Type can either be INT or STR
#class variables will be initiated from the csv in their declaration order

class ComputeImage(object):
    def __init__(self, values_to_load='compute_image.csv'):
        self.blurX, self.blurY = 1, 1
        self.dilatation = 2
        self.first_run = True
        if os.path.isfile(values_to_load):
            self.init_class_from_csv(values_to_load)

    def init_class_from_csv(self, values_to_load):
        content = open(values_to_load, 'r').read().split(',')
        it = 0

        for dir in self.__dir__():
            if dir in self.__dict__ and it < len(content):
                if content[it] == 'int':
                    self.__dict__[dir] = int(content[it + 1])
                if content[it] == 'str':
                    self.__dict__[dir] = str(content[it + 1])
                it += 2

    def run(self, prev_frame, actual_frame, args=None):
        if self.first_run:
            self.first_run = False
        Delta = cv2.absdiff(prev_frame, actual_frame)
        DeltaDilated = cv2.dilate(Delta, None, iterations=self.dilatation)
        DeltaDilatedBlur = cv2.GaussianBlur(DeltaDilated, (self.blurX, self.blurY), 0)
        DeltaDilatedBlurGrey = cv2.cvtColor(DeltaDilatedBlur, cv2.COLOR_BGR2GRAY)
        DeltaDilatedBlurGrey = cv2.cvtColor(DeltaDilatedBlurGrey, cv2.COLOR_GRAY2BGR)
        return DeltaDilatedBlurGrey