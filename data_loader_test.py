#########################################################################
##
##  Data loader source code for TuSimple dataset
##
#########################################################################


import math
import numpy as np
import cv2
import json
import random
import os
from copy import deepcopy
from parameters import Parameters

p = Parameters()

#########################################################################
## some iamge transform utils
#########################################################################
def Translate_Points(point,translation): 
    point = point + translation 
    
    return point

def Rotate_Points(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


#########################################################################
## Data loader class
#########################################################################
class Generator_Test(object):
    ################################################################################
    ## initialize (load data set from url)
    ################################################################################
    def __init__(self):
        # load training set
        self.test_data = []

        category_list = os.listdir(p.test_root_url)
        
        for i in category_list:
            image_list = os.listdir(p.test_root_url+i+'/')
            for j in image_list:
                self.test_data.append(p.test_root_url+i+'/'+j)
        
        #with open("/home/kym/research/autonomous_car_vision/dataset/CULane_dataset/list/train.txt") as f:
        #    self.train_data = f.readlines()

        self.size_test = len(self.test_data)
        print("the number of test sets: ", self.size_test)


    #################################################################################################################
    ## Generate data as much as batchsize and augment data (filp, translation, rotation, gaussian noise, scaling)
    #################################################################################################################
    def Generate(self, sampling_list = None): 
        cuts = [(b, min(b + p.batch_size, self.size_test)) for b in range(0, self.size_test, p.batch_size)]
        random.shuffle(self.test_data)
        
        for start, end in cuts:
            inputs, labels = self.make_input_data(start, end)

            yield inputs, labels

    def make_input_data(self, start, end):
        inputs = []
        labels  = []

        for i in range(start, end):
            image = cv2.resize(cv2.imread(self.test_data[i]), (p.x_size,p.y_size))

            label = int( self.test_data[i].split('/')[-1][:3] )

            inputs.append(np.rollaxis(image, axis=2, start=0))
            labels.append(label)
        
        return np.array(inputs), np.array(labels)
