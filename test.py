#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
import numpy as np
from copy import deepcopy
from data_loader_test import Generator_Test
import time
from parameters import Parameters
import os
from tqdm import tqdm

p = Parameters()

###############################################################
##
## testing
## 
###############################################################
class Tester():
    def __init__(self):
        print('Testing setup')
        
        print("Get dataset")
        self.loader = Generator_Test()


    ##############################
    ## testing
    ##############################
    def testing(self, classification_agent):

        print('Testing loop')
        classification_agent.evaluate_mode()

        progressbar = tqdm(range(self.loader.size_test//p.batch_size))

        total_num = 0
        correct = 0
        prevTime = time.time()
        for inputs, labels in self.loader.Generate():
            prediction = classification_agent.predict(inputs)
            correct += np.sum(prediction==labels)
            total_num += labels.size
            progressbar.update(1)
        progressbar.close()

        curTime = time.time()
        sec = curTime - prevTime

        print("------------ test result ------------------")
        print("the number of total test data: ", total_num)
        print("accuracy: ", correct/total_num)
        print("time: ", sec)


