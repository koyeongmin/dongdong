#############################################################################################################
##
##  Source code for training. In this source code, there are initialize part, training part, ...
##
#############################################################################################################

import cv2
import torch
import numpy as np
from data_loader import Generator
from parameters import Parameters
from test import Tester
from agent import Agent
import copy

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Training():
    print('Training')

    ####################################################################
    ## Hyper parameter
    ####################################################################
    print('Initializing hyper parameter')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')

    if p.model_path == "":
        classification_agent = Agent()
    else:
        classification_agent = Agent()
        classification_agent.load_weights(0, "tensor(1.9614)")

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        classification_agent.cuda()

    #########################################################################
    ## Get tester
    #########################################################################
    print("Get tester")
    tester = Tester()

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 1
    sampling_list = None
    for epoch in range(p.n_epoch):
        classification_agent.training_mode()
        for inputs, labels in loader.Generate(sampling_list):
            total_num = 0
            correct = 0

            loss, prediction = classification_agent.train(inputs, labels)
            loss = loss.cpu().data
            correct += np.sum(prediction==labels)
            total_num += labels.size

            print("epoch: ", epoch, "step: ", step, "loss: ", loss)
            print("accuracy: ", correct/total_num)
            print("correct: ", correct)
            print("total_num: ", total_num)

            if step%100 == 0:
                classification_agent.save_model(int(step/100), loss) 
                tester.testing(classification_agent)
                classification_agent.training_mode()

            step += 1          


def testing(lane_agent, test_image, step, loss):
    "Testing"


if __name__ == '__main__':
    Training()

