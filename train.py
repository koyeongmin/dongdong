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
import copy
import agent

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
        classification_agent = agent.Agent()
    else:
        classification_agent = agent.Agent()
        classification_agent.load_weights(0, "tensor(1.3984)")

    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        classification_agent.cuda()

    ##############################
    ## Loop for training
    ##############################
    print('Training loop')
    step = 0
    sampling_list = None
    for epoch in range(p.n_epoch):
        classification_agent.training_mode()
        for inputs, labels in loader.Generate(sampling_list):
            loss = classification_agent.train(inputs, labels)

            print(loss)
            step += 1

        if step%100 == 0:
            classification_agent.save_model(int(step/100), loss)           


def testing(lane_agent, test_image, step, loss):
    "Testing"


if __name__ == '__main__':
    Training()

