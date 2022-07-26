#########################################################################
##
## train agent that has some utility for training and saving.
##
#########################################################################

import torch.nn as nn
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function as F
from network import classification_network
from parameters import Parameters

p = Parameters()

############################################################
##
## agent for lane detection
##
############################################################
class Agent(nn.Module):

    #####################################################
    ## Initialize
    #####################################################
    def __init__(self):
        super(Agent, self).__init__()

        self.p = Parameters()

        self.classification_network = classification_network()

        self.setup_optimizer()

        print("model parameters: ")
        print(self.count_parameters(self.classification_network))

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def setup_optimizer(self):
        self.classification_network_optim = torch.optim.Adam(self.classification_network.parameters(),
                                                    lr=self.p.l_rate,
                                                    weight_decay=self.p.weight_decay)


    #####################################################
    ## train
    #####################################################
    def train(self, inputs, labels):

        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()
        prediction = self.classification_network(inputs)
        labels_tensor = torch.from_numpy(labels).long().cuda()

        c = nn.CrossEntropyLoss() 
        loss = c(prediction, labels_tensor)   


        self.classification_network_optim.zero_grad()
        loss.backward()
        self.classification_network_optim.step()

        return loss, torch.argmax(prediction, dim=1).cpu().data.numpy().astype(np.int64)

    #####################################################
    ## predict lanes
    #####################################################
    def predict(self, inputs):
        inputs = torch.from_numpy(inputs).float() 
        inputs = Variable(inputs).cuda()

        return torch.argmax( self.classification_network(inputs), dim=1 ).cpu().data.numpy().astype(np.int64)

    #####################################################
    ## Training mode
    #####################################################                                                
    def training_mode(self):
        self.classification_network.train()

    #####################################################
    ## evaluate(test mode)
    #####################################################                                                
    def evaluate_mode(self):
        self.classification_network.eval()

    #####################################################
    ## Setup GPU computation
    #####################################################                                                
    def cuda(self):
        GPU_NUM = 0
        device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        self.classification_network.cuda()

    #####################################################
    ## Load save file
    #####################################################
    def load_weights(self, epoch, loss):
        self.classification_network.load_state_dict(
            torch.load(p.model_path+str(epoch)+'_'+str(loss)+'_'+'classification_network.pkl', map_location='cuda:0'),False
        )

    #####################################################
    ## Save model
    #####################################################
    def save_model(self, epoch, loss):
        torch.save(
            self.classification_network.state_dict(),
            p.save_path+str(epoch)+'_'+str(loss)+'_'+'classification_network.pkl'
        )

