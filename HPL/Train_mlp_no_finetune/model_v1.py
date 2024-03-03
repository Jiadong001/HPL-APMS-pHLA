'''
model 1 : Projection Block (MLP)
fully connection layers + batchnorm
'''
import math
import numpy as np

import torch
import torch.nn as nn

class Projection(nn.Module):  # 5 layers
    def __init__(self, input_len):
        super(Projection, self).__init__()
        self.in_len = input_len
        self.projection = nn.Sequential(
                                        nn.Linear(self.in_len, 1024),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(1024),

                                        nn.Linear(1024, 512),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(512),

                                        nn.Linear(512, 128),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(128),

                                        nn.Linear(128, 32),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(32, 2)
                                        )
        
    def forward(self, hla_pep_inputs):
        '''
        hla_pep_inputs: [batch_size, output_len of pretrained model*2]
        '''
        outputs = self.projection(hla_pep_inputs)   # outputs: [batch_size, 2]
        
        return outputs.view(-1, outputs.size(-1))

## previous versions
class Projection00(nn.Module):    # 3 layers
    def __init__(self, input_len):
        super(Projection00, self).__init__()
        self.in_len = input_len
        self.projection = nn.Sequential(
                                        nn.Linear(self.in_len, 256),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(256),

                                        nn.Linear(256, 64),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(64, 2)
                                        )
        
    def forward(self, hla_pep_inputs):
        '''
        hla_pep_inputs: [batch_size, output_len of pretrained model*2]
        '''
        outputs = self.projection(hla_pep_inputs)   # outputs: [batch_size, 2]
        
        return outputs.view(-1, outputs.size(-1))

class Projection10(nn.Module):      # 4 layers
    def __init__(self, input_len):
        super(Projection10, self).__init__()
        self.in_len = input_len
        self.projection = nn.Sequential(
                                        nn.Linear(self.in_len, 1024),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(1024),

                                        nn.Linear(1024, 256),       # 512->5
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(256),

                                        nn.Linear(256, 64),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(64, 2)
                                        )
        
    def forward(self, hla_pep_inputs):
        '''
        hla_pep_inputs: [batch_size, output_len of pretrained model*2]
        '''
        outputs = self.projection(hla_pep_inputs)   # outputs: [batch_size, 2]
        
        return outputs.view(-1, outputs.size(-1))

class Projection11(nn.Module):      # 4 layers + dropout
    def __init__(self, input_len):
        super(Projection11, self).__init__()
        self.in_len = input_len
        self.projection = nn.Sequential(
                                        nn.Linear(self.in_len, 1024),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(1024),
                                        nn.Dropout(0.3),

                                        nn.Linear(1024, 256),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(256),
                                        nn.Dropout(0.3),

                                        nn.Linear(256, 64),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(64, 2)
                                        )
        
    def forward(self, hla_pep_inputs):
        '''
        hla_pep_inputs: [batch_size, output_len of pretrained model*2]
        '''
        outputs = self.projection(hla_pep_inputs)   # outputs: [batch_size, 2]
        
        return outputs.view(-1, outputs.size(-1))

class Projection_lit(nn.Module):    # ->1 2
    def __init__(self, input_len):
        super(Projection_lit, self).__init__()
        self.in_len = input_len
        self.projection = nn.Sequential(
                                        nn.Linear(self.in_len, 1024),
                                        nn.ReLU(True),

                                        #output layer
                                        nn.Linear(1024, 2)
                                        )
        
    def forward(self, hla_pep_inputs):
        '''
        hla_pep_inputs: [batch_size, output_len of pretrained model*2]
        '''
        outputs = self.projection(hla_pep_inputs)   # outputs: [batch_size, 2]
        
        return outputs.view(-1, outputs.size(-1))

