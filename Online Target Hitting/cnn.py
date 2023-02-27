'''
This script is used to generate the feedforward via CNN
'''
import numpy as np
import CNN
import torch
import math
import time
import matplotlib.pyplot as plt
# %%
limit_joint    = np.array([90, 60, 60]) * math.pi / 180
limit_pressure = np.array([4500, 6500, 6000])
root_path      = '/home/hao/Desktop/Interception/cnn_model'

def get_norm(y):
    y = y / limit_joint.reshape(-1, 1)
    return y

def get_denorm(ff):
    ff = ff * limit_pressure.reshape(-1, 1)
    return ff

def get_difference(a, t):
    b = np.hstack((np.zeros(3).reshape(-1, 1), (a[:, 1:]-a[:, 0:-1])/t))
    return b

def get_compensated_data(data=None, h_in_left=20):
    I = np.zeros((h_in_left, 3))
    new_data = np.vstack((I, data, I))
    return new_data

def get_data_point(k, data=None, h_in_left=20):
    x0 = torch.tensor(data[0][k:k+2*h_in_left+1, :]).view(1, 1, -1, 3)
    x1 = torch.tensor(data[1][k:k+2*h_in_left+1, :]).view(1, 1, -1, 3)
    x2 = torch.tensor(data[2][k:k+2*h_in_left+1, :]).view(1, 1, -1, 3)
    x = torch.cat((x0, x1, x2), dim=1)
    return x
   
def get_data(dof_list=[0,1,2], h_in_left=20, h_in_right=20, input_data=None):    
    c_data = (get_compensated_data(input_data[0]), get_compensated_data(input_data[1]), get_compensated_data(input_data[2]))
    data_piece = [None] * input_data[0].shape[0]
    for i in range(input_data[0].shape[0]):
        data_piece[i] = get_data_point(i, c_data)
    output_data = torch.cat(data_piece, dim=0)
    return output_data
    
class cnn_model:
    def __init__(self, device='cpu'):
        self.model_list = []
        for i in range(3):
            path  = root_path + '/' + 'dof_' + str(i)
            model = torch.load(path+'/'+'model', map_location=device)
            model.load_state_dict(torch.load(path+'/'+'model_parameter', map_location=device))
            self.model_list.append(model)

    def get_feedforward(self, y_des, t=0.01):
        '''
        y_des: 3 * N
        '''
        y = get_norm(np.copy(y_des))
        v = get_difference(y, t)
        a = get_difference(v, t)
        ff = np.zeros(y.shape)

        cnn_input = get_data(input_data=(y.T, v.T, a.T))

        for dof in range(3):
            model = self.model_list[dof]
            ff[dof, :] = model(cnn_input.float()).cpu().detach().numpy().flatten()
        ff = get_denorm(ff)
        ff = np.vstack((ff, np.zeros(ff.shape[1]).reshape(1, -1)))
        ff[1, :] = -ff[1, :]  # for the sign flip
        return ff