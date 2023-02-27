'''
This script is used to train and validate the black-box model for predicting landing points
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
import time
import pickle5 as pickle
import math
import os
import random
import matplotlib.patches as patches
import tikzplotlib as tpl

from RacketHit import HitRecording
from RealBall import RealBall

# %%

path = os.getcwd()

# all of the following have already been carried out previously but can be redone if necessary
# to redo section set it to True (OVERWRITES CURRENT FILES)
# if set to False, the results from previous run are loaded
generate_normalizers        = False                                     # normalizers for inputs of neural ent
generate_resampling_split   = False                                     # split of used/unused data points for undersampling
generate_train_test_split   = False                                     # split of training/test data
training                    = False                                     # train (includes total validation) the neural net
validation                  = False                                     # validate on some examples 

cuda                        = False                                     # use cuda

dataset_plots               = True
plot_losses                 = True

class NN_mini(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 4)
        self.fc4 = torch.nn.Linear(4, 4)
        self.fc5 = torch.nn.Linear(4, 2)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.fc5(x)
        return x.double()

def train(model, train_dl, test_ds, train_size, test_size, epochs=10):
    #optim = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    model.train()

    training_losses = [validation_loss(model, train_ds)]
    validation_losses = [validation_loss(model, test_ds)]

    for epoch in range(epochs):
        if epoch%10==0:
            print('---- Epoch {} ---- training/validation losses: {}, {}'.format(epoch, training_losses[-1], validation_losses[-1]))

        if epoch%1000==0 and epoch>0:
            torch.save(model.state_dict(), path+'/Model Saves/RacketImpact_13_Resamp_epoch_{}'.format(epoch))

        if epoch%100==0:
            distances_hit_pred = []
            for inp, label in test_ds:
                out = model(inp.float()).detach().numpy()
                dist = ( (label[0]-out[0])**2 + (label[1]-out[1])**2 )**0.5
                distances_hit_pred.append(dist)
            print('Distance Error: mean = {}, std = {}'.format(np.mean(distances_hit_pred), np.std(distances_hit_pred)))

        training_loss_curr = 0
        for i, data in enumerate(train_dl):
            inputs, labels = data
            out = model(inputs.float())
            optim.zero_grad()
            loss = criterion(out, labels)
            training_loss_curr += loss
            loss.backward()
            optim.step()
        training_losses.append(training_loss_curr/train_size)
        validation_losses.append(validation_loss(model, test_ds))

    plt.figure()
    plt.plot([i.detach().numpy() for i in training_losses], label='training loss')
    plt.plot([i.detach().numpy() for i in validation_losses], label='validation loss')
    plt.xlabel('step')
    plt.ylabel('MSE')
    plt.legend()

    return training_losses, validation_losses

def validation_loss(model, test_ds):
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    for data in test_ds:
        inputs, labels = data
        out = model(inputs.float())
        total_loss += criterion(out, labels)
    model.train()


    return total_loss / len(test_ds)

ideal_table = np.array([0.141, 1.74, -0.441]) # from calibration 2022_09_22

height_of_table = 0.765  # m
height_of_ground = ideal_table[2] - height_of_table # m, add a value distance to make sure the table impact is triggered in the state estimation
center_of_table = ideal_table[0:2]
half_width_of_table = 1.525/2
half_length_of_table = 2.74/2

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

with open(r'/home/philip/Desktop/Racket Impact NN/Interception_Data_Continous/Series 13/RacketImpact_13_X_and_Y_simpler', 'rb') as f:
    X = np.array(pickle.load(f))
    Y = np.array(pickle.load(f))


if generate_train_test_split:

    train_size = int(0.8 * len(X))
    test_size = len(Y) - train_size
    split = ['train'] * train_size + ['test'] * test_size
    random.shuffle(split)
    with open(path+'/train_test_split', 'wb') as f:
        pickle.dump(split, f)
else:
    with open(path+'/train_test_split', 'rb') as f:
        split = pickle.load(f)

X_train = np.array([inp for i, inp in enumerate(X) if split[i] == 'train'])
X_test  = np.array([inp for i, inp in enumerate(X) if split[i] == 'test'])
Y_train = np.array([out for i, out in enumerate(Y) if split[i] == 'train'])
Y_test  = np.array([out for i, out in enumerate(Y) if split[i] == 'test'])


# -- Normalize Inputs

if generate_normalizers:
    normalizers = []
    for i in range(len(X_train[0])):
        mean = np.mean(X_train[:, i])
        std = np.std(X_train[:, i])
        X_train[:, i] = ( X_train[:, i] - mean ) / std
        normalizers.append((mean, std))
    with open(path+'/RacketImpact_13_normalizers', 'wb') as f:
        pickle.dump(normalizers, f)
else:
    with open(path+'/RacketImpact_13_normalizers', 'rb') as f:
        normalizers = pickle.load(f)
    for i in range(len(X_train[0])):
        mean = normalizers[i][0]
        std = normalizers[i][1]
        X_train[:, i] = ( X_train[:, i] - mean ) / std
    for i in range(len(X_test[0])):
        mean = normalizers[i][0]
        std = normalizers[i][1]
        X_test[:, i] = ( X_test[:, i] - mean ) / std

n = len(X_train)

if dataset_plots:

    plt.figure(figsize=(8, 8))
    plt.scatter(Y_train[:, 0], -Y_train[:, 1], alpha=0.3, s=20)
    plt.scatter([-100], [-100], alpha=1, s=20, color=colors[0], label='landing points')
    plt.xlabel('$x$ [m]', fontsize=16)
    plt.ylabel('$y$ [m]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Raw Dataset')
    plt.ylim([0, 6])
    plt.xlim([-3, 3])
    rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=2, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label='table')
    plt.gca().add_patch(rect)
    plt.legend(fontsize=16)


# Resampling



if generate_resampling_split:

    x_impacts = Y_train[:, 0]
    y_impacts = Y_train[:, 1]

    print(x_impacts)

    H, xedges, yedges = np.histogram2d(x_impacts, y_impacts, bins=[100, 100])

    split = []

    for i in range(n):
        x_bin = 0
        for i_xedge in range(len(xedges)-1):
            if x_impacts[i] >= xedges[i_xedge] and x_impacts[i] < xedges[i_xedge + 1]:
                x_bin = i_xedge
                break
        y_bin = 0
        for i_yedge in range(len(yedges)-1):
            if y_impacts[i] >= yedges[i_yedge] and y_impacts[i] <= yedges[i_yedge + 1]:
                y_bin = i_yedge
                break
        
        if H[x_bin, y_bin] == 0:
            split.append(False)
            continue

        p = 1 / H[x_bin, y_bin] # porbability of using current data points proportional to 1/(how many data points in this bin)

        split.append(np.random.choice(a=[True, False], p=[p, 1-p]))

        if i%1000==0:
            print('i = {}, n = {} points used'.format(i, len([i for i in split if i])))

        with open(r'/home/philip/Desktop/Racket Impact NN/Interception_Data_Continous/Series 13/resampling_split', 'wb') as f:
            pickle.dump(split, f)

else:
    with open(r'/home/philip/Desktop/Racket Impact NN/Interception_Data_Continous/Series 13/resampling_split', 'rb') as f:
        split = pickle.load(f)

X_train_resamp = torch.DoubleTensor([inp for i, inp in enumerate(X_train) if split[i]])
X_test = torch.DoubleTensor(X_test)
Y_train_resamp = torch.DoubleTensor([out for i, out in enumerate(Y_train) if split[i]])
Y_test = torch.DoubleTensor(Y_test)

train_ds = TensorDataset(X_train_resamp, Y_train_resamp)
test_ds = TensorDataset(X_test, Y_test)

train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=10)

# X_resamp = [X[i] for i in range(n) if split[i]]
# Y_resamp = [Y[i] for i in range(n) if split[i]]

# X = torch.DoubleTensor(X_train_resamp)
# Y = torch.DoubleTensor(Y_train_resamp)

if dataset_plots:
    plt.figure(figsize=(8, 8))
    plt.scatter(Y_train_resamp[:, 0], -Y_train_resamp[:, 1], alpha=0.3, s=20)
    plt.scatter([-100], [-100], alpha=1, s=20, color=colors[0], label='landing points')
    plt.xlabel('$x$ [m]', fontsize=16)
    plt.ylabel('$y$ [m]', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Undersampled Dataset')
    plt.ylim([0, 6])
    plt.xlim([-3, 3])
    rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=2, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label='table')
    plt.gca().add_patch(rect)
    plt.legend(fontsize=16)


# ---- Training

if cuda:
    device = torch.device('cuda')

model = NN_mini()
if cuda:
    model.to(device)
    X.to(device)
    Y.to(device)


if training:

    train_size = len(X_train_resamp)
    test_size = len(Y_test)

    print('training DS size: ', train_size)
    print('test DS size: ', test_size)

    training_losses, validation_losses = train(model, train_dl, test_ds, train_size, test_size, epochs=2001)

    distances_hit_pred = []
    for inp, label in test_ds:
        out = model(inp.float()).detach().numpy()
        dist = ( (label[0]-out[0])**2 + (label[1]-out[1])**2 )**0.5
        distances_hit_pred.append(dist)

    with open(r'/home/philip/Desktop/Racket Impact Model NN Sampling/13_losses_Resamp', 'wb') as f:
        pickle.dump(training_losses, f)
        pickle.dump(validation_losses, f)
        pickle.dump(train_ds, f)
        pickle.dump(test_ds, f)
        pickle.dump(distances_hit_pred, f)

    print('validation distance errors mean = {}, std = {}'.format(np.mean(distances_hit_pred), np.std(distances_hit_pred)))

    plt.show()

else:

    path = r'/home/philip/Desktop/Racket Impact Model NN Sampling/Model Saves/RacketImpact_13_Resamp_epoch_2000'

    model.load_state_dict(torch.load(path))

    with open(r'/home/philip/Desktop/Racket Impact Model NN Sampling/13_losses_Resamp', 'rb') as f:
        training_losses = pickle.load(f)
        validation_losses = pickle.load(f)
        train_ds = pickle.load(f)
        test_ds = pickle.load(f)
        distances_hit_pred = pickle.load(f)
        

errors_NN   = []
actual_list = []
pred_list   = []

if validation:

    it = 0

    for inp, label in test_ds:

        it += 1


        pred_NN = model(inp.float()).detach().numpy()
        errors_NN.append(np.mean([(label[j] - pred_NN[j])**2 for j in range(2)]))

        if it < 50:
            pred_list.append(pred_NN)
            actual_list.append(label)

        if it < 5:
            print('label: ', label)
            print('pred NN: ', pred_NN)
            print('error = ', errors_NN[-1])
    
    print('MSE NN: ', np.mean(errors_NN))

    plt.figure()
    plt.scatter([i[0] for i in actual_list], [i[1] for i in actual_list], label='actual impact')
    plt.scatter([i[0] for i in pred_list], [i[1] for i in pred_list], label='NN pred')
    plt.show()

if plot_losses:

    plt.figure(figsize=(11, 8))
    plt.plot([i.detach().numpy() for i in training_losses[:500]], label='training', linewidth=2)
    plt.plot([i.detach().numpy() for i in validation_losses[:500]], label='validation', linewidth=2)
    plt.xlabel('epoch [-]', fontsize=16)
    plt.ylabel('MSE [m²]', fontsize=16)
    plt.yscale('log')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=16)

if dataset_plots or plot_losses:
    plt.show()