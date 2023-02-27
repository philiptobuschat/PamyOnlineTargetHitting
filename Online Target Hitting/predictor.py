#!/usr/bin/env python3
"""
This script is used to predict the hit point
and hit position in Cartesian space
"""
import numpy as np
import tennicam_client
import SharedArray as sa
import signal_handler
import math
import time
import pickle5 as pickle
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from RealBall import RealBall

# %% parameters and initialization

path                = os.getcwd()

A                   = np.array([[ -0.99591269,  0.07944872, 0.04296286],
                                [ -0.07685274, -0.99529339, 0.05903159],
                                [  0.04745064,  0.0554885,  0.99733117]])   # from calibration 2022_09_22

CALIBRATION         = np.array([ 0.06727473,  0.01105588, -0.08797134])     # from calibration 2022_09_22

real_table          = np.array([0.141, 1.74, -0.441])                       # from calibration 2022_09_22

height_of_table     = 0.765  # m
height_of_ground    = real_table[2] - height_of_table
center_of_table     = real_table[0:2]

x_of_table          = np.array([-1, 0])
y_of_table          = np.array([0, -1])


# -- set up neural network for table impact model
# Alternatively, in RealBall a linear table impact model is implemented.
# In RealBall it can be changed which one is used.

class FcNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(6, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 4)
        self.fc4 = torch.nn.Linear(4, 4)
        self.fc5 = torch.nn.Linear(4, 3)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.fc5(x)
        return x.double()

model = FcNN()
model.load_state_dict(torch.load(path + '/Table Impact Model/TableImpact_5_mini_TanH_epoch_800'))
model.eval()

with open(path + '/Table Impact Model/TableImpact_5_normalizers', 'rb') as f:
    normalizers_inp = pickle.load(f)

Ball = RealBall(model=model, normalizers=normalizers_inp,
                center_of_table=center_of_table, 
                x_of_table=x_of_table, 
                y_of_table=y_of_table, 
                height_of_ground=height_of_ground, 
                height_of_table=height_of_table,
                delta_r=[1e-4, 1e-4, 1e-4], 
                delta_v=[1e-2, 1e-2, 1e-2], 
                delta_omega=[1e-3, 1e-3, 1e-3],
                delta_y=[1e-4, 1e-4, 1e-4],
                ini_delta_r=[1e-4, 1e-4, 1e-4],
                ini_delta_v=[1e-2, 1e-2, 1e-2])

# As an alternative to doing predictions of the ball trajectory we can use a fixed ball trajectory.
# This trajectory is the mean over many recorded ones. 
# Whether we do predictions of use the fixed trajectory is controlled in the online opt script.
with open(path + '/mean_traj', 'rb') as f: # load mean trajectory
    position_load   = np.array(pickle.load(f)).T
    t_stamp_load    = np.array(pickle.load(f))


# -- connect variables to shared array memory for communication between scripts

state_vision    = sa.attach("shm://state_vision")
state_estimator = sa.attach("shm://state_estimator")
state_predictor = sa.attach("shm://state_predictor")
state_ff        = sa.attach("shm://state_ff")
state_robot     = sa.attach("shm://state_robot")
t_stamp         = sa.attach("shm://t_stamp")
position        = sa.attach("shm://position")
k_vision        = sa.attach("shm://k_vision")
x_kf            = sa.attach("shm://x_kf")
hit_position    = sa.attach("shm://hit_position")
hit_time        = sa.attach("shm://hit_time")
ball_index      = sa.attach("shm://ball_index")
time0           = sa.attach("shm://time0")
TargetY         = sa.attach('shm://TargetY')
Setting         = sa.attach('shm://Setting')
theta           = sa.attach('shm://theta')
global_iterator = sa.attach('shm://global_iterator')
use_mean_traj   = sa.attach('shm://use_mean_traj')
hit_missed      = sa.attach('shm://hit_missed')
pred_time       = sa.attach('shm://pred_time')

if_ = 0

# -- main loop

while not signal_handler.has_received_sigint():    
    predictions     = [] # whole predicted trajectories
    hit_points      = [] # calculated interception points
    hit_times       = [] # calculated interception times

    hit_points_used = [] # 
    hit_times_used  = []

    k_predictions   = []
    k_vision_last   = 0
    while int(state_predictor[0]) == 1:

        if if_ == 0:
            if_ = 1

        if t_stamp[0, int(k_vision[0])] > 1.5: # past time is larger than prediciton horizon. turn off predictor
            state_predictor[0] = 0

        # when close to the interception, do no more new predictions to avoid last minute changes in robot the trajectory.
        # This makes interception more smooth and landing points more stable
        if hit_time[0] - max(t_stamp.flatten()) < pred_time: 
            # print('no more predictions')
            time.sleep(0.05)
            continue

        if int(k_vision[0]) == k_vision_last:
            #  print('current prediction done already')
            continue
        
        k_vision_last = k_vision.copy()[0]

        # fill Ball instance with necessary information about the current state
        Ball.t_stamp   = t_stamp[0, :int(k_vision[0])]
        Ball.position  = position
        Ball.time_past = t_stamp[0, int(k_vision[0])]
        Ball.x_meas = x_kf

        # find target y, i.e., the location of the virtual hitting plane
        l_1=0.3768; l_2=0.4038
        TargetY[0] = math.sin(theta[0]) * math.sin(theta[1]) * l_1 + math.sin(theta[0]) * math.sin(theta[1] + theta[2]) * l_2
        print('target Y : ', -TargetY[0]) # minus due to calibration convention for axis directions
        print('time past: ', t_stamp[0, int(k_vision[0])])
        print('do prediciton from state: ', x_kf)

        # -- make new prediction of ball trajectory
        (trajectory, t_stamp_pred, impact_time_list, impact_type) = Ball.Prediction(x=x_kf, time_past=t_stamp[0, int(k_vision[0])])

        if use_mean_traj[0] == 2: # pure interception, no target hitting (--> fixed TargetY/virtual hitting plane)
            (hit_position_, hit_time_, traj) = Ball.GetHitPoint(traj=trajectory, t=t_stamp_pred, hit_point_=np.copy(hit_position), hit_time_=hit_time[0], TargetY=-0.1)

        if use_mean_traj[0] == 1:
            (hit_position_, hit_time_, traj) = Ball.GetHitPoint(traj=position_load, t=t_stamp_load, hit_point_=np.copy(hit_position), hit_time_=hit_time[0], TargetY=-TargetY[0])

        if use_mean_traj[0] == 0:
            # use only points after the impact on table as possible interception points, otherwise we might get weird solutions
            # calibrate those to have the resulting interception points in relation to the robot position to make them more intuitive
            trajectory_cut = []
            t_stamp_cut = []
            for k in range(len(t_stamp_pred)):
                if len(impact_type) > 0:
                    if impact_type[0] == 'Table':
                        if t_stamp_pred[k] < impact_time_list[0] + 0.1:
                            continue

                trajectory_cut.append(A@(trajectory[:, k] + CALIBRATION))
                t_stamp_cut.append(t_stamp_pred[k])

            trajectory_cut = np.array(trajectory_cut).T
            t_stamp_cut = np.array(t_stamp_cut)

            if len(t_stamp_cut) > 1:
                (hit_position_, hit_time_, traj) = Ball.GetHitPoint(traj=trajectory_cut, t=t_stamp_cut, hit_point_=np.copy(hit_position), hit_time_=hit_time[0], TargetY=-TargetY[0])

        print('prediction: ', hit_position, hit_time)

        predictions.append(trajectory)
        hit_points.append(hit_position_)
        hit_times.append(hit_time_)
        k_predictions.append(int(k_vision[0]))

        # take a rolling mean of predicted interception points to reduce abrupt changes which might produce sudden movements of the robot arm
        if x_kf[1] < 1 and x_kf[5] > 0: # after impact
            rolling_mean_length = 2
        else:
            rolling_mean_length = 3

        if len(hit_times) <= rolling_mean_length: # only do rolling averaging if enough predicted hit points available 
            hit_position[0] = np.mean([i[0] for i in hit_points])
            hit_position[1] = np.mean([i[1] for i in hit_points])
            hit_position[2] = np.mean([i[2] for i in hit_points])
            hit_time[0] = round(np.mean(hit_times), 2)
        else:
            hit_position[0] = np.mean([i[0] for i in hit_points[len(hit_times)-rolling_mean_length:]])
            hit_position[1] = np.mean([i[1] for i in hit_points[len(hit_times)-rolling_mean_length:]])
            hit_position[2] = np.mean([i[2] for i in hit_points[len(hit_times)-rolling_mean_length:]])
            hit_time[0] = round(np.mean(hit_times[len(hit_times)-rolling_mean_length:]), 2)

        hit_points_used.append(hit_position.copy())
        hit_times_used.append(hit_time.copy())

    if if_ == 1: # reset for next interception
        if_ = 0

        # -- save interception to data bank
        path_of_file = path + '/Data Collected while Hitting/Series 1/Iterator_{}_predictor'.format(int(global_iterator[0]))
        file = open(path_of_file, 'wb')
        pickle.dump(hit_points, file)
        pickle.dump(hit_times, file)
        file.close()

        print('-------------------------------------------------------------')  
        print(len(predictions), ' predictions done')

        # -- plots
        
        # -- all prediction in 3D plot
        # Ball.Plot3DEstimation(y_to_compare=predictions, y_real=position)

        # -- hit points in 3D plot
        # fig = plt.figure( figsize=(8, 8) )
        # ax = plt.subplot(111, projection='3d')
        # ax.spines['bottom'].set_linewidth(1.5)
        # ax.spines['top'].set_linewidth(1.5)
        # ax.spines['left'].set_linewidth(1.5)
        # ax.spines['right'].set_linewidth(1.5)
        # hit_points = np.array(hit_points)
        # ax.scatter(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], c='black', s=5)
        # ax.plot3D(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2],'grey', linewidth=0.1, label=r'real')
        # plt.show()

        # -- hit points in 3D subplots per coordinate
        # hit_points = np.array(hit_points)
        # hit_points_used = np.array(hit_points_used)
        # fig, axs = plt.subplots(3)
        # axs[0].scatter(k_predictions, hit_points[:, 0])
        # axs[0].scatter(k_predictions, hit_points_used[:, 0])
        # axs[0].set_title('hit points x')
        # axs[1].scatter(k_predictions, hit_points[:, 1])
        # axs[1].scatter(k_predictions, hit_points_used[:, 1])
        # axs[1].set_title('hit points y')
        # axs[2].scatter(k_predictions, hit_points[:, 2])
        # axs[2].scatter(k_predictions, hit_points_used[:, 2])
        # axs[2].set_title('hit points z')
        # plt.show()

        # time.sleep(5)

        # -- showing some plots, only if ball was missed
        # if hit_missed[0] == 1:
        # #     hit_missed[0] = 0
        #     cal_pos = []
        #     for k in range(len(t_stamp.flatten())):
        #         cal_pos.append(np.linalg.inv(A)@position.copy()[:, k] - CALIBRATION)
        #     cal_pos = np.array(cal_pos).T

        #     Ball.t_stamp   = t_stamp[0, :int(k_vision[0])]
        #     Ball.position  = position
        #     Ball.Plot3DEstimation(y_to_compare=predictions, y_real=cal_pos)

        #     hit_points = np.array(hit_points)
        #     hit_points_used = np.array(hit_points_used)

        #     fig, axs = plt.subplots(3)
        #     axs[0].scatter(k_predictions, hit_points[:, 0])
        #     axs[0].scatter(k_predictions, hit_points_used[:, 0])
        #     axs[0].set_title('hit points x')
        #     axs[1].scatter(k_predictions, hit_points[:, 1])
        #     axs[1].scatter(k_predictions, hit_points_used[:, 1])
        #     axs[1].set_title('hit points y')
        #     axs[2].scatter(k_predictions, hit_points[:, 2])
        #     axs[2].scatter(k_predictions, hit_points_used[:, 2])
        #     axs[2].set_title('hit points z')

        #     fig = plt.figure()
        #     plt.scatter(k_predictions, hit_times)
        #     plt.scatter(k_predictions, hit_times_used)
        #     plt.title('hit times')

        #     plt.show()

        # -- reinitializing
        Ball = RealBall(model=model,
                        center_of_table=center_of_table, 
                        x_of_table=x_of_table, 
                        y_of_table=y_of_table, 
                        height_of_ground=height_of_ground, 
                        height_of_table=height_of_table)
