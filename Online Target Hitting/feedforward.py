#!/usr/bin/env python3
'''
This script is used to update the desired
trajectory in angular space and generate
corresponding feedforward inputs.
'''
import numpy as np
import SharedArray as sa
import signal_handler
import PAMY_CONFIG
from cnn import cnn_model
import time
import matplotlib.pyplot as plt
import math
import pickle5 as pickle
import pandas as pd
import os

# %% parameters and initialization

path                    = os.getcwd()

frequency_backend       = 100.0
frequency_control       = 10.0
iteration_per_command   = int(frequency_backend/frequency_control)
Geometry                = PAMY_CONFIG.build_geometry()
cnn                     = cnn_model()

# can be used for logging/plots
# angle_list              = [np.zeros((3, 800))] * 20
# polar_list              = [np.zeros((3, 800))] * 20
# ff_list                 = [np.zeros((3, 800))] * 20
# begin_list              = [None] * 20
# end_list                = [None] * 20
# idx                     = 0

# -- connect variables to shared array memory for communication between scripts

state_ff           = sa.attach('shm://state_ff')           # the state of feedforward.py
reference_length_  = sa.attach('shm://reference_length')   # the length of current reference trajectory
hit_time           = sa.attach('shm://hit_time')           # hit time
hit_position       = sa.attach('shm://hit_position')       # hit point in Cartesian space
pointer_           = sa.attach('shm://pointer')            # the current time point
reference_polar    = sa.attach('shm://reference_polar')    # the desried trajectory in polar coordinate system
velocity_polar     = sa.attach('shm://velocity_polar')     # the velocity in polar coordinate system
acceleration_polar = sa.attach('shm://acceleration_polar') # the acceleration in polar coordinate system
reference_angle    = sa.attach('shm://reference_angle')    # the desired trajectory in angular space
feedforward        = sa.attach('shm://feedforward')        # the feedforward input
ball_index         = sa.attach('shm://ball_index')         # the index of the current ball
state_predictor    = sa.attach('shm://state_predictor')
global_iterator    = sa.attach('shm://global_iterator')
theta              = sa.attach('shm://theta')
p_target           = sa.attach('shm://p_target')
v_target           = sa.attach('shm://v_target')

reference_length   = int(reference_length_[0])

if_ = 0


# -- main loop

while not signal_handler.has_received_sigint():
    
    pre_hit_time     = np.zeros(hit_time.shape)
    pre_hit_position = np.zeros(hit_position.shape)
    reference_length = int(reference_length_[0])

    while int(state_ff[0]) == 1:
        if_ = 1

        if (abs(pre_hit_time[0]-hit_time[0])>0.01) or (np.linalg.norm(pre_hit_position-hit_position, ord=2)>0.02): # recalculate when target hit point or hit time has changed

            print('feedforward GO ...')
            
            pre_hit_time     = np.copy(hit_time)
            pre_hit_position = np.copy(hit_position)
            pointer          = int(pointer_[0])

            time_begin = time.time()
            hit_angle = Geometry.EndToAngle(hit_position, frame='Cartesian')
            print('hit angle: ', hit_angle)
            (p_polar, v_polar, a_polar, p_angle, t_stamp, p_target_curr, v_target_curr) = Geometry.PathPlanning(time_point=pointer, 
                                                                                  theta=theta,
                                                                                  T_go=hit_time[0],
                                                                                  T_back=1.5, 
                                                                                  T_steady=0.2, 
                                                                                  angle=reference_angle[:, pointer],  # absolute angle
                                                                                  velocity_initial=velocity_polar[:, pointer],
                                                                                  acceleration_initial=acceleration_polar[:, pointer],
                                                                                  target=hit_angle)

            p_last_dof = np.zeros(len(t_stamp)).reshape(1, -1)
            p_angle    = np.vstack((p_angle, p_last_dof))

            p_target[:] = p_target_curr[:]
            v_target[:] = v_target_curr[:]

            '''
            update the reference trajectories in shared memory
            '''
            # update the lenght of reference trajectory in shared memory
            reference_length_[0]                            = len(t_stamp) + pointer
            reference_length                                = int(reference_length_[0])
            # overwrite the rest trajectory in shared memory
            reference_polar[:, pointer:reference_length]    = p_polar
            velocity_polar[:, pointer:reference_length]     = v_polar 
            acceleration_polar[:, pointer:reference_length] = a_polar 
            reference_angle[:, pointer:reference_length]    = p_angle
          
            ff = cnn.get_feedforward(reference_angle[0:3, 0:reference_length] - reference_angle[0:3, 0].reshape(-1, 1))
            feedforward[:, pointer:reference_length] = ff[:, pointer:reference_length]     
            print(hit_time, hit_position, time.time() - time_begin)   

            # begin_list[idx]     = int(pointer)
            # end_list[idx]       = int(len(t_stamp) + pointer)
            # angle_list[idx] = np.copy(reference_angle[0:3, :])
            # polar_list[idx] = np.copy(reference_polar[0:3, :])
            # ff_list[idx] = np.copy(feedforward[0:3, :])
            # idx                += 1

    if if_ == 1: # reset for next interception

        if_ = 0
        print('-------------------------------------------------------------')  

        # save interception to data bank
        path_of_file = path + '/Data Collected while Hitting/Series 1/Iterator_{}_feedforward'.format(int(global_iterator[0]))
        file = open(path_of_file, 'wb')
        pickle.dump(reference_polar, file, -1)
        pickle.dump(velocity_polar, file, -1)
        pickle.dump(acceleration_polar, file, -1)
        pickle.dump(reference_angle, file, -1)
        file.close()


        # -- plots
        # fig, axs = plt.subplots(3, 3, figsize=(16, 16))
        # for i_dof in range(3):
        #     line = []
        #     ax = axs[0, i_dof]
        #     for i_idx in range(idx):
        #         t_stamp = np.array(range(begin_list[i_idx], end_list[i_idx])) / 100
        #         ax.set_xlabel(r'Time $t$ in s')
        #         ax.set_ylabel(r'Angle $\theta$ in degree')
        #         line_temp, = ax.plot(t_stamp, angle_list[i_idx][i_dof, begin_list[i_idx]:end_list[i_idx]]*180/math.pi, 
        #                             linewidth=1.0, linestyle='-', label=r'idx: {}'.format(i_idx))
        #         line.append(line_temp)
        #         ax.scatter(t_stamp[0], angle_list[i_idx][i_dof, begin_list[i_idx]]*180/math.pi, s=30)
        #     ax.grid()
        #     ax.legend(handles=line)

        #     line = []
        #     ax = axs[1, i_dof]
        #     for i_idx in range(idx):
        #         t_stamp = np.array(range(begin_list[i_idx], end_list[i_idx])) / 100
        #         ax.set_xlabel(r'Time $t$ in s')
        #         ax.set_ylabel(r'Polar coordinate')
        #         line_temp, = ax.plot(t_stamp, polar_list[i_idx][i_dof, begin_list[i_idx]:end_list[i_idx]], 
        #                             linewidth=1.0, linestyle='-', label=r'idx: {}'.format(i_idx))
        #         line.append(line_temp)
        #         ax.scatter(t_stamp[0], polar_list[i_idx][i_dof, begin_list[i_idx]], s=30)
        #     ax.grid()
        #     ax.legend(handles=line)

        #     line = []
        #     ax = axs[2, i_dof]
        #     for i_idx in range(idx):
        #         t_stamp = np.array(range(end_list[i_idx])) / 100
        #         ax.set_xlabel(r'Time $t$ in s')
        #         ax.set_ylabel(r'Normalized input')
        #         line_temp, = ax.plot(t_stamp, ff_list[i_idx][i_dof, 0:end_list[i_idx]], 
        #                             linewidth=1.0, linestyle='-', label=r'idx: {}'.format(i_idx))
        #         line.append(line_temp)
        #         ax.scatter(t_stamp[0], ff_list[i_idx][i_dof, begin_list[i_idx]], s=30)
        #     ax.grid()
        #     ax.legend(handles=line)

        # plt.show()

        # angle_list = [np.zeros((3, 800))] * 20
        # polar_list = [np.zeros((3, 800))] * 20
        # ff_list = [np.zeros((3, 800))] * 20
        # length_list    = [None] * 20
        # idx            = 0
