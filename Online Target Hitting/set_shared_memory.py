#!/usr/bin/env python3
'''
This script is used to create all
the necessary shared memory.
 - state_initialization: 0 or 1, when it is 1, then execute initilization.py 
 - state_launcher:       0 or 1, when it is 1, then execute launcher.py
 - state_vision:         0 or 1, when it is 1, then execute vision_system.py
 - state_estimator:      0 or 1, when it is 1, then execute ball_estimator.py
 - state_ff:             0 or 1, when it is 1, then execute feedforward.py
 - state_robot:          0 or 1, when it is 1, then execute robot.py
 
 - pointer:  point to the current time index
 - reference_polar: the reference trajectory in polar coordinate system, it should be absolute value
                    in dimension 3 * N
 - velocity_polar:  the velocity in polar coordinate system in dimension 3 * N
 - acceceleration_polar: the acceleration in polar coordinate system in dimension 3 * N
 
 - reference_angle: the reference trajectory in angular space, it should be relative value, in dimension
                    4 * N
 - feedforward: the feedforward inputs, in dimension 4 * N

 - hit_point: the predicted hit point in Cartesian space, [x, y, z]
 - hit_time: the predicted time to hit the ball in s
 
 - ball_index: index of ball, when recording several balls

'''
import SharedArray as sa
import numpy as np
import os
# %%
N = 800
root = '/dev/shm/'

name_list = ['state_initialization',  
                   'state_launcher',  
                     'state_vision',  
                  'state_estimator',   
                  'state_predictor',
                         'state_ff',
                      'state_robot',
                          'pointer',
                  'reference_polar', 
                   'velocity_polar',
               'acceleration_polar',
                  'reference_angle',
                 'reference_length',     
                      'feedforward',   
                     'hit_position', 
                         'hit_time',      
                       'ball_state',
                       'ball_index',
                            'time0',
               'state_combine_data',
                      'online_iter',
                          't_stamp',
                         'position',
                             'x_kf',
                         'k_vision',
                    'HitVel_factor',
                          'TargetY',
                  'LauncherSetting',
                          'Setting',
                            'theta',
                         'p_target',
                         'v_target',
                  'global_iterator',
                    'use_mean_traj',
                       'hit_missed',
                        'pred_time',
                      'past_impact',
                          'OVR_inp',
                        'theta_OVR']

shape_list = [1, 
              1,   
              1, 
              1,
              1,
              1,
              1, 
              1,
              (3, N), 
              (3, N),
              (3, N),
              (4, N), 
              1,
              (4, N),
              3,
              1,
              6,
              1,
              1,
              1,
              1,
              (1, 1000),
              (3, 1000),
              9,
              1,
              1,
              1, 
              1,
              1, 
              7,
              3,
              3,
              1,
              1, 
              1,
              1,
              1,
             16,
              7]

def file_exist(file_name):
    return os.path.exists(root + file_name)

def create_file(file_name, file_shape):
    if file_exist(file_name):
        sa.delete("shm://" + file_name)
        x = sa.create(file_name, file_shape, dtype=float)
        x = sa.attach(file_name)
    else:
        x = sa.create(file_name, file_shape, dtype=float)
    x[0] = 0.0


for i in range(len(name_list)):
    create_file(name_list[i], shape_list[i])
    # print('create file ', name_list[i], ' with size ', shape_list[i])