#!/usr/bin/env python3
'''
This script is used to control the launcher to shoot the ball,
when state_launcher is 1/active.
'''
from ball_launcher_beepy import BallLauncher
import time
import SharedArray as sa
import signal_handler
import time
import numpy as np
import pickle5 as pickle
import os

state_launcher  = sa.attach("shm://state_launcher")
state_vision    = sa.attach("shm://state_vision")
state_estimator = sa.attach("shm://state_estimator")
ball_index      = sa.attach("shm://ball_index")
LauncherSetting = sa.attach('shm://LauncherSetting')
global_iterator = sa.attach('shm://global_iterator')

path            = os.getcwd()

ip              = "10.42.31.174"
port            = 5555

phi             = 0.5  # from 0 to 1
theta           = 0.9  # from 0 to 1
top_left_motor  = 0.175 - 0.04 # correct internal bias of launcher wheel by subtracting the experimentally determined value (-0.04)
top_right_motor = 0.175
bottom_motor    = 0.175

while not signal_handler.has_received_sigint():
    print('launcher waiting ', state_launcher[0])
    time.sleep(0.5)
    if int(state_launcher[0]) == 1:
        print('launcher launch')
        '''
        once the ball is launched, the vision system, ball estimator,
        convolutional neural network and robot begin to work
        '''
        state_launcher[0]  = int(0)
        state_estimator[0] = int(1)

        path_of_file = path + '/Data Collected while Hitting/Series 1/Iterator_{}_launcher'.format(int(global_iterator[0]))
        file = open(path_of_file, 'wb')
        pickle.dump([phi, theta, top_left_motor, top_right_motor, bottom_motor], file)
        file.close()

        with BallLauncher(ip,port,phi,theta,top_left_motor,top_right_motor, bottom_motor, ) as client:
            time_begin = time.time()
            client.launch_ball()
            print(time.time()-time_begin)

        print('-------------------------------------------------------------')  