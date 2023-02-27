#!/usr/bin/env python3
'''
This script is used to detect, when a ball enters the field of vision, collect position measurements
and perform the Kalman filtering
'''
import numpy as np
import tennicam_client
import SharedArray as sa
import signal_handler
import time
import pickle5 as pickle
import torch
import matplotlib.pyplot as plt
import os

from RealBall import RealBall
# %% functions 

def plot_traj(position, t_stamp, x_kf=None):
    '''
    plot evolution of position velocity and spin.
    for position and velocity: compare measurement/finite difference to Kalman filtered value (x_kf)
    '''

    n = len(t_stamp)

    pos_x = [position[0, k] for k in range(n)]
    pos_y = [position[1, k] for k in range(n)]
    pos_z = [position[2, k] for k in range(n)]


    vx = []
    vy = []
    vz = []

    for k in range(n - 1):
        vx.append((position[0, k+1] -position[0, k]) / (t_stamp[k+1] - t_stamp[k]))
        vy.append((position[1, k+1] -position[1, k]) / (t_stamp[k+1] - t_stamp[k]))
        vz.append((position[2, k+1] -position[2, k]) / (t_stamp[k+1] - t_stamp[k]))

    if x_kf is not None:
        n_plots = 3
    else:
        n_plots = 2

    fig = plt.figure( figsize=(12, 12) )
    ax1 = plt.subplot(n_plots, 1, 1)
    ax1.plot(vx, label='x')
    ax1.plot(vy, label='y')
    ax1.plot(vz, label='z')
    ax1.set_ylim([-5, 5])
    ax2 = plt.subplot(n_plots, 1, 2, sharex=ax1)
    ax2.plot(pos_x, label='x meas')
    ax2.plot(pos_y, label='y_meas')
    ax2.plot(pos_z, label='z_meas')
    ax2.set_ylim([-5, 5])

    if x_kf is not None:

        pos_x_kf = [x_kf[k, 0, 0] for k in range(len(x_kf))]
        pos_y_kf = [x_kf[k, 1, 0] for k in range(len(x_kf))]
        pos_z_kf = [x_kf[k, 2, 0] for k in range(len(x_kf))]

        vel_x_kf = [x_kf[k, 3, 0] for k in range(len(x_kf))]
        vel_y_kf = [x_kf[k, 4, 0] for k in range(len(x_kf))]
        vel_z_kf = [x_kf[k, 5, 0] for k in range(len(x_kf))]

        omega_x_kf = [x_kf[k, 6, 0] for k in range(len(x_kf))]
        omega_y_kf = [x_kf[k, 7, 0] for k in range(len(x_kf))]
        omega_z_kf = [x_kf[k, 8, 0] for k in range(len(x_kf))]
        
        ax1.plot(vel_x_kf, label='vx kf')
        ax1.plot(vel_y_kf, label='vy kf')
        ax1.plot(vel_z_kf, label='vz kf')

        ax2.plot(pos_x_kf, label='x kf')
        ax2.plot(pos_y_kf, label='y kf')
        ax2.plot(pos_z_kf, label='z kf')

        ax3 = plt.subplot(n_plots, 1, 3, sharex=ax1)

        ax3.plot(omega_x_kf, label='omega x')
        ax3.plot(omega_y_kf, label='omega y')
        ax3.plot(omega_z_kf, label='omega z')
        ax3.legend()
        ax3.set_ylim([-0.5, 0.5])

    ax1.legend()
    ax2.legend()

    return

# %% parameters and initialization

TENNICAM_CLIENT_DEFAULT_SEGMENT_ID = "tennicam_client"
frontend = tennicam_client.FrontEnd(TENNICAM_CLIENT_DEFAULT_SEGMENT_ID)

path                = os.getcwd()

A                   = np.array([[ -0.99591269,  0.07944872, 0.04296286],
                                [ -0.07685274, -0.99529339, 0.05903159],
                                [  0.04745064,  0.0554885,  0.99733117]])   # from calibration 2022_09_22

CALIBRATION         = np.array([ 0.06727473,  0.01105588, -0.08797134])     # from calibration 2022_09_22

real_table          = np.array([0.141, 1.74, -0.441])                       # from calibration 2022_09_22

height_of_table     = 0.765  # m
height_of_ground    = real_table[2] - height_of_table + 0.01                #  add a small value distance to make sure the table impact is triggered in the state estimation
center_of_table     = real_table[0:2]

x_of_table          = np.array([-1, 0])
y_of_table          = np.array([0, -1])

p0_estimate         = np.array([0.512, 2.977, 0.0419])                      # first position measurement should be close to where we expect it to be. 
                                                                            # Otherwise wrong measurement might trigger the system. Change this value when
                                                                            # moving the launcher or changing the settings significantly.

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
ball_index      = sa.attach("shm://ball_index")
time0           = sa.attach("shm://time0")
global_iterator = sa.attach('shm://global_iterator')

if_ = 0


# -- main loop

while not signal_handler.has_received_sigint():    
    x_kf_list   = []
    k_vision[0] = int(-1)   # 
    last_id     = -1        # keep track of last ball id to not use same measurement twice

    state_predictor[0] = 0  # make sure predictor is turned off initially

    while int(state_estimator[0]) == 1:

        latest = frontend.latest()
    
        if latest.get_ball_id() != -1 and latest.get_ball_id() != last_id:  # after launching the ball and detect the ball

            k_vision_int = int(k_vision[0])

            # -- test if first position measurement is where we expect it to be, otherwise it might be a wrong measurement and we skip it
            if if_ == 0:
                ball_position = np.array(latest.get_position())
                if np.linalg.norm(ball_position - p0_estimate) > 0.3:
                    continue
                print('Start vision system')
                time_stamp_begin = latest.get_time_stamp()
                time0[0]         = time_stamp_begin
        
            ball_position   = np.array(latest.get_position())
            t_stamp_curr    = (latest.get_time_stamp()-time_stamp_begin)/1e9 # [ns] to [s]

            if abs(t_stamp_curr) > 10: # sometime the time stamps we get are wrong
                continue

            t_stamp[0, k_vision_int+1]  = t_stamp_curr
            position[:, k_vision_int+1] = A@(latest.get_position() + CALIBRATION) # calibrate position in shared memory

            if if_ == 0:
                ball_velocity = np.array([0.0, -3.9519, 3.3]) # start with a resonable estimate of the velocity to accelerate convergence of the KF
                step_length   = None
            else:
                ball_velocity = None
                step_length   = t_stamp[0, k_vision_int+1] - t_stamp[0, k_vision_int]
            
            # -- Kalman filter step 
            Ball.StateEstimation(ball_position, ball_velocity, k=k_vision_int, step_length=step_length)
            x_kf_list.append(Ball.x_meas)
            x_kf[0:9] = Ball.x_meas.flatten()[:] # save state estimate to shared memory

            # state predictions after some iterations so that the first prediction is already decent
            if state_predictor[0] == 0 and k_vision_int == 25:
                state_predictor[0] = 1
                print('start predictions')

            if if_ == 0: # activate robot and feedforward scripts
                print('activate robot and ff')
                if int(state_ff[0]) == 0:
                    state_ff[0]     = int(1)
                if int(state_robot[0]) == 0:
                    state_robot[0]  = int(1)
                if_ = 1

            k_vision[0] = int(k_vision[0] + 1)
            last_id = latest.get_ball_id()

    if if_ == 1: # reset for next interception
        if_ = 0

        # -- save interception to data bank
        path_of_file = path + '/Data Collected while Hitting/Series 1/Iterator_{}_estimator'.format(int(global_iterator[0]))
        file = open(path_of_file, 'wb')
        pickle.dump(t_stamp, file)
        pickle.dump(position, file)
        pickle.dump(x_kf_list, file)
        file.close()

        time.sleep(0.5)

        print('-------------------------------------------------------------')  

        # -- plots
    
        # k_last = len(t_stamp[0, :])
        # while t_stamp[0, k_last - 1] == 0.0:
        #     k_last = k_last - 1
        # x_kf_list = np.array(x_kf_list)
        
        # # -- Kalman filtered states
        # plot_traj(position[:, :k_last], t_stamp[0, :k_last], x_kf=x_kf_list)
        # plt.show()

        # # -- whole trajectory in 3D
        # Ball.t_stamp = t_stamp[0, :]
        # Ball.Plot3DEstimation(y_to_compare=None, y_real=position)
        # plt.show()

        # -- reinitialize
        Ball = RealBall(model=model,
                        center_of_table=center_of_table, 
                        x_of_table=x_of_table, 
                        y_of_table=y_of_table, 
                        height_of_ground=height_of_ground, 
                        height_of_table=height_of_table)

        time.sleep(3)

        t_stamp[:, :] = np.zeros((1, 1000))
        position[:, :] = np.zeros((3, 1000))