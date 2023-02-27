#!/usr/bin/env python3
'''
This script is used to read command from shared memory 
and control the robot to hit the ball.
'''
import numpy as np
import SharedArray as sa
import signal_handler
import o80
import o80_pam
import matplotlib.pyplot as plt
import time
import pickle5 as pickle
import os

# %% functions

def print_states():
    '''
    print all states from the shared memory.
    useful to keep overview over cascade and bugfix
    '''

    state_initialization = sa.attach('shm://state_initialization')
    state_predictor      = sa.attach('shm://state_predictor')
    state_estimator      = sa.attach('shm://state_estimator')
    state_launcher       = sa.attach('shm://state_launcher')
    state_ff             = sa.attach('shm://state_ff')
    state_robot          = sa.attach('shm://state_robot')

    print(' -- states: ')
    print('state_initialization: ', state_initialization)
    print('state_predictor: ', state_predictor)
    print('state_estimator: ', state_estimator)
    print('state_launcher: ', state_launcher)
    print('state_ff: ', state_ff)
    print('state_robot: ', state_robot)
    print(' -- ')

    return


# %% parameters and initialization

path                    = os.getcwd()

GLOBAL_ANGLE            = np.array([0.000000, -0.514884, -0.513349, -0.172187]) # rest position of robot arm
anchor                  = np.array([17500, 18500, 16000, 15000])    # base pressures for each DoF

pid_tracking            = np.array([[  -13000,        0,  -600],
                                    [  -80000,        0,  -600],
                                    [   -5000,   -56000,  -200],
                                    [34221.87, 16736.64, 73.24]])

pid_setting             = np.array([[-3.505924158687806e+04, -3.484022215671791e+05/5,  -5.665386729745434e+02],
                                    [-8.228984656729296e+04,   -1.304087541343074e+04,  -4.841489121599795e+02],
                                    [    -36752.24956301624,    -246064.5612272051/10,      -531.2866756516057],
                                    [ 3.422187330173758e+04, 1.673663594798479e+05/10,      73.238165769446297]])

pid_DoF4                = np.array([1.5e5, 2e4, 0])                 # PID control for last DoF 2022_10_04

frequency_backend       = 500.0
frequency_control       = 100.0
t                       = 1/frequency_control
iteration_per_command   = int(frequency_backend/frequency_control)  # backend iterations per frontend command
stop_point              = 180                                       # stop predictions and feedforward recalculations this many iterations before the end. 
                                                                    # Ideally, this is at the interception

soft_DoF1               = False                                     # option to linearize inputs to DoF1 around the interception point. Tracking is less accurate,
                                                                    # but interception may be more smooth (landing points more stable)
soft_DoF1_length        = int(5)                                    # half the length(radius) of the linearized section as number of robot iteratons

mode                    = 'ff+fb'                                   # use either feedforward ('ff'), feedback ('fb') or combined ('ff+fb') control strategy
pid                     = pid_tracking                              # use setting when using only feedback control, otherwise use tracking
factor_fb               = 1                                         # tune useage of feedback and feedforward parts
factor_ff               = 1

frontend                = o80_pam.FrontEnd("real_robot")            # connect to o80 backend


# -- connect variables to shared array memory for communication between scripts

pointer                 = sa.attach('shm://pointer')
feedforward             = sa.attach('shm://feedforward')
reference_angle         = sa.attach('shm://reference_angle')
reference_length        = sa.attach('shm://reference_length')
state_initialization    = sa.attach('shm://state_initialization')
state_vision            = sa.attach('shm://state_vision')
state_predictor         = sa.attach('shm://state_predictor')
state_estimator         = sa.attach('shm://state_estimator')
state_ff                = sa.attach('shm://state_ff')
state_robot             = sa.attach('shm://state_robot')
hit_time                = sa.attach('shm://hit_time')
ball_index              = sa.attach('shm://ball_index')
time0                   = sa.attach('shm://time0')
t_stamp                 = sa.attach('shm://t_stamp')
global_iterator         = sa.attach('shm://global_iterator')
theta                   = sa.attach('shm://theta')

if_ = 0


# -- main Loop

while not signal_handler.has_received_sigint():
    res_i           = 0                                                     # for PID, I-term
    DoF4_res_i      = 0                                                     # for PID, I-term
    angle_delta_pre = np.zeros(4).reshape(4, -1)                            # for PID, D-term
    theta_loc       = np.array(frontend.latest().get_positions())           # measured robot state
    obs_pressure    = np.array(frontend.latest().get_observed_pressures())  # measured pressure 

    iteration       = frontend.latest().get_iteration()                     # current iterations
    iteration_begin = frontend.latest().get_iteration()                     # initial iterations

    if soft_DoF1:   
        lin_diff    = 0  # linearized change per iteration
        lin_start   = 0  # set input pressure at first iteration of this section
        lin_start_i = 0  # iterations number of the first iterations of this section
    soft_DoF1_flag  = False

    theta_list      = [] # measured angles (robot state)
    reference_list  = [] # referece/target trajectory for angles
    robot_t_stamp   = [] # time stamps of robot iterations

    press_hist      = [] # observed pressures 
    press_set_hist  = [] # set/ideal pressures

    time_0          = time.time()

    # start robot when the initilization script is finished and we receive the signal to start
    while int(state_robot[0]) == 1 and state_initialization[0] == 0:

        if_ = 1

        # if option chosen: when close to interception point, linearize 
        if soft_DoF1:

            # flag whether currently in linearized setion
            soft_DoF1_flag = False

            if (np.abs(pointer[0] - hit_time[0]*100) < soft_DoF1_length+1):

                soft_DoF1_flag = True

                if lin_diff == 0:
                    lin_diff    = int(round((feedforward[0, int(pointer[0])+11] - feedforward[0, int(pointer[0])])/11))
                    lin_start   = feedforward[0, int(pointer[0])].astype(int)
                    lin_start_i = pointer[0]


        # -- PID calculation

        angle_delta = (reference_angle[:, int(pointer[0])]-theta_loc).reshape(4, -1)
        res_d = (angle_delta - angle_delta_pre) / t
        if not soft_DoF1_flag: # avoid big jumps in I-term: suspend accumulation 
            res_i += angle_delta * t

        feedback = pid[:, 0].reshape(4, -1)*angle_delta + pid[:, 1].reshape(4, -1)*res_i + pid[:, 2].reshape(4, -1)*res_d
        angle_delta_pre = np.copy(angle_delta)
        

        # -- input calculation

        if mode == 'ff+fb' or mode == 'fb+ff':
            pressure_ago = (anchor + factor_fb*feedback.flatten() + factor_ff*feedforward[:, int(pointer[0])]).astype(int)
            pressure_ant = (anchor - factor_fb*feedback.flatten() - factor_ff*feedforward[:, int(pointer[0])]).astype(int)
            
            if soft_DoF1_flag:
                pressure_ago[0] = anchor[0] + lin_diff * (pointer[0] - lin_start_i) + lin_start
                pressure_ant[0] = anchor[0] - (lin_diff * (pointer[0] - lin_start_i) + lin_start)

        elif mode == 'fb':
            pressure_ago = (anchor + feedback.flatten()).astype(int)
            pressure_ant = (anchor - feedback.flatten()).astype(int)

        elif mode == 'ff':
            pressure_ago = (anchor + feedforward[:, int(pointer[0])]).astype(int)
            pressure_ant = (anchor - feedforward[:, int(pointer[0])]).astype(int)

        press_set_hist.append((pressure_ago, pressure_ant))


        # -- PI control of DoF4
        
        diff  = theta[3] - theta_loc[3]
        DoF4_res_i = DoF4_res_i + diff*t

        pressure_ago[3] = anchor[3] + (pid_DoF4[0]*diff + pid_DoF4[1]*DoF4_res_i)
        pressure_ant[3] = anchor[3] - (pid_DoF4[0]*diff + pid_DoF4[1]*DoF4_res_i)


        # -- send commands to robot

        frontend.add_command(pressure_ago, pressure_ant,
                            o80.Iteration(iteration),
                            o80.Mode.QUEUE)

        frontend.pulse()
        frontend.add_command(pressure_ago, pressure_ant,
                             o80.Iteration(iteration+iteration_per_command-1),
                             o80.Mode.QUEUE)

        observation = frontend.pulse_and_wait()


        # -- update and save values
        
        theta_loc = np.array(observation.get_positions())
        iteration += iteration_per_command
        
        theta_list.append(theta_loc)
        reference_list.append(reference_angle[:, int(pointer[0])])
        robot_t_stamp.append(time.time() - time_0)
        press_hist.append(frontend.latest().get_observed_pressures())

        pointer[0] = pointer[0] + 1   

        if int(pointer[0]) > (int(reference_length[0])-stop_point) and int(state_ff[0])==1:
            state_ff[0]        = int(0)
            state_predictor[0] = int(0)
            print('robot: prediction and ff off!')
        if int(pointer[0]) >= int(reference_length[0]) or time.time() - time_0 > 10:
            time.sleep(2)   # make sure none of the other scripts are terminated too early and data is lost
            state_vision[0]    = int(0)
            state_estimator[0] = int(0)
            state_predictor[0] = int(0)
            state_robot[0]     = int(0)
            print('robot: robot, estimator and vision off!')
    
    if if_ == 1: # reset for next interception
        if_ = 0

        state_estimator[0] = int(0)
        state_robot[0]     = int(0)

        # save interception to data bank
        path_of_file = path + '/Data Collected while Hitting/Series 1/Iterator_{}_robot'.format(int(global_iterator[0]))
        file = open(path_of_file, 'wb')
        pickle.dump(theta_list, file, -1)
        pickle.dump(reference_list, file, -1)
        pickle.dump(robot_t_stamp, file, -1)
        pickle.dump(np.array(press_hist), file, -1)
        pickle.dump(np.array(press_set_hist), file, -1)
        file.close()

        print('-------------------------------------------------------------') 


        # -- plots, tracking error of pamy

        # iteration_end = iteration
        # iteration = iteration_begin

        # position     = np.array([])
        # pressure_ago = np.array([])
        # pressure_ant = np.array([])

        # while iteration < iteration_end:
        #     observation  = frontend.read(iteration)
        #     obs_position = np.array( observation.get_positions() )
        #     obs_pressure = np.array(observation.get_observed_pressures())
        #     pressure_ago = np.append(pressure_ago, obs_pressure[:, 0])
        #     pressure_ant = np.append(pressure_ant, obs_pressure[:, 1])
        #     position     = np.append(position, obs_position)
        #     iteration   += iteration_per_command
        # position     = position.reshape(-1, 4).T
        # pressure_ago = pressure_ago.reshape(-1, 4).T
        # pressure_ant = pressure_ant.reshape(-1, 4).T

        # t_stamp_plot = np.array(range(int(pointer[0])))/frequency_control
        # fig, axs = plt.subplots(2, 4, figsize=(16, 16))
        # for i_dof in range(4):
        #     ax = axs[0, i_dof]
        #     ax.set_xlabel(r'Time $t$ in s')
        #     ax.set_ylabel(r'Angle $\theta$ in degree')
        #     line = []
        #     line_temp, = ax.plot(t_stamp_plot, reference_angle[i_dof, 0:int(pointer[0])]*180/np.pi, 
        #                         linewidth=1.5, linestyle='--', label=r'ref')
        #     line.append( line_temp )
        #     line_temp, = ax.plot(t_stamp_plot, position[i_dof, :]*180/np.pi, 
        #                         linewidth=1.0, linestyle='-', label=r'result')
        #     line.append( line_temp )
        #     # ax.axvline((int(reference_length)-stop_point)/frequency_control, 
        #     #                          linewidth=0.5, linestyle='--')
        #     ax.axvline(hit_time, linewidth=1.0, linestyle='--')
        #     ax.grid()
        #     ax.legend(handles=line)

        #     ax = axs[1, i_dof]
        #     ax.set_xlabel(r'Time $t$ in s')
        #     ax.set_ylabel(r'Normalized inputs')
        #     line = []
        #     line_temp, = ax.plot(t_stamp_plot, feedforward[i_dof, 0:int(pointer[0])], 
        #                         linewidth=1.5, linestyle='--', label=r'ref')
        #     line.append( line_temp )
        #     ax.axvline((int(reference_length)-stop_point)/frequency_control, color='red', linewidth=0.5, linestyle='--')
        #     ax.grid()
        #     ax.legend(handles=line)
        # plt.suptitle('Tracking Error of Pamy')
        # plt.show()
