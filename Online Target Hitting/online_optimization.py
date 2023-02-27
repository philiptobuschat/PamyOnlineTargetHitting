#!/usr/bin/env python3
import numpy as np
import SharedArray as sa
import signal_handler
import time
import pickle5 as pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

from TargetHitting import TargetHitting
from RacketHit import HitRecording

# ---- Setup

path            = os.getcwd()

A               = np.array([[ -0.99591269,  0.07944872,  0.04296286],
                            [ -0.07685274, -0.99529339,  0.05903159],
                            [  0.04745064,   0.0554885,  0.99733117]])  # from calibration 2022_09_22

CALIBRATION     = np.array([ 0.06727473,  0.01105588, -0.08797134])     # from calibration 2022_09_22

# -- connect variables to shared array memory for communication between scripts

state_initialization  = sa.attach('shm://state_initialization')
state_launcher        = sa.attach('shm://state_launcher')
state_estimator       = sa.attach('shm://state_estimator')
state_robot           = sa.attach('shm://state_robot')
ball_index            = sa.attach("shm://ball_index")
online_iter           = sa.attach("shm://online_iter")
Setting               = sa.attach('shm://Setting')
theta                 = sa.attach('shm://theta')
t_stamp               = sa.attach('shm://t_stamp')
position              = sa.attach('shm://position')
time0                 = sa.attach('shm://time0')
hit_position          = sa.attach("shm://hit_position")
hit_time              = sa.attach("shm://hit_time")
p_target              = sa.attach('shm://p_target')
v_target              = sa.attach('shm://v_target')
TargetY               = sa.attach('shm://TargetY')
global_iterator       = sa.attach('shm://global_iterator')
use_mean_traj         = sa.attach('shm://use_mean_traj')
hit_missed            = sa.attach('shm://hit_missed')
pred_time             = sa.attach('shm://pred_time')


# -- settings

mode = 'NN'                                             # choose gradient approximation method
                                                        # 'NN':                 regression
                                                        # 'Physical':           first principle
                                                        # 'Regression':         simplified regression
                                                        # 'simple Physical':    simplified first principle

use_mean_traj[0]        = int(0)                        # 0: current traj for target hitting, 1: fixed mean trajectory for target hitting, 2: current traj for pure interception
fix_initial_guess       = False
iter_max                = int(input('iter max = '))    

pred_time[0]            = 0.4                           # [s], how long before predicted interception do we stop make new trajectory predictions
                                                        # small value -> reliable interception
                                                        # large value -> smooth interception

constant_policy         = False                         # option to not use online potimizatio but only do repeated interception with a constant policy
print_land_point_stats  = True                          # print at each iteration statistics of all landing points after each iteration

if mode == 'Physical':
    step_ini    = 0.1                                   # step length of initial guess calculation
    k_max_ini   = 100                                   # max iteration number of initial guess calculation
    step0       = 0.05                                  # step length of first online opt iteration. following iterations generally have diminishing step lengths, which can be specified later
elif mode == 'NN':
    step_ini    = 1e-2
    k_max_ini   = 100
    step0       = 0.05
elif mode == 'Regression':
    step_ini    = 1e-2
    k_max_ini   = 100
    step0       = 0.15*3
elif mode == 'simple Physical':
    step_ini    = 1e-2
    k_max_ini   = 100
    step0       = 0.15

target_pos_1    = np.array([ 0.747,  2.78, -0.438])     # mark 1                    (far end of opponent table side)
target_pos_2    = np.array([ 0.164,  2.79,  -0.44])     # mark 2                    (far end of opponent table side)
target_pos_3    = np.array([-0.413,  2.78, -0.432])     # mark 3                    (far end of opponent table side)
target_pos_4    = np.array([ 0.122, 0.888,  -0.44])     # mark 4                    (center of robot table side)
target_pos_5    = np.array([ 0.747,  1.78, -0.438])     # mark 1 - 1.0m in y        (near end of opponent table side)
target_pos_6    = np.array([ 0.164,  1.79,  -0.44])     # mark 2 - 1.0m in y        (near end of opponent table side)
target_pos_7    = np.array([-0.413,  1.78, -0.432])     # mark 3 - 1.0m in z        (near end of opponent table side)
target_pos_8    = np.array([  -1.0,  2.78, -0.432])     # off table to the left     (as an extreme target choice)
target_pos_9    = np.array([-0.481,   3.0,  -0.44])     # left corner               (edge of table)


targets_pos     = [target_pos_1, target_pos_2, target_pos_3, 
                   target_pos_4, target_pos_5, target_pos_6, 
                   target_pos_7, target_pos_8, target_pos_9]

target_vel  = np.zeros(3)                                # target state does not include velocity
targets     = [np.hstack(((A@(i+CALIBRATION))[0:2], target_vel)) for i in targets_pos]

target_Init = targets[1]    # target choice for initial guess calculation can be different than choice of target afterwards
target      = targets[1]

# load mean trajectory
if use_mean_traj[0] == 1:
    with open(path+'/mean_traj', 'rb') as f: 
        position_load    = np.array(pickle.load(f)).T
        t_stamp_load     = np.array(pickle.load(f))
else:
    position_load   = None
    t_stamp_load    = None


# ---- Initialization

TargetHitting = TargetHitting(path=path, target=target_Init, position=position_load, t_stamp=t_stamp_load)

# -- calculate initial guess
if not fix_initial_guess:
    TargetHitting.InitialSettings(mode=mode, step_length=step_ini, tol=1e-2, k_max=k_max_ini, penalize_target_velocity=False)
    theta[:] = TargetHitting.theta # update calculated initial guess for theta in shared memory

    TargetHitting.theta[5] = 0                                                      # set interception velocities to zero
    TargetHitting.theta[6] = 0                                                      # set interception velocities to zero

# -- Initialize Theta Manually:

# values used for different initial guess experiment
initial_thetas = [                                                                  # six targets that form roughly a circle around the center of the opponents table side
    np.array([ -0.05,  0.709,  1.56, -0.05,  6,  0.,  0.]),                         # right, medium
    np.array([   0.1,  0.709,  1.56,   0.2,  6,  0.,  0.]),                         # right, far
    np.array([   0.4,  0.709,  1.56,   0.2,  6,  0.,  0.]),                         # center, far
    np.array([   0.7,  0.709,  1.56,   0.0,  6,  0.,  0.]),                         # left, medium 
    np.array([   0.6,  0.709,  1.56,  -0.2,  6,  0.,  0.]),                         # left, short
    np.array([   0.1,  0.709,  1.56, -0.25,  6,  0.,  0.])                          # right, short
]

if fix_initial_guess:
    theta[:] = np.array([ 0.5, 0.709, 1.56, -0.1, 6, 0., 0.])                       # update theta directly in shared memory 
    # theta[:] = np.array([ 0.32, 0.709, 1.56, 0.05, 6, 0., 0.])                    # roughly target 2

TargetHitting.theta = theta.copy()
TargetHitting.theta0 = theta.copy()
TargetHitting.target = target

print('initial theta: ', theta)

online_iter[0]  = int(0)                                                            # iteration number
missed          = 0                                                                 # count missed balls

p_target_hist   = []                                                                # target hit point cylinder coords (from minimum jerk principle setting)
v_target_hist   = []                                                                # target hit velocity cylinder coords (from minimum jerk principle setting)
TargetY_hist    = []                                                                # target y coord, local of virtual hitting plane

step_lengths    = []                                                                # history of step lengths

hit_missed[0]   = 0                                                                 # flag in shared memory whether last ball was missed


while online_iter[0] < iter_max:

    # -- perform the interception

    # load and update global iterator for indexing in data bank
    path_globaliterator = path+'/Data Collected while Hitting/GloablIterator'
    with open(path_globaliterator, 'rb') as file:
        current_global_iterator = pickle.load(file)
    with open(path_globaliterator, 'wb') as file:
        current_global_iterator += 1
        pickle.dump(current_global_iterator, file)
    global_iterator[0] = current_global_iterator                                    # update iterator in shared memory so all files will be saved with the correct index

    # start cascade with activation the initilization script
    print()
    print('theta = ', theta)
    state_initialization[0] = 1                 
    time0[0]                = time.time()

    # wait until hitting cascade is finished    
    states = [state_initialization, state_launcher, state_robot, state_estimator]
    while 1 in states:
        time.sleep(0.05)

    # save this run's ball position measurements before they are reset in shared memory
    position_last   = position.copy() 
    t_stamp_last    = t_stamp.copy()

    # remove zeros at the end of the measurements
    last_measurement = len(t_stamp.flatten())-1
    while np.count_nonzero(position_last[:, last_measurement]) == 0 and last_measurement >= 0:
        last_measurement -= 1

    position_last   = position_last[:, :last_measurement]
    t_stamp_last    = t_stamp_last[0, :last_measurement]

    # save values
    p_target_hist.append(p_target.copy())
    v_target_hist.append(v_target.copy())
    TargetY_hist.append(TargetY.copy())


    # -- analyze the interception
    
    # wait so files can be saved before trying to read them
    time.sleep(0.2) 

    # use HitRecording class for data analysis
    Hit = HitRecording(index=int(global_iterator[0]), path=path+'/Data Collected while Hitting/Series 1/', load_prints=True, rho_limit=0.95)
    if not Hit.load: # this means the interception was not successful or the data has problem (might happen when the position measurements around table/racket impact are corrupt or otherwise occluded)
        online_iter[0]  += 1
        missed          += 1
        iter_max        += 1                                        # when interception failed, do one more in total
        if len(TargetHitting.theta_history) > 0:
            TargetHitting.theta = TargetHitting.theta_history[-1]
        # hit_missed[0]   = 1                                       # only needed when used in other scripts as well
        # Hit.PlotRobotStates()                                     # can be useful when tryping to find out why interception failed
        continue                                                    # do not update policy based on a failed interception

    # print('theta REAL: ', Hit.robot_angles, Hit.robot_velocities)
    # print('theta0 dot REF : ', Hit.reference_velocities[0])         # difference between ref and real can be interesting, 
    # print('theta0 dot REAL: ', Hit.robot_velocities[0])             # especially when using the soft_DoF1 option in the robot.py script


    # -- perform online optimization iteration

    step_length = step0 / ((online_iter[0]-missed)+1)**0.5
    step_lengths.append(step_length)

    # when repeating same policy withouot update, save theta before and overwrite it again after using the commands below
    # but either way carry out TargetHitting.StepUpdate because it updates import other parameters as well

    if constant_policy:
        theta_save          = theta.copy()

    TargetHitting.StepUpdate(position_last, t_stamp_last, mode=mode, step_length=step_length, penalize_target_velocity=False, update_traj=False)

    if constant_policy:
        theta[:]            = theta_save[:]
        TargetHitting.theta = theta[:]                              # when using constant policy, update value in TargetHitting
    else:
        theta[:]            = TargetHitting.theta                   # when doing online optimization, update value in shared memory

    if print_land_point_stats:
        impact_mean_x   = np.mean([i[0] for i in TargetHitting.impact_history])
        impact_mean_y   = np.mean([i[1] for i in TargetHitting.impact_history])
        impact_std_x    = np.std([i[0] for i in TargetHitting.impact_history])
        impact_std_y    = np.std([i[1] for i in TargetHitting.impact_history])

        print('impact stats: ', impact_mean_x, impact_mean_y, impact_std_x, impact_std_y)

    print(' {}/{} balls successfully intercepted'.format(online_iter[0]-missed, online_iter[0]))

    online_iter[0] += 1

print()
print('---- iteration finished ----')
print()

impact_mean_x   = np.mean([i[0] for i in TargetHitting.impact_history])
impact_mean_y   = np.mean([i[1] for i in TargetHitting.impact_history])
impact_std_x    = np.std([i[0] for i in TargetHitting.impact_history])
impact_std_y    = np.std([i[1] for i in TargetHitting.impact_history])

print('impact stats: ', impact_mean_x, impact_mean_y, impact_std_x, impact_std_y)

p_target_hist   = np.array(p_target_hist).T
v_target_hist   = np.array(v_target_hist).T


# -- save this online optimization to online learning data bank

with open(path+'/Data Online Learning/Run_{}'.format(int(global_iterator[0])), 'wb') as f:
    pickle.dump(target_Init, f)
    pickle.dump(target, f)
    pickle.dump(TargetHitting.theta_history, f)
    pickle.dump(TargetHitting.impact_history, f)
    pickle.dump(TargetHitting.error_history, f)
    pickle.dump(p_target_hist, f)
    pickle.dump(v_target_hist, f)
    pickle.dump(step_lengths, f)
    pickle.dump(mode, f)
    pickle.dump(use_mean_traj, f)
    pickle.dump(missed, f)
    pickle.dump(pred_time, f)

print('final global iterator: ', global_iterator)


# ---- Plots

if len(TargetHitting.theta_history) > 1:

    # -- plot evolution of predictions

    # hit_pos_list = np.array(hit_pos_list)
    # hit_time_list = np.array(hit_time_list)

    # plt.figure()
    # for i in range(len(hit_time_list)):
    #     plt.plot(hit_time_list[i])
    #     plt.title('hit time prediction')

    # for j in range(3):
    #     plt.figure()
    #     for i in range(len(hit_time_list)):
    #         plt.plot(hit_pos_list[i][:, j])
    #         plt.title('hit pos prediction - DoF {}'.format(j))

    # -- plot target position and velocity (cylinder frame)

    # titles = ['theta0', 'r', 'h']

    # fig, axs = plt.subplots(3, 2)
    # for j in range(3):
    #     axs[j, 0].plot(p_target_hist[j, :])
    #     axs[j, 0].set_title('p target - {}'.format(titles[j]))
    #     axs[j, 1].plot(v_target_hist[j, :])
    #     axs[j, 1].set_title('v target - {}'.format(titles[j]+'_dot'))
    # fig.suptitle('Targets for PathPlaning')

    # -- plot TargetY, which can be for hitting position calculation

    TargetY_hist = np.array(TargetY_hist)
    plt.figure()
    plt.plot(TargetY_hist)
    plt.title('targetY')
    plt.ylabel('y [m]')

    print('error history: ', TargetHitting.error_history)

    TargetHitting.PlotThetaHisotry()

    plt.figure()
    plt.plot([i**0.5 for i in TargetHitting.error_history], label='error')
    # plt.plot([i**0.5 for i in TargetHitting.error_history], label='distance')
    plt.xlabel('iteration [-]')
    plt.ylabel('distance error [m]')
    plt.legend()

    plt.figure()
    plt.scatter([i[0] for i in TargetHitting.impact_history], [i[1] for i in TargetHitting.impact_history])
    plt.scatter(TargetHitting.target[0], TargetHitting.target[1], marker='x')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    # -- mean landing point error

    mean_error = [] # epsilon_i
    for k in range(1, len(TargetHitting.impact_history)+1):
        mean_impact_x = np.mean([i[0] for i in TargetHitting.impact_history[:k]])
        mean_impact_y = np.mean([i[1] for i in TargetHitting.impact_history[:k]])
        mean_impact = np.array([mean_impact_x, mean_impact_y])
        mean_error.append(np.linalg.norm(TargetHitting.target[:2] - mean_impact))

    # plt.figure()
    # plt.xlabel('iteration [-]')
    # plt.ylabel('error of mean impact [m]')
    # plt.plot(mean_error)

    plt.show()