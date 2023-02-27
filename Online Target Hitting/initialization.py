#!/usr/bin/env python3
'''
This script is one part of interception task.
It will complete the following tasks:
    - turn state_initialization to 0/inactive
    - return Pamy to the initial posture
    - initialize the the hit_point and hit_time randomly
    - generate the reference trajectory in augular space,
      and the corresponding feedforward controller based
      on the hit_point and hit_time
    - generate the feedforward input
    - turn state_launcher to 1/active, begin to shoot a ball
'''
import numpy as np
import SharedArray as sa
import PAMY_CONFIG
import math
import numpy as np
import o80
import o80_pam
import signal_handler
from cnn import cnn_model
import time
import matplotlib.pyplot as plt
import pickle5 as pickle
import os
# %%

frontend             = o80_pam.FrontEnd("real_robot")
Geometry             = PAMY_CONFIG.build_geometry()
Pamy                 = PAMY_CONFIG.build_pamy(frontend=frontend)
cnn                  = cnn_model()

state_initialization = sa.attach("shm://state_initialization") 
state_launcher       = sa.attach("shm://state_launcher")
hit_position         = sa.attach("shm://hit_position")         
hit_time             = sa.attach("shm://hit_time")         
reference_polar      = sa.attach("shm://reference_polar")
velocity_polar       = sa.attach("shm://velocity_polar")
acceleration_polar   = sa.attach("shm://acceleration_polar")
reference_angle      = sa.attach("shm://reference_angle")
reference_length     = sa.attach("shm://reference_length")
feedforward          = sa.attach("shm://feedforward")
pointer              = sa.attach("shm://pointer")
ball_index           = sa.attach("shm://ball_index")
theta                = sa.attach("shm://theta")


tol_factor = 5

print('initialize pamy')


Pamy.AngleInitialization(Geometry.initial_posture, tolerance_list=[tol_factor*0.1, tol_factor*0.3, tol_factor*0.3, tol_factor*0.1])
Pamy.PressureInitialization()


print('initialize pamy done')

while not signal_handler.has_received_sigint():

  print('initializer waiting ', state_initialization[0])
  time.sleep(0.5)

  if int(state_initialization[0]) == 1:

    print('secondary initilization pamy')
    Pamy.AngleInitialization(Geometry.initial_posture, tolerance_list=[tol_factor*0.1, tol_factor*0.3, tol_factor*0.3, tol_factor*1.0])
    Pamy.PressureInitialization()
    print('secondary initilization pamy done')
    angle_initial = np.array(frontend.latest().get_positions())

    hit_position[0:3] = np.array([-0.55, 0.0,  0.05])  # in Cartesian space
    # hit_position[0:3] = np.array([-0.62025401, -0.000, -0.10456108])
    hit_angle = Geometry.EndToAngle(hit_position, frame='Cartesian')
    hit_time[0] = 1.0
    pointer[0] = int(0) 

    time_begin = time.time()
    (p_polar, v_polar, a_polar, p_angle, t_stamp, _, _) = Geometry.PathPlanning(time_point=0, 
                                                                          theta=theta,
                                                                          T_go=hit_time[0], 
                                                                          T_back=1.5, T_steady=0.2, 
                                                                          angle=angle_initial, 
                                                                          target=hit_angle)

    p_last_dof = np.zeros(len(t_stamp)).reshape(1, -1)
    p_angle    = np.vstack((p_angle, p_last_dof))


    reference_length[0]                   = len(t_stamp)
    reference_polar[:, 0:len(t_stamp)]    = p_polar
    velocity_polar[:, 0:len(t_stamp)]     = v_polar
    acceleration_polar[:, 0:len(t_stamp)] = a_polar
    reference_angle[:, 0:len(t_stamp)]    = p_angle
    feedforward[:, 0:len(t_stamp)]        = cnn.get_feedforward(p_angle[0:3, :] - p_angle[0:3, 0].reshape(-1, 1))
    
    # fig = plt.figure(figsize=(16, 16))
    # for i in range(3):
    #     plt.plot(t_stamp, feedforward[i, 0:len(t_stamp)], label=r'dof {}'.format(i))
    # plt.legend(ncol=3)
    # plt.show()

    state_initialization[0] = int(0)  # turn state_initialization to inactive

    print('-------------------------------------------------------------')  
    
    state_launcher[0] = int(1)   # activate the launcher
    print(time.time() - time_begin)
  