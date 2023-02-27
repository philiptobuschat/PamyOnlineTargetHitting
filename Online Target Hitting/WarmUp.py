'''
This script is used to warm the robot arm up.
'''
import PAMY_CONFIG
import math
import os
import numpy as np
import o80
import o80_pam
import matplotlib.pyplot as plt
# %%
frontend                   = o80_pam.FrontEnd("real_robot")
Geometry                   = PAMY_CONFIG.build_geometry()
Pamy                       = PAMY_CONFIG.build_pamy(frontend=frontend)
(time_list, position_list) = PAMY_CONFIG.get_recorded_ball(Geometry=Geometry)
# %%
number_iteration = 3
# %%
Pamy.AngleInitialization(Geometry.initial_posture)
Pamy.PressureInitialization()
angle_initial_read = np.array(frontend.latest().get_positions())