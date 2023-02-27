'''
This script is the configuration of Pamy
'''
import numpy as np
from RealRobot import Robot
from RealRobotGeometry import RobotGeometry
import pickle5 as pickle
# %%
'''
The initial posture of pamy, all the value is absolute value
in angular space in rad.
'''
GLOBAL_INITIAL = np.array([0.000000,  0.514884,  0.513349, 0.0])
# %%
def FulfillZeros( a ):
    b = np.zeros(  [len(a), len(max(a, key = lambda x: len(x)))]  )
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b
# %%
'''
All the necessary parameters that used to built the model and
inverse model of Pamy.
'''
dof_list = [0, 1, 2, 3]

delta_d_list   = np.array([1e-9, 1e-11, 1e-8, 1e-9]) 
delta_y_list   = np.array([1e-3, 1e-3, 1e-3, 1e-3])
delta_w_list   = np.array([1e-9, 1e-11, 1e-8, 1e-9])
delta_ini_list = np.array([1e-7, 1e-7, 1e-7, 1e-7])

inverse_model_num = [[ -1.603053083740808,  4.247255623123213,  -3.894718257376192, 1.248397916673204],
                     [-39.4562899165285,    107.6338556336952,  -101.4865793303678, -33.2833280994107],
                     [-0.5340642923467604, 0.9405453623185243,  -0.4103846610378554                   ],
                     [  1.139061120808877, -1.134002042583525                                        ]]
inverse_model_num = FulfillZeros( inverse_model_num ) * 1e5

inverse_model_den = [[1.000000000000000, -2.397342550890251, 2.136262918863130, -0.683005338583667],
                     [1.000000000000000, -1.972088694237271, 1.604586816873790, -0.428252150060600],
                     [1.000000000000000, -1.702657081211864, 0.823186864989291                    ],
                     [1.000000000000000, -0.825587854345462                                       ]]
inverse_model_den = FulfillZeros( inverse_model_den )

ndelay_list = np.array([2, 2, 3, 1])

order_num_list = np.array([3, 3, 2, 1])
order_den_list = np.array([3, 3, 2, 1])

model_num = [[ -0.62380966054252,    1.49548544287500,  -1.33262144624559,    0.42606532841061],
             [-0.0253445015260062,  0.0499816049205160, -0.0406674530288672, 0.0108538372707263],
             [ -1.87243373940214,    3.18811256549306,  -1.54136285983862                     ],
             [ 0.877916014980719,  -0.724796799103451                                         ]]
model_num = FulfillZeros( model_num ) * 1e-5

model_den = [[1.000000000000000, -2.649479088497819, 2.429562874042614, -0.778762680621906],
             [1.000000000000000, -2.727926418358119, 2.572126764707656, -0.843549359806079],
             [1.000000000000000, -1.761108869843411, 0.768418085460389                    ],
             [1.000000000000000, -0.995558554204924                                       ]]
model_den = FulfillZeros( model_den )   

model_num_order   = [3, 3, 2, 1]
model_den_order   = [3, 3, 2, 1]
model_ndelay_list = [2, 2, 3, 1]

anchor_ago_list = np.array([17500, 18500, 16000, 15000])
anchor_ant_list = np.array([17500, 18500, 16000, 15000])

strategy_list = np.array([2, 2, 2, 2])

pid_list = [[-3.505924158687806e+04, -3.484022215671791e+05/5, -5.665386729745434e+02],
            [-8.228984656729296e+04, -1.304087541343074e+04/2, -4.841489121599795e+02],
            [    -36752.24956301624,    -246064.5612272051/10,     -531.2866756516057],
            [ 3.422187330173758e+04, 1.673663594798479e+05/10,     73.238165769446297]]
pid_list = FulfillZeros( pid_list )

pressure_min = [-4500 - 1000, -6500 - 1000, -6000 - 1000, -7000 - 1000]
pressure_max = [4500 + 1000, 6500 + 1000, 6000 + 1000, 7000 + 1000]

weight_list = [(0.1, 1.0),
               (1.0, 1.0),
               (0.1, 1.0),
               (1.0, 1.0)]
# %%
'''
Build the robot model
'''
def build_pamy(frontend=None):
    Pamy = Robot(frontend, dof_list, model_num, model_den,
                model_num_order, model_den_order, model_ndelay_list,
                inverse_model_num, inverse_model_den, order_num_list, order_den_list,
                ndelay_list, anchor_ago_list, anchor_ant_list, strategy_list, pid_list,
                delta_d_list, delta_y_list, delta_w_list, delta_ini_list,
                pressure_min, pressure_max, weight_list)
    return Pamy
# %%
def build_geometry():
    Pamy_geometry = RobotGeometry(initial_posture=GLOBAL_INITIAL)
    return Pamy_geometry