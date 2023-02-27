import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

class RealBall:

    def __init__(self, model,
                 center_of_table, x_of_table, y_of_table, 
                 normalizers=None,
                 half_length_of_table=1.37, half_width_of_table=0.7625,
                 height_of_ground=0.0, height_of_table=0.76,
                 mass_of_ball=0.0027, radius_of_ball=0.02, rho_of_ball=1.225,
                 gravity=9.802, prediction_horizon=1.5, frequency=0.01,
                 delta_r=[1e-4, 1e-4, 1e-4], 
                 delta_v=[1e-2, 1e-2, 1e-2], 
                 delta_omega=[1e-3, 1e-3, 1e-3],
                 delta_y=[1e-4, 1e-4, 1e-4],
                 ini_delta_r=[1e-4, 1e-4, 1e-4],
                 ini_delta_v=[1e-2, 1e-2, 1e-2], 
                 ini_delta_omega=[1e-2, 1e-2, 1e-2],
                 kd = 0.1062,
                 km = 0.0057):
        
# %% parameters of neural network for table impact model
        self.table_impact_model = model
        self.table_impact_normalizers = normalizers

# %% parameters for table
        self.center_of_table = center_of_table
        self.x_of_table = x_of_table
        self.y_of_table = y_of_table
        self.half_length_of_table = half_length_of_table
        self.half_width_of_table = half_width_of_table
        # constraint in z-axis
        self.height_of_ground = height_of_ground
        self.height_of_table  = height_of_table
# %% parameters for ping-pong ball    
        self.kd = kd
        self.km = km
        self.mass_of_ball = mass_of_ball  # kg
        self.radius_of_ball = radius_of_ball # m
        self.rho_of_ball = rho_of_ball

# %% parameters for system or environment
        # sampling frequency
        self.frequency = frequency
        self.gravity = gravity # m/s^2
        self.prediction_horizon = prediction_horizon
# %% parameters for Kalman Filter

        self.delta_rx = delta_r[0]
        self.delta_ry = delta_r[1]
        self.delta_rz = delta_r[2]

        self.delta_vx = delta_v[0]
        self.delta_vy = delta_v[1]
        self.delta_vz = delta_v[2]

        self.delta_omegax = delta_omega[0]
        self.delta_omegay = delta_omega[1]
        self.delta_omegaz = delta_omega[2]

        self.delta_yx = delta_y[0]
        self.delta_yy = delta_y[1]
        self.delta_yz = delta_y[2]

        self.ini_delta_rx = ini_delta_r[0]
        self.ini_delta_ry = ini_delta_r[1]
        self.ini_delta_rz = ini_delta_r[2]

        self.ini_delta_vx = ini_delta_v[0]
        self.ini_delta_vy = ini_delta_v[1]
        self.ini_delta_vz = ini_delta_v[2]

        self.ini_delta_omegax = ini_delta_omega[0]
        self.ini_delta_omegay = ini_delta_omega[1]
        self.ini_delta_omegaz = ini_delta_omega[2]

# %% indirect parameters
        self.area_of_ball = math.pi * (self.radius_of_ball ** 2)  # section area m^2

        self.ini_omega = [1e-3, 1e-3, 1e-3]
    
        self.Q = np.diag([  self.delta_rx, self.delta_ry, self.delta_rz, 
                            self.delta_vx, self.delta_vy, self.delta_vz, 
                            self.delta_omegax, self.delta_omegay, self.delta_omegaz])
        self.R = np.diag([self.delta_yx, self.delta_yy, self.delta_yz])
        self.P_m = np.diag([self.ini_delta_rx, self.ini_delta_ry, self.ini_delta_rz, 
                            self.ini_delta_vx, self.ini_delta_vy, self.ini_delta_vz, 
                            self.ini_delta_omegax, self.ini_delta_omegay, self.ini_delta_omegaz,])
        self.H = np.hstack( ( np.identity(3), np.zeros((3, 6)) ))
        self.I = np.identity(9)
# %% position & velocity
        self.position = np.zeros((3, 1000))
        # self.velocity = None
        self.position_kf = np.zeros((3, 1000))
        self.velocity_kf = np.zeros((3, 1000))
        self.omega_kf = np.zeros((3, 1000))
        # self.position_var = None
        # self.velocity_var = None
        self.time_past = None
        self.x_meas = None
        self.r_meas = None
        self.x_pred = None
# %% parameters for impact
        self.t_to_impact = 100  # big enough
        self.impact_flag = False
        self.t_stamp = None
        self.impact_time_list = []
        self.impact_type = []
        self.state_before_impact = []
        self.state_after_impact = []
# %% input the initial states of the ball

    def GetInitial(self, r, v):

        self.position_kf[:, 0] = r
        self.velocity_kf[:, 0] = v

        self.omega_kf[:, 0] = self.ini_omega
        self.time_past = int( 0 )

    def Falling(self, x, T=None):
        if T is None:
            T = self.frequency
        
        v_norm = np.linalg.norm(x[3:6], ord=2)

        x_est = np.hstack((x[0] + x[3] * T, 
                           x[1] + x[4] * T, 
                           x[2] + x[5] * T - 0.5 * self.gravity * T**2, 
                           x[3] + T * (-self.kd * v_norm * x[3] + self.km * (x[7]*x[5] - x[8]*x[4])) , 
                           x[4] + T * (-self.kd * v_norm * x[4] + self.km * (x[8]*x[3] - x[6]*x[5])) , 
                           x[5] + T * (-self.kd * v_norm * x[5] + self.km * (x[6]*x[4] - x[7]*x[3]) - self.gravity),  
                           x[6],
                           x[7],
                           x[8],)).reshape(-1, 1)

        return x_est

    def JacobianMatrix(self, x, T=None):
        if T is None:
            T = self.frequency

        # define following for easier notation

        I = np.identity(3)
        Z = I * 0

        v_norm = np.linalg.norm(x[3:6], ord=2)

        vx = x[3].item()
        vy = x[4].item()
        vz = x[5].item()

        omegax = x[6].item()
        omegay = x[7].item()
        omegaz = x[8].item()

        # Calculate individual 3x3 sub-matrices of Jacobian s.t.
        # Jac = (I,   T*I, 0,  )
        #       (0,   F_1, F_2 )
        #       (0,   0,   I,  )

        F_1_drag = - T * self.kd * np.array([[(v_norm + vx**2 / v_norm), (vy * vx / v_norm), (vz * vx / v_norm)],
                                             [(vy * vx / v_norm), (v_norm + vy**2 / v_norm), (vz * vy / v_norm)],
                                             [(vz * vx / v_norm), (vy * vz / v_norm), (v_norm + vz**2 / v_norm)]])
                        
        F_1_magnus = T * self.km * np.array([[0.0, - omegaz, omegay],
                                             [omegaz, 0.0, - omegax],
                                             [- omegay, omegax, 0.0]])

        F_1 = I + F_1_drag + F_1_magnus

        F_2 = - T * np.array([[v_norm * vx, 0.0, 0.0],
                            [0.0, v_norm * vy, 0.0],
                            [0.0, 0.0, v_norm * vz]])

        Jac = np.block([[I,   T*I, Z, ],
                        [Z,   F_1, F_2],
                        [Z,   Z  , I  ]])
        return Jac
# %% import the impact model
    def ImpactWithTable( self, x ): 

        before_impact   = np.array([x[3].item(), x[4].item(), x[5].item(), x[6].item(), x[7].item(), x[8].item()])

        A               = np.array([[ -0.99591269,  0.07944872, 0.04296286],
                                    [ -0.07685274, -0.99529339, 0.05903159],
                                    [  0.04745064,   0.0554885, 0.99733117]]) # from calibration 2022_09_22

        CAL             = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        inv_A           = np.linalg.inv(A)

        before_impact[0:3] = A@before_impact[0:3]
        before_impact[3:6] = A@before_impact[3:6]

        if self.table_impact_normalizers is not None: # if model expects normalized input, we normalize the inputs
            for i in range(6):
                before_impact[i] = (before_impact[i] - self.table_impact_normalizers[i][0]) / self.table_impact_normalizers[i][1]

        after_impact = self.table_impact_model(torch.Tensor(before_impact.T)).T
        after_impact = after_impact.detach().numpy()
        after_impact = inv_A@(after_impact)

        x_after = np.array([x[0].item(), x[1].item(), x[2].item(), after_impact[0], after_impact[1], after_impact[2], 0.0, 0.0, 0.0])

        # -- alternative linear table impact model with or without spin
        '''
        IM = np.array([[ 0.90089277,  0.07471497, -0.05945311],
        [-0.00405459,  0.56571272, -0.05611265],
        [-0.00939996, -0.03824324, -0.89371729]])

        IM_spin = np.array([[ 0.92218783,  0.04610198, -0.03673059, -3.39081688,  0.25554156, -0.1943666 ],
                        [ 0.04064519,  0.81947474, -0.31047129, -7.00512388, -1.61519793,  5.92262433],
                        [-0.01085564, -0.11349084, -0.82546919, -0.09295188,  0.479439,   -0.29948612],
                        [ 0.,          0.,          0.,          0.,          0.,          0.        ],
                        [ 0.,          0.,          0.,          0.,          0.,          0.        ],
                        [ 0.,          0.,          0.,          0.,          0.,          0.        ]])

        before_impact = np.array([x[3].item(), x[4].item(), x[5].item(), x[6].item(), x[7].item(), x[8].item()])

        # after_impact = IM_spin@before_impact
        after_impact = IM@before_impact[0:3]
        x_after = np.array([x[0].item(), x[1].item(), x[2].item(), after_impact[0], after_impact[1], after_impact[2], 0.0, 0.0, 0.0])
        '''

        return x_after

    def ImpactWithGround(self, x):
        CM = np.array([[0.40, 0.0,  0.0],
                       [ 0.0, 0.4,  0.0],
                       [ 0.0, 0.0, -0.5]])
        x_res = CM@x[3:6].reshape(-1, 1)
        x_est = np.vstack((x[0:2], self.height_of_ground, x_res, x[6:15]))
        return x_est

    def RegionCheck( self, x=None ):
        if x is None:
            x = np.copy( self.x_meas )
        
        flag = True
        # coordinates of the ball in x-y plane
        cord_x = x[0]
        cord_y = x[1]
    
        # distance to x-direction
        dis_x = abs( -self.x_of_table[1] * cord_x + self.x_of_table[0] * cord_y - (-self.x_of_table[1]*self.center_of_table[0]+self.x_of_table[0]*self.center_of_table[1]) ) / (self.x_of_table[0]**2 + self.x_of_table[1]**2 )
        # distance to y-direction
        dis_y = abs( -self.y_of_table[1] * cord_x + self.y_of_table[0] * cord_y - (-self.y_of_table[1]*self.center_of_table[0]+self.y_of_table[0]*self.center_of_table[1]) ) / (self.y_of_table[0]**2 + self.y_of_table[1]**2 )

        if ( dis_x <= self.half_length_of_table ) and ( dis_y <= self.half_width_of_table ):
            flag = True
        else:
            flag = False
        return flag

    def ImpactDetection( self, x=None, T=None ):
        if x is None:
            x = np.copy( self.x_meas )
        if T is None:
            T = self.frequency
        
        if self.RegionCheck( x ):
            height_limit = self.height_of_ground + self.height_of_table
            #height_limit = -0.465
        else:
            height_limit = self.height_of_ground
        # velocity and position of the ball in z-axis
        vz = x[5]
        rz = x[2]

        height_difference = height_limit - rz
        # introduce nonlinear model???

        if vz**2 - 2*self.gravity*height_difference < 0:
            return 0.0

        t = (vz + np.sqrt(vz**2 - 2*self.gravity*height_difference)) / self.gravity

        #print('until impact: t = {}, z = {}'.format(t, rz))

        return t  
# %% predict the rest trajectory
    def Prediction( self, x=None, prediction_horizon=None, time_past=None ):
        '''
        x:  the final state of the ball
        prediction_horizon: how long time to predict, in s
        time_past: how long time has past, in s
        '''
        if x is None:
            x = np.copy( self.x_meas )
        if prediction_horizon is None:
            prediction_horizon = self.prediction_horizon
        if time_past is None:
            time_past = self.time_past

        #time_point = len(self.t_stamp) - 1
        #trajectory = np.copy( self.position_kf[:, 0:time_point+1] )
        trajectory = np.array([[x[0]], [x[1]], [x[2]]])
        t_stamp = np.array([time_past])
        impact_time_list = []
        impact_type = []
        for i in range( len(self.impact_time_list) ):
            if self.impact_time_list[i] < time_past:
                impact_time_list.append( np.float(self.impact_time_list[i]) )
                impact_type.append( self.impact_type[i] )
            else:
                break

        if prediction_horizon < time_past:
            print(prediction_horizon, time_past)
            print('No more prediction!!!')
            return (trajectory, t_stamp, impact_time_list, impact_type)  
        else:
            self.x_kf_before_impact = None
            x_before = np.copy( x )
            step_left = round( (prediction_horizon-time_past) /self.frequency )
            for step in range( step_left ):
                time_past += self.frequency
                x_after = self.Falling(x_before)

                impact_flag = False

                if self.RegionCheck( x_after ):
                    # ball is within the table
                    height_limit = self.height_of_ground + self.height_of_table
                    type = 'Table'
                else:
                    # ball is out of the table
                    height_limit = self.height_of_ground
                    type = 'Ground'

                if x_after[2] < height_limit:
                    impact_flag = True
                    rz = x_before[2]
                    vz = x_before[5]
                    height_difference = height_limit - rz

                    temp = vz**2 - 2*self.gravity*height_difference
                    if temp < 0:
                        t_to_impact = 0
                        print('limit case t to impact = 0')
                    else:
                        t_to_impact = (vz + np.sqrt(vz**2 - 2*self.gravity*height_difference)) / self.gravity

                    x_after = self.Falling( x_before, t_to_impact )
                    self.x_kf_before_impact = x_after.copy()
                    if type == 'Table':
                        x_after = self.ImpactWithTable( x_after )
                    elif type == 'Ground':
                        x_after = self.ImpactWithGround( x_after )
                    
                    x_after = self.Falling(x_after, self.frequency-t_to_impact)

                    impact_time_list.append( np.float(time_past) )

                    impact_type.append( type )

                t_stamp = np.append( t_stamp, time_past )
                trajectory = np.append( trajectory, x_after[0:3].reshape(-1, 1), axis=1 )
                x_before = np.copy( x_after )

                #if type == 'Ground' and impact_flag == True:
                #    break

            return (trajectory, t_stamp, impact_time_list, impact_type)    

# %% Constant and define the matrices
    def StateEstimation(self, r, v, k=None, step_length=None):

        # r for positions from sensor, v for velocities from sensor

        if step_length is None:
            step_length = self.frequency

        if self.time_past is None:
            self.GetInitial(r, v)
            self.x_meas = np.hstack((self.position_kf[:, 0],
                                    self.velocity_kf[:, 0],      
                                    self.omega_kf[:, 0])).reshape(-1, 1)
            self.t_stamp = np.array([self.time_past])
            self.t_to_impact = self.ImpactDetection()  # detect the impact time for the next step
            # self.x_meas_list.append( self.x_meas )
        else:
            
            # in which state?
            if self.t_to_impact > step_length:
                self.impact_flag = False
            else:
                self.impact_flag = True

            # Kalman Filter
            self.r_meas = r.reshape(-1, 1)
            
            if self.impact_flag == False:
                self.x_pred = self.Falling(self.x_meas, step_length)
              
            else: # perform an impact 

                self.impact_time_list.append(k)
                self.x_pred = self.Falling(self.x_meas, self.t_to_impact)

                self.state_before_impact.append(self.x_meas[0:15])

                if self.RegionCheck(x=self.x_pred):
                    self.x_pred = self.ImpactWithTable( self.x_pred ).reshape(-1, 1)
                    #print('Impact of table at k = {}, t = {}'.format(k, self.t_stamp[-1]))
                else:
                    self.x_pred = self.ImpactWithGround( self.x_pred )
                    self.impact_type.append('Ground')

                self.x_pred = self.Falling(self.x_pred, step_length-self.t_to_impact)

            if len(self.impact_time_list) > 0:
                if k - self.impact_time_list[-1] < 2:
                    Q = 50 * self.Q
                else:
                    Q = self.Q
            else:
                Q = 1*self.Q

            A = self.JacobianMatrix(self.x_meas, step_length)

            P_p = A@self.P_m@A.T + Q
            K = P_p@self.H.T @ np.linalg.inv(self.H@P_p@self.H.T + self.R)

            x_before = self.x_meas

            self.x_meas = np.array(self.x_pred + K@(self.r_meas - self.H@self.x_pred)).reshape(-1, 1)
            self.P_m = (self.I - (K@self.H))@P_p

            self.time_past += step_length
            
            if len(self.t_stamp) >= self.position_kf.shape[1]:
                return

            self.position_kf[:, len(self.t_stamp)] = self.x_meas[0:3].flatten()
            self.velocity_kf[:, len(self.t_stamp)] = self.x_meas[3:6].flatten()
            self.omega_kf[:, len(self.t_stamp)]    = self.x_meas[6:9].flatten()

            self.t_stamp = np.append( self.t_stamp, self.time_past )
            # predict the impact time for the next step
            self.t_to_impact = self.ImpactDetection()
            if self.impact_flag == True:
                self.impact_flag = False
                self.state_after_impact.append( self.x_meas[0:9] )
    
    def GetHitPoint(self, traj=None, t=None, hit_point_=None, hit_time_=None, TargetY=None):
        print()
        print('GetHitPoint')
        if traj is None or t is None:
            (traj, t, impact_time, impact_type) = self.Prediction()

        def find_nearest(a, goal):
            idx = (np.abs(a-goal)).argmin()
            return idx

        if TargetY is None:
            TargetY = 0.0

        '''strategy 1'''
        # if len(impact_type) == 1:
        #     if impact_type[0] == 'Table':

        #         point = find_nearest(t, impact_time[0]) 
        #         z_position =  np.max(traj[2, point:])
        #         time_point = np.int(np.where(traj[2, point:]==z_position)[0]) + point
        #         hit_time = t[time_point]
        #         hit_point = traj[:, time_point]

        #         if np.linalg.norm(hit_point, ord=2)>=(0.3768+0.4038):  # l_1 + l_2
        #             print('-------Out--------', hit_point)
        #             return (hit_point_, hit_time_, traj)
        #         else:
        #             print('-------Correct--------')
        #             return (hit_point, hit_time, traj)
        #     else:
        #         print('-------Ground--------')
        #         return (hit_point_, hit_time_, traj)  
        # else:
        #     print('-------Empty--------')
        #     return (hit_point_, hit_time_, traj)  
        
        '''strategy 2'''
        point = find_nearest(traj[1, :], TargetY)
        hit_time = t[point]
        hit_point = traj[:, point]

        if np.linalg.norm(hit_point, ord=2)>=(0.3768 + 0.4038): #(0.373352+0.4100):  # l_1 + l_2
            print('-------Out--------', hit_point)
            return (hit_point_, hit_time_, traj)
        else:
            print('-------Correct--------', hit_point, hit_time)
            return (hit_point, hit_time, traj)

        '''strategy 3'''
        # if len(impact_type) != 0:
        #     if impact_type[0] == 'Table':

        #         point = find_nearest(traj[1, :], 0.00)
        #         hit_time = t[point]
        #         hit_point = traj[:, point]

        #         if np.linalg.norm(hit_point, ord=2)>=(0.3768+0.4038):  # l_1 + l_2
        #             print('-------Out--------', hit_point)
        #             return (hit_point_, hit_time_, traj)
        #         else:
        #             print('-------Correct--------')
        #             return (hit_point, hit_time, traj)
        #     else:
        #         print('-------Ground--------')
        #         return (hit_point_, hit_time_, traj)  
        # else:
        #     print('-------Empty--------')
        #     return (hit_point_, hit_time_, traj)  

    def Plot2DEstimation( self ):
    
        # print(self.time_past)
        # print(self.t_stamp)
        # print(self.position_kf)
        speed_list = [np.linalg.norm(self.velocity_kf[:, i]) for i in range(0, 150)]

        fig = plt.figure( figsize=(16, 8) )
        plt.plot(speed_list)
        plt.show()


        fig = plt.figure( figsize=(16, 8) )
        # plt.plot(self.t_stamp, self.position_kf[0, :len(self.t_stamp)] , linewidth=0.5, label=r'estimated x')
        # plt.plot(self.t_stamp, self.position_kf[:,0] + self.var_position[:, 0] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.position_kf[:,0] - self.var_position[:, 0] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.position[0, :len(self.t_stamp)] , linewidth=1, linestyle='--', label=r'x')

        # plt.plot(self.t_stamp, self.position_kf[1, :len(self.t_stamp)] , linewidth=0.5, label=r'estimated y')
        # plt.plot(self.t_stamp, self.position_kf[:,1] + self.var_position[:, 1] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.position_kf[:,1] - self.var_position[:, 1] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.position[1, :len(self.t_stamp)] , linewidth=1, linestyle='--', label=r'y')

        # plt.plot(self.t_stamp, self.position_kf[2, :len(self.t_stamp)] , linewidth=0.5, label=r'estimated z')
        # plt.plot(self.t_stamp, self.position_kf[:,2] + self.var_position[:, 2] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.position_kf[:,2] - self.var_position[:, 2] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.position[2, :len(self.t_stamp)] , linewidth=1, linestyle='--', label=r'z')
        
        #plt.xlabel('Time in t')
        #plt.ylabel('Position in m')
        #plt.legend(ncol=7, loc='upper center')
        #plt.show()

        # fig = plt.figure( figsize=(16, 8) )
        # plt.plot(self.t_stamp, self.velocity_kf[0, :] , linewidth=0.5, label=r'estimated x velocity')
        # # plt.plot(self.t_stamp, self.velocity_kf[:,0] + self.var_velocity[:, 0] , linewidth=0.5, linestyle='--')
        # # plt.plot(self.t_stamp, self.velocity_kf[:,0] - self.var_velocity[:, 0] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.velocity[0, :] , linewidth=1, linestyle='--', label=r'x velocity')

        # plt.plot(self.t_stamp, self.velocity_kf[1 ,:] , linewidth=0.5, label=r'estimated y velocity')
        # # plt.plot(self.t_stamp, self.velocity_kf[:,1] + self.var_velocity[:, 1] , linewidth=0.5, linestyle='--')
        # # plt.plot(self.t_stamp, self.velocity_kf[:,1] - self.var_velocity[:, 1] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.velocity[1, :] , linewidth=1, linestyle='--', label=r'y velocity')

        # plt.plot(self.t_stamp, self.velocity_kf[2, :] , linewidth=0.5, label=r'estimated z velocity')
        # # plt.plot(self.t_stamp, self.velocity_kf[:,2] + self.var_velocity[:, 2] , linewidth=0.5, linestyle='--')
        # # plt.plot(self.t_stamp, self.velocity_kf[:,2] - self.var_velocity[:, 2] , linewidth=0.5, linestyle='--')
        # plt.plot(self.t_stamp, self.velocity[2, :] , linewidth=1, linestyle='--', label=r'z velocity')
      
        # plt.xlabel('Time in t')
        # plt.ylabel('Velocity in m/s')
        # plt.legend(ncol=7, loc='upper center')
        # plt.show()


    def _set_axes_radius(self, ax, origin, radius):
        x, y, z = origin
        ax.set_xlim3d([x - radius, x + radius])
        ax.set_ylim3d([y - radius, y + radius])
        ax.set_zlim3d([z - radius, z + radius])
        
    def set_axes_equal(self, ax: plt.Axes):
        limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
        origin = np.mean( limits, axis=1 )
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        self._set_axes_radius(ax, origin, radius)

    def Plot3DEstimation( self, y_sensor=None, y_to_compare=None, y_real=None, if_table='yes' ):

        if y_real is None:
            y_real = np.copy(self.position_kf)
        if y_to_compare is None:
            y_to_compare = [np.copy(self.position_kf)]

        fig = plt.figure( figsize=(8, 8) )
        ax = plt.subplot(111, projection='3d')

        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)

        for i in range(len(y_to_compare)):
            if i%10 == 0 or True:
                ax.scatter(y_to_compare[i][0, :], y_to_compare[i][1, :], y_to_compare[i][2, :], c='red', s=0.5)
                ax.plot3D(y_to_compare[i][0, :], y_to_compare[i][1, :], y_to_compare[i][2, :],'grey', linewidth=0.1)
                #ax.scatter(y_to_compare[0, :], y_to_compare[1, :], y_to_compare[2, :], c='red', s=0.5)
                #ax.plot3D(y_to_compare[0, :], y_to_compare[1, :], y_to_compare[2, :],'grey', linewidth=0.1)

        #ax.scatter(y_sensor[0, 0:len(self.t_stamp)], y_sensor[1, 0:len(self.t_stamp)], y_sensor[2, 0:len(self.t_stamp)], c='blue', s=5)
        #ax.plot3D(y_sensor[0, 0:len(self.t_stamp)], y_sensor[1, 0:len(self.t_stamp)], y_sensor[2, 0:len(self.t_stamp)],'grey', linewidth=0.1, label=r'trajectory')
        #ax.scatter(y_sensor[0, 0], y_sensor[1, 0], y_sensor[2, 0], c='g', s=50 )

        ax.scatter(y_real[0, 0:len(self.t_stamp)], y_real[1, 0:len(self.t_stamp)], y_real[2, 0:len(self.t_stamp)], c='black', s=5)
        
        ax.plot3D(y_real[0, 0:len(self.t_stamp)], y_real[1, 0:len(self.t_stamp)], y_real[2, 0:len(self.t_stamp)],'grey', linewidth=0.1, label=r'real')

        plt.legend(ncol=1, loc='upper center', shadow=True, fontsize=14)
        ax.set_xlabel(r'$x$ in m', fontsize=14)
        ax.set_ylabel(r'$y$ in m', fontsize=14)   
        ax.set_zlabel(r'$z$ in m', fontsize=14)
        self.set_axes_equal(ax)   

        # draw the table
        if if_table == 'yes':
            v1 = self.center_of_table + self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
            v1 = np.append(v1, self.height_of_ground + self.height_of_table)

            v2 = self.center_of_table + self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
            v2 = np.append(v2, self.height_of_ground + self.height_of_table)

            v3 = self.center_of_table - self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
            v3 = np.append(v3, self.height_of_ground + self.height_of_table)

            v4 = self.center_of_table - self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
            v4 = np.append(v4, self.height_of_ground + self.height_of_table)

            vtx = list([tuple(v1), tuple(v3), tuple(v4) , tuple(v2)])
            table_area = Poly3DCollection([vtx], facecolors='lightsteelblue', linewidths=1, alpha=0.5) 
            # table_area.set_color('linen')
            table_area.set_edgecolor('k')
            ax.add_collection3d( table_area )

        #plt.show()

    def PlotHitPoint(self, y=None, hit_position=None, if_table='yes', if_traj='yes'):
        if y is None:
            y = np.copy(self.position_kf)
        
        fig = plt.figure( figsize=(8, 8) )
        ax = plt.subplot(111, projection='3d')

        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)

        ax.scatter(hit_position[0, :], hit_position[1, :], hit_position[2, :],c='r', s=20)
        
        if if_traj=='yes':
            ax.scatter(y[0, 0:len(self.t_stamp)], y[1, 0:len(self.t_stamp)], y[2, 0:len(self.t_stamp)], c='blue', s=5)
            ax.plot3D(y[0, 0:len(self.t_stamp)], y[1, 0:len(self.t_stamp)], y[2, 0:len(self.t_stamp)],'grey', linewidth=0.1, label=r'trajectory')
            ax.scatter(y[0, 0], y[1, 0], y[2, 0], c='g', s=50 )

        plt.legend(ncol=1, loc='upper center', fontsize=14)
        ax.set_xlabel(r'$x$ in m', fontsize=14)
        ax.set_ylabel(r'$y$ in m', fontsize=14)   
        ax.set_zlabel(r'$z$ in m', fontsize=14)
        self.set_axes_equal(ax)   

        # draw the table
        if if_table == 'yes':
            v1 = self.center_of_table + self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
            v1 = np.append(v1, self.height_of_ground + self.height_of_table)

            v2 = self.center_of_table + self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
            v2 = np.append(v2, self.height_of_ground + self.height_of_table)

            v3 = self.center_of_table - self.half_width_of_table * self.x_of_table + self.half_length_of_table * self.y_of_table
            v3 = np.append(v3, self.height_of_ground + self.height_of_table)

            v4 = self.center_of_table - self.half_width_of_table * self.x_of_table - self.half_length_of_table * self.y_of_table
            v4 = np.append(v4, self.height_of_ground + self.height_of_table)

            vtx = list([tuple(v1), tuple(v3), tuple(v4) , tuple(v2)])
            table_area = Poly3DCollection([vtx], facecolors='lightsteelblue', linewidths=1, alpha=0.5) 
            # table_area.set_color('linen')
            table_area.set_edgecolor('k')
            ax.add_collection3d( table_area )

        plt.show()
        
