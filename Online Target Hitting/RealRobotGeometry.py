import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import MinJerk_penalty as MJ_penalty
import numba as nb
import pickle5 as pickle

class RobotGeometry:

    def __init__(self, initial_posture=np.array([0, -30, -30, 0])/180*math.pi, 
                 l_1=0.3768, l_2=0.4038, step_size=0.01):
        # initial posture is given in joint space
        self.initial_posture = initial_posture
        self.l_1 = l_1
        self.l_2 = l_2
        self.step_size = step_size
        (_, self.position_final) = self.AngleToEnd(self.initial_posture[0:3],  frame='Cylinder')
        self.acceleration_final = [0, 0, 0]

    # from angle space to end effector space in Cartesian Coordinate or Polar Coordinate
    def AngleToEnd(self, angle, frame='Cartesian', l_1=None, l_2=None):
        # from joint space to end effector space
        # position_A is the end effector of the first robot arm
        # position_B is the end effector of the second robot arm
        # can be transformed into Cartesian space, Polar space or Cylinder space
        if l_1 is None:
            l_1 = self.l_1
        if l_2 is None:
            l_2 = self.l_2
    
        theta0 = angle[0]
        theta1 = angle[1]
        theta2 = angle[2]

        if frame == 'Cartesian':
            x = math.cos(theta0) * math.sin(theta1) * l_1
            y = math.sin(theta0) * math.sin(theta1) * l_1
            z = math.cos(theta1) * l_1
            position_A = np.array([x, y, z])
            x = math.cos(theta0) * math.sin(theta1) * l_1 + math.cos(theta0) * math.sin(theta1 + theta2) * l_2
            y = math.sin(theta0) * math.sin(theta1) * l_1 + math.sin(theta0) * math.sin(theta1 + theta2) * l_2
            z = math.cos(theta1) * l_1 + math.cos(theta1 + theta2) * l_2
            position_B = np.array([x, y, z])

        elif frame == 'Cylinder':
            # theta0, r, z
            position_A = np.array([theta0, abs(l_1*math.sin(theta1)), (l_1*math.cos(theta1))])
            r = np.sqrt(l_1**2 + l_2**2 - 2*l_1*l_2*math.cos(math.pi-abs(theta2)))
            gama = math.acos( (l_1**2 + r**2 - l_2**2) / (2*l_1*r) )
            if theta2 > 0:
                alpha = theta1 + gama
            else:
                alpha = theta1 - gama
            position_B = np.array([theta0, abs(r*math.sin(alpha)), (r*math.cos(alpha))])

        return (position_A, position_B)

    def AngleToEnd_Velocities(self, theta, frame='Cylinder', l_1=None, l_2=None):
        '''
        From joint positions and their derivatives to derivatives of end effector (of second arm)
        return (theta0_dot, rho_dot, h_dot)
        '''
        if l_1 is None:
            l_1 = self.l_1
        if l_2 is None:
            l_2 = self.l_2

        theta0 = theta[0]
        theta1 = theta[1]
        theta2 = theta[2]

        theta0_dot = theta[4]
        theta1_dot = theta[5]
        theta2_dot = theta[6]

        if frame == 'Cylinder':

            r_dot = l_1 * math.cos(theta1) * theta1_dot + l_2 * math.cos(theta1 + theta2) * (theta1_dot + theta2_dot)
            h_dot = - l_1 * math.sin(theta1) * theta1_dot - l_2 * math.sin(theta1 + theta2) * (theta1_dot + theta2_dot)

            cyl_vel = np.array([theta0_dot, r_dot, h_dot])

            return cyl_vel

    def EndToAngle( self, position, frame='Cartesian'):
        l_1 = self.l_1
        l_2 = self.l_2
        # transform coordinates in Cartesian space into joint space
        if frame == 'Cartesian':
            x = position[0]
            y = position[1]
            z = position[2]

            try:

                # l = np.linalg.norm(position, ord=2)
                # theta0 = - math.atan( y / np.sqrt( x**2 + y**2 ) )
                # gama = math.acos( (l_1**2 + l**2 - l_2**2) / (2*l_1*l) )
                # theta1 = -(math.pi / 2 - (gama + math.asin(y / l)) )
                # gama = math.acos( (l_1**2 + l_2**2 - l**2) / (2*l_1*l_2) )
                # theta2 = -( math.pi - gama )
                # angle = np.array( [theta0, theta1, theta2] )
                l      = np.linalg.norm(position, ord=2)
                theta0 = math.atan(y/x)

                gama   = math.acos((l_1**2+l**2-l_2**2)/(2*l_1*l)) if l<=(l_1+l_2) else 0
                alpha  = math.asin(z/l)
                theta1 = math.pi/2 - gama - alpha

                beta   = math.acos((l_1**2 + l_2**2 - l**2)/(2*l_1*l_2) ) if l<=(l_1+l_2) else math.pi
                theta2 = math.pi - beta

                angle  = np.array([theta0, theta1, theta2])
            except:
                print('l1 = {}, l2 = {}, l = {}'.format(l_1, l_2, l))
                print('position = {}'.format(position))
                print('cos(gamma1) = ', (l_1**2 + l**2 - l_2**2) / (2*l_1*l))
                print('cos(gamma2) = ', (l_1**2 + l_2**2 - l**2) / (2*l_1*l_2))

        return angle

    def CalAngularTrajectory(self, trajectory, angle, frame='Cartesian'):
        # transform trajectory in end effector space into joint space
        l_1 = self.l_1
        l_2 = self.l_2

        eps = 1e-10
        angle_trajectory = np.array([])

        if frame == 'Cartesian':
            for i in range(trajectory.shape[1]):
                x = trajectory[0, i]
                y = trajectory[1, i]
                z = trajectory[2, i]

                angle = self.EndToAngle(trajectory[:, i], frame=frame)
                angle_trajectory = np.append(angle_trajectory, angle)
                
        elif frame == 'Cylinder':
            # dimension of trajectory: 3 * length
            angle_trajectory = np.zeros(trajectory.shape)
            for i in range(trajectory.shape[1]):

                theta0 = trajectory[0, i]
                l      = trajectory[1, i]
                z      = trajectory[2, i]

                r = np.sqrt( l**2 + z**2 ) if np.sqrt( l**2 + z**2 ) <= l_1+l_2 else l_1+l_2
                if abs(z/r) <= 1:
                    z_over_r = z/r
                elif z/r < 0:
                    z_over_r = -1
                elif z/r > 0:
                    z_over_r = 1
                alpha = abs(math.pi/2 - math.asin( z_over_r))
                 
                theta2 = angle[2]
                theta0_ = theta0

                if abs((l_1**2 + l_2**2 - r**2) / (2*l_1*l_2)) <= 1:
                    beta = (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2)
                elif (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) > 0:
                    beta = 1
                elif (l_1**2 + l_2**2 - r**2) / (2*l_1*l_2) < 0:
                    beta = -1

                theta2_ = math.pi - math.acos(beta)
                if abs(-theta2_ - theta2) < abs(theta2_ - theta2):
                    theta2_ = -theta2_

                gamma = math.acos((l_1**2 + r**2 - l_2**2) / (2*l_1*r))
                if theta2_ > 0:
                    theta1_ = alpha - gamma
                else:
                    theta1_ = alpha + gamma

                angle = np.array([theta0_, theta1_, theta2_])
                angle_trajectory[:, i] = angle     

        return angle_trajectory

    def PathPlanning( self, time_point, theta, T_go=1.0, T_back=1.0, T_steady=0.1,
                      angle=None, velocity_initial=np.array([0, 0, 0]), 
                      acceleration_initial=np.array([0, 0, 0]),
                      target=None, frequency=100, plan_weight=(6, 10), 
                      HitVel_factor=None):
        '''
        angle is absolute variable in angular space
        target is absolute variable in angular space
        velocity_initial and acceleration_initial should be in the Cylinder
        coordinate system
        '''
        (_, p_initial) = self.AngleToEnd(angle[0:3], frame='Cylinder')  # theta1, r, h
        (_, p_target)  = self.AngleToEnd(target[0:3], frame='Cylinder')  # theta1, r, h


        (_, p_target_theta)  = self.AngleToEnd(theta, frame='Cylinder')  # theta0, r, h

        v_target_theta = self.AngleToEnd_Velocities(theta) # theta0_dot, r_dot, h_dot
        v_target = np.array([4.0/p_target[1], 0, 0])

        print('v target old = {}, v target theta = {}'.format(v_target, v_target_theta))
        print('p target old = {}, p target theta = {}'.format(p_target, p_target_theta))
        

        # p_target = p_target_theta
        v_target = np.array([v_target_theta[0], 0.0, 0.0])
        # v_target[0] = 4

        v_final  = np.array([0, 0, 0])
        a_target = np.array([0, 0, 0])
        a_final  = np.array([0, 0, 0])
        
        if (time_point >= int(T_go * frequency)) and (time_point<int((T_go+T_back)*frequency)):
            m_list = [[1.0, 1.0],
                      [1.0, 1.0],
                      [1.0, 1.0]]
            m_list = np.array(m_list)
            n_list = [[plan_weight[1], 0.1],
                      [           1.0, 0.1],
                      [           1.0, 0.1]]
            n_list = np.array(n_list)
            # target is given in joint space
            t = np.array([time_point/frequency, T_go+T_back, T_go+T_back+T_steady])
            # corresponding positions
            p = np.array([p_initial, self.position_final, self.position_final]).T
            p = p - self.position_final.reshape(-1, 1)
            # corresponding velocities
            v = np.array([velocity_initial, v_final, v_final]).T
            a = np.array([acceleration_initial, a_final, a_final]).T

        elif time_point < int(T_go * frequency):
            m_list = [[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]]
            m_list = np.array(m_list)
            n_list = [[plan_weight[0], plan_weight[1], 0.1],
                      [           1.0,            1.0, 0.1],
                      [           1.0,            1.0, 0.1]]
            n_list = np.array(n_list)
            t = np.array([time_point/frequency, T_go, T_go+T_back, T_go+T_back+T_steady])
            p = np.array([p_initial, p_target, self.position_final, self.position_final]).T
            p = p - self.position_final.reshape(-1, 1)
            v = np.array([velocity_initial, v_target, v_final, v_final]).T
            a = np.array([acceleration_initial, a_target, a_final, a_final]).T
        
        [p_mjp, v_mjp, a_mjp, t_stamp] = MJ_penalty.PathPlanning(p, v, a, t, 1/frequency, m_list, n_list)    
        p_mjp = p_mjp + self.position_final.reshape(-1, 1)
        p_angular_mjp = self.CalAngularTrajectory(p_mjp, angle[0:3] + self.initial_posture[0:3], frame='Cylinder')

        return (p_mjp, v_mjp, a_mjp, p_angular_mjp, t_stamp, p_target, v_target)

if __name__ == '__main__':
    Robot = RobotGeometry()
    
    # %% read data of balls
    path_of_file = "/home/hao/Desktop/Hao/" + 'BallsData' + '.txt'
    file = open(path_of_file, 'rb')
    time_list = pickle.load(file)
    position_list = pickle.load(file)
    velocity_list = pickle.load(file)
    file.close()

    position_mean = np.mean( position_list, axis=0 )
    # offset angles of the upright posture
    offset = np.array( [2.94397627, -0.078539855235, -0.06333859293225] )
    # angles of initial posture
    angle_initial_ref = np.array( [2.94397627, -0.605516948, -0.5890489142699] )
    # anchor angles to hit the ball
    angle_anchor_ref = np.array( [ 2.94397627, -1.452987321865, -0.87660612618] )
    # after calibration
    angle_initial = angle_initial_ref - offset
    angle_anchor = angle_anchor_ref - offset

    (_, position_anchor) = Robot.AngleToEnd(angle_anchor, frame='Cartesian')
    position_error = position_anchor - position_mean
    position_list = position_list + position_error
    
    l_1 = 0.4
    l_2 = 0.38
    index_list = range(len(time_list))

    for index in index_list:
        position = position_list[index, :] # x, y, z
        angle = np.array([0, 0, 0]) / 180 * math.pi
        if np.linalg.norm(position, ord=2) <= l_1+l_2:
            T = time_list[index]
            V = velocity_list[index]
            target = Robot.EndToAngle( position, frame='Cartesian') # theta1, theta2, theta3
            t_list = np.array([0, T, T+1.0, T+1.1])
            (p_mja, p_mjp, p_angular_mja, p_angular_mjp, p_angular_mjl, t_stamp) = Robot.PathPlanning( time_point=0, T_go=T, T_back=1.0, T_steady=0.2,
                                                            angle=angle, velocity_initial=np.array([0, 0, 0]), 
                                                            acceleration_initial=np.array([0, 0, 0]),
                                                            target=target, frequency=100 )
            GetPlot(p_mja, p_mjp, p_angular_mja, p_angular_mjp, p_angular_mjl, t_list, 0.01, index)