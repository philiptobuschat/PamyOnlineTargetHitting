import math
import numpy as np
import o80
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tensorflow as tf
from LimitCheck import LimitCheck

class Filter:

    def __init__(self, dof, y_des, dim, pressure_min, pressure_max, 
                 num, den, order_num, order_den, ndelay, angle_initial, total_iteration ):
        '''
        y_des:         relative desired trajectory
        angle_initial: initial angle
        
        y_des + angle_initial = absolute trajectory
        '''

        self.dof = dof
        y_des = np.array(y_des).reshape(-1, 1)
        self.y_des = np.copy(y_des)
        self.pressure_min = pressure_min
        self.pressure_max = pressure_max
        self.order_num = order_num
        self.order_den = order_den
        self.num = num[0:order_num+1]
        self.den = den[0:order_den+1]
        self.delay = ndelay
        self.dim = dim
        self.bounds = []
        for i in range(self.dim):
            self.bounds.append( (self.pressure_min, self.pressure_max) )
        self.angle_initial = angle_initial
        self.total_iteration = total_iteration
        self.pid_for_tracking = np.array([[-13000, 0, -300],
                                          [80000/10, 0, 300],
                                          [-5000/5, 0, -100],
                                          [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])

    def GenerateCovMatrix(self, variance_d, variance_y, variance_w, variance_ini):
        Q = np.eye(self.dim) * variance_d
        R = np.eye(self.dim) * variance_y + self.Bd@self.Bd.T * variance_w
        P = np.eye(self.dim) * variance_ini
        self.Q = Q
        self.R = R
        return(P)

    def GenerateLocalMatrix(self, mode_name='none', fs=100):
        '''
        For now this function only works for some special cases.
        '''
        if mode_name == 'none':
            if self.order_den == 2:
                A = [[          0,            1,           0,           0],
                    [-self.den[2], -self.den[1], self.num[2], self.num[1]],
                    [           0,            0,           0,           1],
                    [           0,            0,           0,           0]]
                b = [[          0], 
                     [self.num[0]], 
                     [          0], 
                     [          1]]
                bd = [[0], 
                      [1], 
                      [0], 
                      [0]]
                c = [0, 1, 0, 0]
                bdes = [0, 0, 0, 0]
            elif self.order_den == 3:
                A = [[           0,            1,            0,           0,           0,           0],
                     [           0,            0,            1,           0,           0,           0],
                     [-self.den[3], -self.den[2], -self.den[1], self.num[3], self.num[2], self.num[1]],
                     [           0,            0,            0,           0,           1,           0],
                     [           0,            0,            0,           0,           0,           1],
                     [           0,            0,            0,           0,           0,           0]]
                b = [[          0], 
                     [          0], 
                     [self.num[0]], 
                     [          0], 
                     [          0], 
                     [           1]]
                bd = [[0], 
                      [0], 
                      [1], 
                      [0], 
                      [0], 
                      [0]]
                c = [0, 0, 1, 0, 0, 0]
                bdes = [0, 0, 0, 0, 0]
            elif self.order_den == 1:
                A = [[-self.den[1], self.num[1]],
                     [           0,           0]]
                b = [[self.num[0]], 
                     [          1]]
                bd = [[1], 
                      [0]]
                c = [1, 0]
                bdes = [0, 0, 0]


        elif mode_name == 'pd':

            kp = self.pid_for_tracking[self.dof][0]
            ki = self.pid_for_tracking[self.dof][1]
            kd = self.pid_for_tracking[self.dof][2]
            
            '''
            den[2] = beta2
            den[1] = beta1
            num[2] = alpha2
            num[1] = alpha1
            num[0] = alpha0
            '''

            if self.order_den == 2:
                '''
                z[k] = [q[k-6], q[k-5], q[k-4], q[k-3], q[k-2], q[k-1], q[k], u[k-d-1], u[k-d]]
                '''
                A = [[0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [self.num[2]*kd*fs, -self.num[2]*(kp+kd*fs)+self.num[1]*kd*fs, 
                      -self.num[1]*(kp+kd*fs)+self.num[0]*kd*fs, -self.num[0]*(kp+kd*fs), 
                      0, -self.den[2], -self.den[1], self.num[2], self.num[1]],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0]]
                b = [[          0],
                     [          0],
                     [          0],
                     [          0],
                     [          0],
                     [          0], 
                     [self.num[0]], 
                     [          0], 
                     [          1]]
                bd = [[0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [0], 
                      [1], 
                      [0], 
                      [0]]
                c = [0, 0, 0, 0, 0, 0, 1, 0, 0]
                bdes = [-self.num[2]*kd*fs, self.num[2]*(kp+kd*fs)-self.num[1]*kd*fs, 
                        self.num[1]*(kp+kd*fs)-self.num[0]*kd*fs, self.num[0]*(kp+kd*fs)]
            elif self.order_den == 3:
                '''
                z[k] = [q[k-6], q[k-5], q[k-4], q[k-3], q[k-2], q[k-1], q[k], u[k-d-2], u[k-d-1], u[k-d]]
                '''

                A = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [self.num[3]*kd*fs, -self.num[3]*(kp+kd*fs)+self.num[2]*kd*fs,
                      -self.num[2]*(kp+kd*fs)+self.num[1]*kd*fs, -self.num[1]*(kp+kd*fs)+self.num[0]*kd*fs,
                      -self.num[0]*(kp+kd*fs)-self.den[3], -self.den[2], -self.den[1], self.num[3], self.num[2], self.num[1]],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
                b = [[          0], 
                     [          0],
                     [          0],
                     [          0],
                     [          0],
                     [          0], 
                     [self.num[0]], 
                     [          0], 
                     [          0], 
                     [          1]]
                bd = [[0],
                      [0],
                      [0],
                      [0],
                      [0],
                      [0], 
                      [1], 
                      [0], 
                      [0], 
                      [0]]
                c = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                bdes = [-self.num[3]*kd*fs, self.num[3]*(kp+kd*fs)-self.num[2]*kd*fs, 
                        self.num[2]*(kp+kd*fs)-self.num[1]*kd*fs, self.num[1]*(kp+kd*fs)-self.num[0]*kd*fs, 
                        self.num[0]*(kp+kd*fs)]
            elif self.order_den == 1:
                A = [[-self.den[1], self.num[1]],
                     [           0,           0]]
                b = [[self.num[0]], 
                     [          1]]
                bd = [[1], 
                      [0]]
                c = [1, 0]
                bdes = [0, 0, 0]


        self.y_des_bar = np.array([])
        for i in range(-self.delay-self.order_den-1, self.dim-self.delay):
            if i<0:
                self.y_des_bar = np.append(self.y_des_bar, self.y_des[0])
            else:
                self.y_des_bar = np.append(self.y_des_bar, self.y_des[i])
        self.y_des_bar = self.y_des_bar.reshape(-1, 1)

        A = np.array(A)
        b = np.array(b)
        bd = np.array(bd)
        c = np.array(c)
        bdes = np.array(bdes)

        return (A, b, bd, bdes, c)

    def GenerateGlobalMatrix(self, mode_name='none'):
        '''
        mode_name: how to generate the local matrix
        '''
        (A, b, bd, bdes, c) = self.GenerateLocalMatrix(mode_name=mode_name)

        Ao = c@A
        Bu = np.zeros((1, self.dim))
        Bu[0, 0] = c@b
        Bd = np.zeros((1, self.dim))
        Bd[0, 0] = c@bd
        Bdes = np.zeros((1, self.dim+len(bdes)-1))
        Bdes[0, 0:len(bdes)] = bdes

        self.A_power_list = [np.eye(A.shape[0])]
        I_A = A@np.eye(A.shape[0])
        
        for row in range(1, self.dim):
            self.A_power_list.append(I_A)
            
            M_temp = Bu[row-1]
            Md_temp = Bd[row-1]
            Mdes_temp = Bdes[row-1]

            M_temp = np.insert(M_temp, 0, c@I_A@b)
            Md_temp = np.insert(Md_temp, 0, c@I_A@bd)
            Mdes_temp = np.insert(Mdes_temp, 0, 0)

            M_temp = np.delete(M_temp, -1)
            Md_temp = np.delete(Md_temp, -1)
            Mdes_temp = np.delete(Mdes_temp, -1)

            I_A = A@I_A

            Ao = np.vstack((Ao, c@I_A@A))
            Bu = np.vstack((Bu, M_temp))
            Bd = np.vstack((Bd, Md_temp))
            Bdes = np.vstack((Bdes, Mdes_temp))

        self.Ao = Ao
        self.Bu = Bu
        self.Bd = Bd
        self.Bdes = Bdes

        self.Bu_inv = np.linalg.inv( self.Bu.T@self.Bu )

        self.part_1 = self.Bu_inv@self.Bu.T@self.y_des
        self.part_2 = self.Bu_inv@self.Bu.T@self.Ao
        self.part_3 = self.Bu_inv@self.Bu.T@self.Bd
        self.part_4 = self.Bu_inv@self.Bu.T@self.Bdes@self.y_des_bar

    def GenerateVector(self, ff, y, mode_name='none' ):
        '''
        ff: only the feedforward
        y:  the relative real trajectory
        '''
        # y_diff = y[0, 0] - self.y_des[0, 0]
        # ff_diff = ff[0, 0]
        y_diff = 0
        ff_diff = 0
        # print('{}. initial error is: {:.2f} degree'.format(self.dof, y_diff*180/math.pi) )
        if mode_name == 'none':
            if self.order_den == 2:
                z0 = [y_diff, y_diff, ff_diff, ff_diff]
            elif self.order_den == 3:
                z0 = [y_diff, y_diff, y_diff, ff_diff, ff_diff, ff_diff]
            elif self.order_den == 1:
                z0 = [y_diff, ff_diff]
        elif mode_name == 'pd':
            if self.order_den == 2:
                z0 = [y_diff, y_diff, y_diff, y_diff, y_diff, y_diff, y_diff, ff_diff, ff_diff]
            elif self.order_den == 3:
                z0 = [y_diff, y_diff, y_diff, y_diff, y_diff, y_diff, y_diff, ff_diff, ff_diff, ff_diff]
            elif self.order_den == 1:
                z0 = [y_diff, ff_diff]            
            
        # shift d time points to the right
        u = []
        for i in range(self.dim):
            idx = i - self.delay
            if idx < 0:
                u.append( 0 )
            else:
                u.append(ff[idx])

        self.u_bar = np.copy( np.array( u ).reshape(-1, 1) )
        self.z0 = np.array( z0 ).reshape(-1, 1)

    def StepOptimization(self, y, y_out, d):
        y = y.reshape(-1, 1)
        y_out = y_out.reshape(-1, 1)
        d = d.reshape(-1, 1)
        y_err = y - y_out
        n = d.shape[0]  # == self.Bd.shape[1]

        # u[0] = 0    
        u = np.zeros(1)

        sc = np.dot( self.C, self.B )
        for k in range(1, n):
            
            Sum = 0

            for idx in range(0, k):
                Matrix_temp = np.dot(self.C, self.A_power_list[k-idx])
                Matrix_temp = np.dot(Matrix_temp, self.B)

                Sum += np.dot(Matrix_temp, u[idx])

                Matrix_temp = np.dot(self.C, self.A_power_list[k-idx])
                Matrix_temp = np.dot(Matrix_temp, self.bd)

                Sum += np.dot(Matrix_temp, d[idx])
            
            Matrix_temp = np.dot(self.C, self.A_power_list[0])
            Matrix_temp = np.dot(Matrix_temp, self.bd)

            Sum += np.dot(Matrix_temp, d[k])
            
            T = y[k+1] - Sum
            u = np.append( u, 1/sc * T)
        return u

    def Optimization(self, y, ff, d_m, P_m, number_iteration, 
                     weight=(1, 1), mode_name='none'):
        '''
        u_ini: only the feedforward input
        y:     absolute real trajectory
        y - angle_initial = relative real trajectory
        '''
        ff = LimitCheck(ff, self.dof)

        y = np.array( y ).reshape(-1, 1)
        y = y - self.angle_initial[self.dof]
        ff = np.array( ff ).reshape(-1, 1)

        '''
        update the initial state z0, 
        and the shifted input u_bar
        '''
        self.GenerateVector(ff, y, mode_name)

        matrix_Q = np.copy( self.Q )
        d_m = np.array( d_m ).reshape(-1, 1)
        I = np.eye(self.dim)
        # z = y - self.Ao * self.z0 - self.Bu * self.u_bar
        z = y -  self.Ao@self.z0- self.Bu@self.u_bar -self.Bdes@self.y_des_bar
        # print("...step 1: prediction update")
        # prediction update
        d_p = I@d_m
        P_p = P_m + matrix_Q
        # print("...step 2: measurement update")
        # measurement update
        K = P_p@self.Bd.T@( np.linalg.inv( self.Bd@P_p@self.Bd.T + self.R ) )
        # K = P_p * self.Bd.T * np.linalg.inv(self.Bd * P_p * self.Bd.T + self.R)
        d = d_p + K@( z - self.Bd@d_p )
        # d = d_p + K * (z - self.Bd * d_p)
        '''
        P = ( I - K@self.Bd )@P_p
        '''

        a = 1 - weight[0]
        b = 1 - weight[1]
        power_ratio_initial = 1 - a/(self.total_iteration-1) * number_iteration
        power_ratio_final = 1 - b/(self.total_iteration-1) * number_iteration
        initial_value = 10**power_ratio_initial
        final_value = 10**power_ratio_final
        F = np.diag(np.log10(np.linspace(initial_value, final_value, num=self.dim, endpoint=True)))
     
        P = (I - K@self.Bd)@P_p@((I-K@self.Bd).T) + K@(F@self.R@F)@(K.T)
        # P = (I - K * self.Bd) * P_p 

        # calculate the variables in lifted space
        d_lifted = self.Bd@d
        P_lifted = self.Bd@P@self.Bd.T

        # print("...step 3: solve the optimization equation")
        # with bounds
        def fun(x):
            x = x.reshape(-1, 1)
            a = abs( self.y_abs - np.dot(self.Bu, x) - np.dot(self.Bd, d) )
            f = 0.5 * np.asscalar( np.dot( a.T, a ) )
            return f

        # u_ini = np.array(u_ini).reshape(-1, 1)
        # result = minimize(fun, u_ini, method="SLSQP", bounds=self.bounds)

        # u = result.x
        # u = np.array(u).reshape(-1, 1)

        # without bounds, calculate the explicit expression

        u = self.part_1 - self.part_2@self.z0 - self.part_3@d - self.part_4

        u = LimitCheck(u, self.dof)

        # u = self.StepOptimization( self.y_abs, y, d )
        # u = np.array( u )

        # shift d time points to the left
        for _ in range(self.delay):
            u = np.insert(u, -1, u[-1])
            u = np.delete(u, 0)
        
        return(u, d, P, d_lifted, P_lifted)



        





        

