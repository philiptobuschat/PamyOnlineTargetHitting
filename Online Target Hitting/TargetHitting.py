'''
Define the features necessary for the online optimization to hit targets with the returned balls
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle5 as pickle
import math

from RealBall import RealBall

# %% neural network for the landing points prediction

class NN_mini(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = torch.nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 4)
        self.fc4 = torch.nn.Linear(4, 4)
        self.fc5 = torch.nn.Linear(4, 2)
    def forward(self, x):
        x = x.float()
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        x = self.fc5(x)
        return x.double()


# %% online optimization

class TargetHitting():
    '''
    Define the functions necessary for the online optimization to hit targets with the returned balls
    '''
    def __init__(self, path, target=None, position=None, t_stamp=None, theta0=None):

        self.A          = np.array([[ -0.99591269,  0.07944872,  0.04296286],
                                    [ -0.07685274, -0.99529339,  0.05903159],
                                    [  0.04745064,  0.0554885,   0.99733117]])  # from calibration 2022_09_22

        self.CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134])    # from calibration 2022_09_22

        self.l_1 = 0.3768
        self.l_2 = 0.4038

        # initial guess for theta is either specified or an average of collected data
        if theta0 is None:
            self.theta0 = np.array([0.1, 0.709, 1.56, 0.0675, 6.16, 0.0, 0.0])
            self.theta  = self.theta0.copy()
        else:
            self.theta0 = theta0
            self.theta  = self.theta0.copy()

        self.target     = target

        self.impact_history = []
        self.error_history = []
        self.theta_history = []

        if position is None or t_stamp is None:
            # use example trajectory to find robot settings, from racket impact measurement series 1, index 104
            self.t_stamp = np.array(   [0.75536333, 0.76106726, 0.76674227, 0.77228314, 0.77757082, 0.78386458,
                                        0.78883354, 0.79495475, 0.79987123, 0.80551501, 0.81189658, 0.81671424,
                                        0.82206566, 0.82769587, 0.83332147, 0.83894042, 0.84426675, 0.8503849,
                                        0.85548954, 0.86114202, 0.86657562, 0.87254118, 0.87770778, 0.88338202,
                                        0.88869888])
            self.position = np.array( [[-4.64228562e-01, -4.63892477e-01, -4.63317373e-01, -4.63482700e-01,
                                        -4.61889724e-01, -4.60053167e-01, -4.60327459e-01, -4.60038917e-01,
                                        -4.59262423e-01, -4.57925801e-01, -4.58073031e-01, -4.56196245e-01,
                                        -4.56542104e-01, -4.55657140e-01, -4.55580093e-01, -4.53596929e-01,
                                        -4.51700845e-01, -4.52529120e-01, -4.52941435e-01, -4.53395034e-01,
                                        -4.52685483e-01, -4.52236034e-01, -4.48614602e-01, -4.47895461e-01,
                                        -4.47124761e-01],
                                        [-1.52514901e-01, -1.45504456e-01, -1.37246818e-01, -1.29030675e-01,
                                        -1.21777432e-01, -1.17202663e-01, -1.10349829e-01, -1.02119464e-01,
                                        -9.52319511e-02, -8.98948087e-02, -7.94653997e-02, -7.45633420e-02,
                                        -6.51631917e-02, -5.86255179e-02, -5.00433402e-02, -4.54482827e-02,
                                        -3.89466933e-02, -2.95196778e-02, -2.13798153e-02, -1.47147929e-02,
                                        -6.25025464e-03,  1.60790921e-05,  7.01414105e-03,  1.24816523e-02,
                                        1.94680717e-02],
                                        [ 1.99521408e-02,  2.40803529e-02,  2.83636784e-02,  3.18066041e-02,
                                        3.55822725e-02,  3.72245040e-02,  4.12732928e-02,  4.48137238e-02,
                                        4.57386664e-02,  4.74306041e-02,  5.07611131e-02,  5.10075417e-02,
                                        5.24365761e-02,  5.13420680e-02,  5.29356378e-02,  5.17601315e-02,
                                        5.17069842e-02,  5.27352738e-02,  5.27106755e-02,  5.08770922e-02,
                                        5.05300277e-02,  4.91787676e-02,  4.83927817e-02,  4.60573004e-02,
                                        4.35138555e-02]] )
        else:
            self.t_stamp = t_stamp
            self.position = position

        path_model = path+'/Racket Impact Model/RacketImpact_8_mini_TanH_Norm_epoch_1000'
        self.model = NN_mini()
        self.model.load_state_dict(torch.load(path_model))
        self.model.eval()

        with open(path+'/Racket Impact Model/RacketImpact_8_normalizers', 'rb') as f:
            self.normalizers = pickle.load(f)

        self.std_diag = np.diag([i[1] for i in self.normalizers])
        
    def InitialSettings(self, mode='NN', step_length=1e-3, tol=1e-4, k_max=None, penalize_target_velocity=False):
        '''
        find initial settings to play on target via offline gradient descent, i.e. solve model(trajectory, theta) = target for theta
        '''
        theta_curr  = self.theta0.copy()
        theta_red   = np.array([theta_curr[0].item(), theta_curr[3].item(), theta_curr[4].item()]) # reduced theta with only three parameters
        diff        = tol + 1
        errors      = []
        k           = 0
        impact_hist = []
        theta_hist  = []

        vel_bef, pos, k_hit = self.ModelInput_Physical(theta_curr)

        print()
        print('start initial guess calculation... ')

        while diff > tol:

            if k_max is not None:
                if k > k_max:
                    break
            k += 1
            if k%250 == 0:
                print(' ------- iteration {} -------'.format(k))

            if mode == 'NN':
                model_input = self.ModelInput_NN(theta_red)
                J = torch.autograd.functional.jacobian(self.model, model_input).detach().numpy().T
                J = np.linalg.inv(self.std_diag)@J # scale jacobian back based on std of normalization
                Jac = J.copy()
                M = self.model(model_input).detach().numpy()

            elif mode == 'Physical':
                theta_inp = np.array([theta_red[0], theta_curr[1], theta_curr[2], theta_red[1], theta_red[2], theta_curr[5], theta_curr[6]])
                J = self.JacobianPhysical(theta_inp, vel_bef, pos, z_table = -0.43 + -0.083375)
                Jac = J[:2, :].T
                Jac = np.vstack((Jac[0, :], Jac[3, :], Jac[4, :]))
                M = self.GetHitPoint(theta_inp, vel_bef, pos)

                if k == 1 :
                    print()
                    print(' ---- k = {} ---- '.format(k))
                    print('theta into physical jacobian: ', theta_inp)
                    print('predicted impact: ', M[:2])
                    print('Jac: ', Jac)

            if mode in ['NN', 'Physical']:
                impact_hist.append(M)
                err = 1/2 * np.linalg.norm(M[:2] - self.target[:2])**2
                step_matrix = step_length * np.diag([1, 1, 1])
                theta_red = theta_red - step_matrix @ (Jac@(M[:2] - self.target[:2]))

                if theta_red[0] < -0.2:
                    theta_red[0] = -0.2
                if theta_red[0] > 0.5:
                    theta_red[0] = 0.5

                if theta_red[1] < -0.3:
                    theta_red[1] = -0.3
                if theta_red[1] > 0.3:
                    theta_red[1] = 0.3

                theta_hist.append(theta_red)

                diff = np.linalg.norm(Jac@(M[:2] - self.target[:2]))

        print('initial guess calculation finied after {} iteration'.format(k))
        print()
                
        self.theta[0] = theta_red[0]
        self.theta[3] = theta_red[1]
        self.theta[4] = theta_red[2]

        # print('initial theta: ', self.theta)
        # print('predicted initial Hit point: ', M)
        # print('target: ', self.target)

        if self.theta[0] > math.pi/2:
            print('theta 0 on boundary')
            self.theta[0] = math.pi/2

        if self.theta[0] < -0.1:
            print('theta 0 on boundary')
            self.theta[0] = -0.1

        # -- plot results
        # fig, axs = plt.subplots(3)
        # axs[0].plot([i[0] for i in theta_hist])
        # axs[0].set_title('theta0')
        # axs[1].plot([i[1] for i in theta_hist])
        # axs[1].set_title('theta3')
        # axs[2].plot([i[2] for i in theta_hist])
        # axs[2].set_title('theta0_dot')

        # plt.figure()
        # plt.plot([i[0] for i in impact_hist], [i[1] for i in impact_hist], marker='x')
        # plt.show()

        return theta_curr, errors, M, impact_hist, theta_hist

    def ImpactFromTraj_uncalibrated(self, meas_pos, meas_t_stamp, z_table = -0.44, T_step_falling=0.005):
        
        '''
        Find where a ball hit the table if measurements available or 
        simulate where ball must have crossed the table height if measurements are not available.
        Do this in uncalibrated system where the table is properly even
        '''

        # find visible impacts
        table_impacts, racket_impact = self.impact_detection_uncalibrated(meas_pos, z_table = z_table)

        if racket_impact == -1:
            print('no racket impact found')
            return
        
        table_impact = -1
        for i in table_impacts: # use first impact on table after racket impact for the following
            if i > racket_impact:
                table_impact = i
                break

        Ball = RealBall(model = None,
                        center_of_table=np.array([0.80, 1.71, -0.435]),
                        x_of_table=np.array([-1, 0]),
                        y_of_table=np.array([0, -1]),
                        height_of_ground= -0.435 -0.76,
                        height_of_table=0.0)

        ball_position = meas_pos[:, racket_impact]
        ball_velocity = (meas_pos[:, racket_impact+3] - meas_pos[:, racket_impact]) / (meas_t_stamp[racket_impact+3] - meas_t_stamp[racket_impact])
        Ball.StateEstimation(ball_position, ball_velocity)

        v_meas = []
        v_filt = []

        # estimate until when we can use data. If there are abrupt changes in the trajectory we don't use the data after that
        k_end_cut_off = len(meas_t_stamp) - 3
        errors_cut_off_x = []
        errors_cut_off_z = []
        for k_end in range(racket_impact + 5, len(meas_t_stamp)):
            test_px = np.poly1d( np.polyfit(meas_t_stamp[racket_impact:k_end], meas_pos[0, racket_impact:k_end], 2) )
            test_pz = np.poly1d( np.polyfit(meas_t_stamp[racket_impact:k_end], meas_pos[2, racket_impact:k_end], 2) )
            err_cut_off_x = 0
            err_cut_off_z = 0
            for j in range(racket_impact, k_end):
                err_cut_off_x += (test_px(meas_t_stamp[j]) - meas_pos[0, j])**2
                err_cut_off_z += (test_pz(meas_t_stamp[j]) - meas_pos[2, j])**2
            errors_cut_off_x.append(err_cut_off_x)
            errors_cut_off_z.append(err_cut_off_z)
            err_tot= err_cut_off_x + err_cut_off_z
            if err_tot>5e-3:
                k_end_cut_off = k_end - 5
                break

        pos_filt = []

        # filter trajectory until we have no more useful data or reach found impact
        step = 1
        while Ball.x_meas[2] > z_table and step < len(meas_t_stamp)-racket_impact: # Kalman Filter data until table height reached, no more data, or detected table impact(possibly slightly above set table height)
            pos_filt.append(Ball.x_meas.copy().flatten())
            ball_meas = meas_pos[:, racket_impact+step]
            t_step = meas_t_stamp[racket_impact+step] - meas_t_stamp[racket_impact+step-1]
            Ball.StateEstimation(ball_meas, np.zeros(3), k=step+racket_impact ,step_length=t_step)

            v_filt.append(Ball.x_meas.copy().flatten()[3:6])
            v_meas.append( (meas_pos[:, racket_impact+step] - meas_pos[:, racket_impact+step-1]) / t_step)

            if step+racket_impact == table_impact or step+racket_impact == k_end_cut_off:
                break

            step += 1

        if step+racket_impact == table_impact or Ball.x_meas[2] <= z_table: # either impact on table or ball crossed table height limit
            return np.array([Ball.x_meas[0], Ball.x_meas[1], Ball.x_meas[3], Ball.x_meas[4], Ball.x_meas[5]]).flatten()

        x_pred = Ball.x_meas

        pos_pred = []

        # predict trajectory until we cross table height
        while x_pred[2] > z_table: # apply Falling model until Ball is at height of table
            pos_pred.append(x_pred.copy().flatten())
            x_pred = Ball.Falling(x_pred, T=T_step_falling)

        plots = False
        if plots: # plot state filtering and prediction for finding the impact 
            pos_pred.append(Ball.x_meas.copy().flatten())

            fig = plt.figure( figsize=(8, 8) )
            ax = plt.subplot(111, projection='3d')

            ax.spines['bottom'].set_linewidth(1.5)
            ax.spines['top'].set_linewidth(1.5)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['right'].set_linewidth(1.5)

            pos_filt = np.array(pos_filt)
            pos_pred = np.array(pos_pred)

            ax.scatter(pos_filt[:, 0], pos_filt[:, 1], pos_filt[:, 2], c='black', s=5)
            ax.plot3D(pos_filt[:, 0], pos_filt[:, 1], pos_filt[:, 2],'grey', linewidth=0.1, label=r'real')
            ax.scatter(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], c='red', s=5)
            ax.plot3D(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],'red', linewidth=0.1, label=r'real')

            v_filt = np.array(v_filt).T
            v_meas = np.array(v_meas).T

            fig, axs = plt.subplots(3)
            for i in range(3):
                axs[i].plot(v_filt[i, :], label='EKF')
                axs[i].plot(v_meas[i, :], label='Meas')

            plt.show()

        return np.array([x_pred[0], x_pred[1], x_pred[3], x_pred[4], x_pred[5]]).flatten()

    def StepUpdate(self, meas_pos, meas_t_stamp, mode='NN', step_length=1e-3, penalize_target_velocity=False, update_traj=True):
        '''
        perform a step update of online optimization
        Modes NN (neural network) and Physical (first-principle) model are tested and work similarly well.
        Modes Regression (simple data-driven) and simple physical (first-principle with stronger assumptions) are not tested.
        '''

        self.theta_history.append(self.theta.copy()) #save last theta settings

        # -- get racket impact and extract landing point
        _, racket_impact = self.impact_detection(meas_pos) # find racket impact in calibrated system, where conditions can be formulated intuitively in coordinates around the robot
        if racket_impact == -1:
            print('no racket impact found in step update')
            return

        pos_uncal = []
        for k in range(len(meas_t_stamp.flatten())):
            pos_uncal.append( np.linalg.inv(self.A)@meas_pos[:, k] - self.CALIBRATION )
        pos_uncal = np.array(pos_uncal).T

        impact = self.ImpactFromTraj_uncalibrated(pos_uncal, meas_t_stamp) # find (measured or simulated) impact on table, in uncalibrated system where table in properly even

        print('impact at (UNCALIBRATED): ', impact[0:3])
        impact[0:2] = (self.A@(np.array([impact[0], impact[1], -0.44]) + self. CALIBRATION))[0:2]
        impact[2:5] = self.A@(impact[2:5])

        if update_traj:
            self.position = meas_pos
            self.t_stamp = meas_t_stamp

        theta_curr  = self.theta.copy()
        theta_red   = np.array([theta_curr[0].item(), theta_curr[3].item(), theta_curr[4].item()])

        # -- perform gradient descent step on error function 1/2||impact - predicted impact(theta) ||**2

        err = np.linalg.norm(impact[:2] - self.target[:2])**2

        if mode == 'NN':
            model_input = self.ModelInput_NN(theta_red)
            J = torch.autograd.functional.jacobian(self.model, model_input).detach().numpy().T
            J = np.linalg.inv(self.std_diag)@J # scale jacobian back based on std of normalization

            M = self.model(model_input).detach().numpy()

        elif mode == 'Physical':
            vel_bef, pos, k_hit = self.ModelInput_Physical(self.theta)
            J = self.JacobianPhysical(self.theta, vel_bef, pos, z_table = -0.43 + -0.083375)
            J = J[:2, :].T
            J = np.vstack((J[0, :], J[3, :], J[4, :]))

        if mode in ['NN', 'Physical']:
            Jac = J[:, :2]
            step_matrix = step_length * np.diag([1, 0.6, 0])
            theta_red = theta_red - step_matrix @ (Jac@(impact[:2] - self.target[:2]))
            self.theta[0] = theta_red[0]
            self.theta[3] = theta_red[1]
            self.theta[4] = theta_red[2]

        elif mode in ['Regression', 'simple Physical']:
            
            alpha_im = math.atan((impact[0] + 0.6)/(-impact[1]))
            alpha_ta = math.atan((self.target[0] + 0.6)/(-self.target[1]))
            delta_alpha = alpha_im - alpha_ta
            l_im = np.sqrt((impact[0] + 0.6)**2 + impact[1]**2)
            l_ta = np.sqrt((self.target[0] + 0.6)**2 + self.target[1]**2)
            delta_l = l_im - l_ta

            # print('alpha impact = {}, alpha target = {}'.format(alpha_im, alpha_ta))
            # print('l impact = {}, l target = {}'.format(l_im, l_ta))
            # print('delta alpha = {}, delta l = {}'.format(delta_alpha, delta_l))

        if mode == 'Regression':
            J = np.array([[0.91, 0.0], [0.0, 1.85]]) # values based on linear regression
            # J = [dalpha_dtheta0 [rad/rad],    0                   ]
            #     [0,                           dl_dtehta3 [m/rad]  ]  
            delta = np.array([delta_alpha, delta_l])
            update = - step_length * J@delta
            self.theta[0] = self.theta[0] + update[0]
            self.theta[3] = self.theta[3] + update[1]

        elif mode == 'simple Physical':
            v = 8.5 # [m], estimate of ball speed after racket impact
            g = 9.81 # [m/s^2], gravity
            delta_z = 0.44 # height difference between table and pamy
            racket_stickiness = 1 # [-], 1 if outgoing angle is racket angle (sticky), 2 if incoming angle = outgoimg angle (elastic)

            sqrt = np.sqrt(math.sin(self.theta[3])**2 / g**2 + 2*delta_z/(v**2*g))

            dl_dtheta3 = v**2*(math.cos(racket_stickiness*self.theta[3]*2)/g + math.sin(self.theta[3])*math.cos(self.theta[3])**2 /(g**2*sqrt) - math.sin(self.theta[3])*sqrt )
            dalpha_d_theta0 = racket_stickiness
            J = np.diag((dalpha_d_theta0, dl_dtheta3))

            delta = np.array([delta_alpha, delta_l])
            update = - step_length * J@delta
            self.theta[0] = self.theta[0] + update[0]
            self.theta[3] = self.theta[3] + update[1]
        
        if self.theta[0] > np.pi/2:
            print('theta 0 on boundary')
            self.theta[0] = np.pi/2

        if self.theta[0] < -np.pi/2:
            print('theta 0 on boundary')
            self.theta[0] = -np.pi/2

        if self.theta[3] > np.pi/4:
            print('theta 3 on boundary')
            self.theta[3] = np.pi/4

        if self.theta[3] < -np.pi/4:
            print('theta 3 on boundary')
            self.theta[3] = -np.pi/4
        
        self.impact_history.append(impact)
        self.error_history.append(err)

    def JacobianPhysical(self, theta, vel_bef, pos, z_table = -0.43):
        '''
        get gradient (Jacobian) for the physical model.
        '''

        def Jac_Falling(x, T):

            kd = 0.1062
            g = 9.802
            I = np.identity(3)
            Z = I * 0

            v_norm = np.linalg.norm(x[3:6], ord=2)
            vx = x[3].item(); vy = x[4].item(); vz = x[5].item()
            px = x[0].item(); py = x[1].item(); pz = x[2].item()

            F_1 = - T * kd * np.array([[(v_norm + vx**2 / v_norm), (vy * vx / v_norm), (vz * vx / v_norm)],
                                    [(vy * vx / v_norm), (v_norm + vy**2 / v_norm), (vz * vy / v_norm)],
                                    [(vz * vx / v_norm), (vy * vz / v_norm), (v_norm + vz**2 / v_norm)]])

            F_1 = F_1.reshape(3, 3)

            Jac = np.block([[I,   T*I  ],
                            [Z,   I+F_1]])

            if T != 0.005:
                T = vz/g + np.sqrt(vz**2/g**2 + 2*(pz - z_table)/g)
                sqr = np.sqrt(vz**2/g**2 + 2*(x[2] - z_table)/g)
                dT_dz = 1/g * 1/sqr
                dT_dvz = 1/g * (1 + vz/g * 1/sqr)
                Jac_add = np.array([
                    [0, 0, dT_dz*vx,                    0, 0, dT_dvz*vx],
                    [0, 0, dT_dz*vy,                    0, 0, dT_dvz*vy],
                    [0, 0, dT_dz*vz,                    0, 0, T+dT_dvz*vz],
                    [0, 0, -kd*dT_dz*v_norm*vx,         0, 0, -kd*dT_dvz*v_norm*vx],
                    [0, 0, -kd*dT_dz*v_norm*vy,         0, 0, -kd*dT_dvz*v_norm*vy],
                    [0, 0, -dT_dz*(kd*v_norm*vz + g),   0, 0, -kd*dT_dvz*(v_norm*vz+g)],
                ])
                Jac = Jac + Jac_add

            return Jac

        def Falling(x, T):
            gravity = 9.802
            kd = 0.1062
            v_norm = np.linalg.norm(x[3:6])

            x_pred = np.array([ x[0] + T * x[3], 
                                x[1] + T * x[4], 
                                x[2] + T * x[5] - 0.5 * T**2 * gravity, 
                                x[3] - T * kd * v_norm * x[3], 
                                x[4] - T * kd * v_norm * x[4], 
                                x[5] - T * kd * v_norm * x[5] - T * gravity])
            return x_pred.flatten()

        # d() referes to derivative w.r.t. robot settings theta unless other specified

        # -- derivative of falling model / ball dynamics

        M_impact = np.array([[0.75, 0.0, 0.0], # simple impact matrix
                            [0.0, -0.75, 0.0],
                            [0.0, 0.0, 0.75]])

        R               = self.Rot_theta(theta)
        R_inv           = np.linalg.inv(R)
        v_R             = self.RacketVelocityCartesian(theta)
        vel_aft         = R_inv@M_impact@R@(vel_bef - v_R) + v_R

        x_curr          = np.array([pos[0], pos[1], pos[2], vel_aft[0], vel_aft[1], vel_aft[2]])
        x0              = x_curr.copy()
        dx_hit_dx_aft   = np.identity(6)

        while x_curr[2] > z_table:

            t_to_impact = (x_curr[5] + np.sqrt(x_curr[5]**2 + 2*9.802*(x_curr[2] - z_table))) / 9.802
            if t_to_impact == 0:
                break

            if t_to_impact > 0.005:
                T = 0.005
                dx_prehit_dx_aft = dx_hit_dx_aft.copy()
            else:
                T = t_to_impact
                Jac_last_ana = Jac_Falling(x_curr, T)

            #dx_hit_dx_aft = dx_hit_dx_aft@Jac_Falling(x_curr, T)
            dx_hit_dx_aft = Jac_Falling(x_curr, T)@dx_hit_dx_aft
            
            x_curr = Falling(x_curr, T)

            # -- test analytic jacobian vs numerical finite different jacobian
            if t_to_impact <= 0.005 and False:

                Jac_ana = Jac_Falling(x_curr, T).round(decimals=4)
                print()
                print()
                print('last Jac Falling ana: ')
                print(Jac_ana)

                x_running = x_curr.copy()
                t_to_impact = (x_running[5] + np.sqrt(x_running[5]**2 + 2*9.802*(x_running[2] - z_table))) / 9.802
                x_running = Falling(x_running, t_to_impact)

                delta = 1e-7
                Jac_last_num = np.zeros((6, 6))
                for col in range(6):
                    x_running_alt = x_curr.copy()
                    x_running_alt[col] += delta
                    t_to_impact = (x_running_alt[5] + np.sqrt(x_running_alt[5]**2 + 2*9.802*(x_running_alt[2] - z_table))) / 9.802
                    x_running_alt = Falling(x_running_alt, t_to_impact)
                    Jac_last_num[:, col] = (x_running_alt - x_running) / delta
                Jac_last_num = Jac_last_num.round(decimals=4)
                print('last Jac Falling num: ', Jac_last_num)
                print('diff: ', Jac_ana - Jac_last_num)
                print('diff norm: ', np.linalg.norm(Jac_ana - Jac_last_num))

        theta1      = theta[0]
        theta2      = theta[1]
        theta3      = theta[2]
        theta4      = theta[3]
        theta1_dot  = theta[4]
        theta2_dot  = theta[5]
        theta3_dot  = theta[6]

        # -- derivatives of rotation matrix

        def R_y(alpha):
            return np.array([[math.cos(alpha),  0, math.sin(alpha)],
                            [0,                1, 0              ],
                            [-math.sin(alpha), 0, math.cos(alpha)]])

        def dR_y(alpha):
            return np.array([[-math.sin(alpha), 0, math.cos(alpha)],
                            [0,                0, 0              ],
                            [-math.cos(alpha), 0, -math.sin(alpha)]])

        def R_z(alpha):
            return np.array([[math.cos(alpha),  -math.sin(alpha), 0],
                            [math.sin(alpha),  math.cos(alpha),  0],
                            [0,                0,                1]])

        def dR_z(alpha):
            return np.array([[-math.sin(alpha), -math.cos(alpha), 0],
                            [math.cos(alpha),  -math.sin(alpha), 0],
                            [0,                0,                0]])

        dR_dtheta1      = R_z(-theta4) @ R_y(theta3) @ R_y(theta2) @ dR_z(-theta1) * (-1) # -1 is inner derivative
        dR_dtheta2      = R_z(-theta4) @ R_y(theta3) @ dR_y(theta2) @ R_z(-theta1)
        dR_dtheta3      = R_z(-theta4) @ dR_y(theta3) @ R_y(theta2) @ R_z(-theta1)
        dR_dtheta4      = dR_z(-theta4) @ R_y(theta3) @ R_y(theta2) @ R_z(-theta1) * (-1) # -1 is inner derivative
        dR_dtheta1_dot  = np.zeros((3, 3))
        dR_dtheta2_dot  = np.zeros((3, 3))
        dR_dtheta3_dot  = np.zeros((3, 3))

        dR_dtheta_list = [dR_dtheta1, dR_dtheta2, dR_dtheta3, dR_dtheta4, dR_dtheta1_dot, dR_dtheta2_dot, dR_dtheta3_dot]
        
        def cos(a): return math.cos(a)
        def sin(a): return math.sin(a)

        dv_R_dtheta1 = np.array([
            self.l_1*(-sin(theta1)*cos(theta2)*theta2_dot - cos(theta1)*sin(theta2)*theta1_dot) + self.l_2*(-sin(theta1)*cos(theta2+theta3)*(theta2_dot+theta3_dot) - cos(theta1)*sin(theta2+theta3)*theta1_dot),
            self.l_1*(cos(theta1)*cos(theta2)*theta2_dot - sin(theta1)*sin(theta2)*theta1_dot) + self.l_2*( cos(theta1)*cos(theta2+theta3)*(theta2_dot+theta3_dot) - sin(theta1)*sin(theta2+theta3)*theta1_dot),
            0,
        ])
        dv_R_dtheta2 = np.array([
            self.l_1*(-cos(theta1)*sin(theta2)*theta2_dot - sin(theta1)*cos(theta2)*theta1_dot) + self.l_2*(-cos(theta1)*sin(theta2+theta3)*(theta2_dot+theta3_dot) - sin(theta1)*cos(theta2+theta3)*theta1_dot),
            self.l_1*(-sin(theta1)*sin(theta2)*theta2_dot + cos(theta1)*cos(theta2)*theta1_dot) + self.l_2*(-sin(theta1)*sin(theta2+theta3)*(theta2_dot+theta3_dot) + cos(theta1)*cos(theta2+theta3)*theta1_dot),
            self.l_1*(-cos(theta2)*theta2_dot) - self.l_2*(cos(theta2+theta3)*(theta2_dot+theta3_dot))
        ])
        dv_R_dtheta3 = np.array([
            self.l_2*(-cos(theta1)*sin(theta2+theta3)*(theta2_dot+theta3_dot) - sin(theta1)*cos(theta2+theta3)*theta1_dot),
            self.l_2*(cos(theta1)*cos(theta2+theta3)*theta1_dot - sin(theta1)*sin(theta2+theta3)*(theta2_dot+theta3_dot)),
            self.l_2*(-cos(theta2+theta3)*(theta2_dot+theta3_dot))
        ])
        dv_R_dtheta4 = np.zeros(3)
        dv_R_dtheta1_dot = np.array([
            self.l_1*(-sin(theta1)*sin(theta2)) + self.l_2*(-sin(theta1)*sin(theta2+theta3)),
            self.l_1*(cos(theta1)*sin(theta2)) + self.l_2*(cos(theta1)*sin(theta2+theta3)),
            0
        ])
        dv_R_dtheta2_dot = np.array([
            self.l_1*(cos(theta1)*cos(theta2)) + self.l_2*(cos(theta1)*cos(theta2+theta3)),
            self.l_1*(sin(theta1)*cos(theta2)) + self.l_2*(sin(theta1)*cos(theta2+theta3)),
            self.l_1*(-sin(theta2)) + self.l_2*(-sin(theta2+theta3))
        ])
        dv_R_dtheta3_dot = np.array([
            self.l_2*(cos(theta1)*cos(theta2+theta3)),
            self.l_2*(sin(theta1)*cos(theta2+theta3)),
            self.l_2*(-sin(theta2+theta3))
        ])

        dv_R_dtheta_list = [dv_R_dtheta1, dv_R_dtheta2, dv_R_dtheta3, dv_R_dtheta4, dv_R_dtheta1_dot, dv_R_dtheta2_dot, dv_R_dtheta3_dot]

        for i in dv_R_dtheta_list: # flip sign pamy angular vs cartesian 
            i[0] = -i[0]
            i[1] = -i[1] 

        dx_aft = np.zeros((6, 7))
        for i in range(7):
            dR_dtheta_i = dR_dtheta_list[i]
            dv_R_dtheta_i = dv_R_dtheta_list[i]
            dv_aft_dtheta_i = ( R_inv @ M_impact @ dR_dtheta_i - R_inv @ dR_dtheta_i @ R_inv @ M_impact @ R ) @ ( vel_bef - v_R) - R_inv @ M_impact @ R @ dv_R_dtheta_i + dv_R_dtheta_i
            dx_aft[3:6, i] = dv_aft_dtheta_i[:]

        dx_hit = dx_hit_dx_aft @ dx_aft

        # -- numerical dR_dtheta to bugfix

        test_dR = False

        if test_dR:

            R = self.Rot_theta(theta)

            delta = 1e-7
            J = np.zeros((3, 3, 7)) 
            for i in range(7):

                print()
                print(' -- theta {}'.format(i))

                theta_alt = theta.copy()
                theta_alt[i] = theta_alt[i] + delta

                R_alt = self.Rot_theta(theta_alt)

                diff = (R_alt - R) / delta

                print('analytical: ', dR_dtheta_list[i])
                print('numerical: ', diff)
                print('diff: ', np.linalg.norm(dR_dtheta_list[i] - diff))
                print()

        # -- numerical dv_R_dtheta to bugfix

        test_v_R = False

        if test_v_R:

            delta = 1e-7
            print('---- v_R testing')
            for i in range(7):

                print()
                print(' -- theta {}'.format(i))

                theta_alt = theta.copy()
                theta_alt[i] = theta_alt[i] + delta


                v_R = self.RacketVelocityCartesian(theta)
                v_R_alt = self.RacketVelocityCartesian(theta_alt)

                diff = (v_R_alt - v_R) / delta

                print('analytical: ', dv_R_dtheta_list[i])
                print('numerical: ', diff)
                print('diff: ', np.linalg.norm(dv_R_dtheta_list[i] - diff))
                print()

        # -- numerical dx_aft (i.e. dx_aft / dtheta) to bugfix

        test_x_aft = False

        if test_x_aft:

            R = self.Rot_theta(theta)
            R_inv = np.linalg.inv(R)
            v_R = self.RacketVelocityCartesian(theta)
            vel_aft = R_inv@M_impact@R@(vel_bef - v_R) + v_R

            delta = 1e-7
            J = np.zeros((3, 7)) 
            print('---- x_aft testing')
            for col in range(7):
                theta_alt = theta.copy()
                theta_alt[col] = theta_alt[col] + delta

                R_alt = self.Rot_theta(theta_alt)
                R_inv_alt = np.linalg.inv(R_alt)
                v_R_alt = self.RacketVelocityCartesian(theta_alt)
                vel_aft_alt = R_inv_alt@M_impact@R_alt@(vel_bef - v_R_alt) + v_R_alt

                diff = (vel_aft_alt - vel_aft) / delta
                J[:, col] = diff[:]

            print('dx_aft num: ', J)
            print('dx_aft ana: ', dx_aft[3:, :])
            print('diff: ', dx_aft[3:, :] - J)
            print('diff norm: ', np.linalg.norm(dx_aft[3:, :] - J))
            print()

        # ---- numerical dx_hit_dx_aft

        test_dx_hit_dx_aft = False

        if test_dx_hit_dx_aft:

            delta = 1e-6
            dx_hit_dx_aft_num = np.identity(6)
            for inp in range(0, 6):

                dx_aft = x0.copy()
                dx_aft_alt = x0.copy()
                dx_aft_alt[inp] += delta

                x_running = dx_aft.copy()
                while x_running[2] > z_table:
                    t_to_impact = (x_running[5] + np.sqrt(x_running[5]**2 + 2*9.802*(x_running[2] - z_table))) / 9.802
                    if t_to_impact == 0:
                        break
                    if t_to_impact > 0.005:
                        T = 0.005
                    else:
                        T = t_to_impact
                    x_running = Falling(x_running, T)

                x_running_alt = dx_aft_alt.copy()
                while x_running_alt[2] > z_table:
                    t_to_impact = (x_running_alt[5] + np.sqrt(x_running_alt[5]**2 + 2*9.802*(x_running_alt[2] - z_table))) / 9.802
                    if t_to_impact == 0:
                        break
                    if t_to_impact > 0.005:
                        T = 0.005
                    else:
                        T = t_to_impact
                    x_running_alt = Falling(x_running_alt, T)

                dx_hit_dx_aft_num[:, inp] = ( x_running_alt - x_running ) / delta


            print()
            print('---- dx_hit_dx_aft testing')
            print('analytical: ', dx_hit_dx_aft)
            print('numerical: ', dx_hit_dx_aft_num)
            print('diff: ', np.linalg.norm(dx_hit_dx_aft - dx_hit_dx_aft_num))
            print()

        # ---- numerical dx_prehit_dx_aft
        
        test_dx_prehit_dx_aft = False

        if test_dx_prehit_dx_aft:

            delta = 1e-6
            dx_prehit_dx_aft_num = np.identity(6)
            for inp in range(3, 6):

                dx_aft = x0.copy()
                dx_aft_alt = x0.copy()
                dx_aft_alt[inp] += delta

                x_running = dx_aft.copy()
                while Falling(x_running, T)[2] > z_table:
                    x_running = Falling(x_running, T)

                x_running_alt = dx_aft_alt.copy()
                while Falling(x_running_alt, T)[2] > z_table:
                    x_running_alt = Falling(x_running_alt, T)

                dx_prehit_dx_aft_num[:, inp] = ( x_running_alt - x_running ) / delta

            print()
            print('---- dx_prehit_dx_aft testing')
            print('analytical: ', dx_prehit_dx_aft)
            print('numerical: ', dx_prehit_dx_aft_num)
            print('diff: ', np.linalg.norm(dx_prehit_dx_aft - dx_prehit_dx_aft_num))
            print()

        # ---- numerical dx_prehit_dtheta
        
        test_dx_prehit_dtheta = False

        if test_dx_prehit_dtheta:

            R = self.Rot_theta(theta)
            R_inv = np.linalg.inv(R)
            v_R = self.RacketVelocityCartesian(theta)
            vel_aft = R_inv@M_impact@R@(vel_bef - v_R) + v_R

            R_alt = self.Rot_theta(theta_alt)
            R_inv_alt = np.linalg.inv(R_alt)
            v_R_alt = self.RacketVelocityCartesian(theta_alt)
            vel_aft_alt = R_inv_alt@M_impact@R_alt@(vel_bef - v_R_alt) + v_R_alt

            delta = 1e-6
            dx_prehit_dx_aft_num = np.identity(6)
            for inp in range(0, 6):

                dx_aft = x0.copy()
                dx_aft_alt = x0.copy()
                dx_aft_alt[inp] += delta

                x_running = dx_aft.copy()
                while Falling(x_running, T)[2] > z_table:
                    x_running = Falling(x_running, T)

                x_running_alt = dx_aft_alt.copy()
                while Falling(x_running_alt, T)[2] > z_table:
                    x_running_alt = Falling(x_running_alt, T)

                dx_prehit_dx_aft_num[:, inp] = ( x_running_alt - x_running ) / delta

            print()
            print('---- dx_prehit_dx_aft testing')
            print('analytical: ', dx_prehit_dx_aft)
            print('numerical: ', dx_prehit_dx_aft_num)
            print('diff: ', np.linalg.norm(dx_prehit_dx_aft - dx_prehit_dx_aft_num))
            print()

        # ---- numerical dx_hit_dtheta

        test_dx_hit_dtheta = False

        if test_dx_hit_dtheta:

            delta = 1e-6

            dx_hit_dtheta_num = np.zeros((6, 7))

            R = self.Rot_theta(theta)
            R_inv = np.linalg.inv(R)
            v_R = self.RacketVelocityCartesian(theta)
            vel_aft = R_inv@M_impact@R@(vel_bef - v_R) + v_R

            x_running = np.zeros(6)
            x_running[0:3] = pos.copy().flatten()
            x_running[3:6] = vel_aft.copy().flatten()
            while x_running[2] > z_table:
                t_to_impact = (x_running[5] + np.sqrt(x_running[5]**2 + 2*9.802*(x_running[2] - z_table))) / 9.802
                if t_to_impact == 0:
                    break
                if t_to_impact > 0.005:
                    T = 0.005
                else:
                    T = t_to_impact
                x_running = Falling(x_running, T)

            delta = 1e-7

            print('---- dx_hit_dtheta testing')
            for col in range(7):
                theta_alt = theta.copy()
                theta_alt[col] = theta_alt[col] + delta

                R_alt = self.Rot_theta(theta_alt)
                R_inv_alt = np.linalg.inv(R_alt)
                v_R_alt = self.RacketVelocityCartesian(theta_alt)

                vel_aft_alt = R_inv_alt@M_impact@R_alt@(vel_bef - v_R_alt) + v_R_alt

                x_running_alt = np.zeros(6)
                x_running_alt[0:3] = pos.copy().flatten()
                x_running_alt[3:6] = vel_aft_alt.copy().flatten()
                while x_running_alt[2] > z_table:
                    t_to_impact = (x_running_alt[5] + np.sqrt(x_running_alt[5]**2 + 2*9.802*(x_running_alt[2] - z_table))) / 9.802
                    if t_to_impact == 0:
                        break
                    if t_to_impact > 0.005:
                        T = 0.005
                    else:
                        T = t_to_impact
                    x_running_alt = Falling(x_running_alt, T)

                diff = ( x_running_alt - x_running ) / delta
                dx_hit_dtheta_num[:, col] = diff[:]

            print('dx_hit num: ', dx_hit_dtheta_num)
            print('dx_hit ana: ', dx_hit)
            print('dx_hit ana alt: ', Jac_last_ana@dx_prehit_dx_aft)
            print('diff: ', dx_hit - dx_hit_dtheta_num)
            print('diff norm: ', np.linalg.norm(dx_hit - dx_hit_dtheta_num))
            print()

        # ---- numerical dy_hit_dtheta3

        test_dy_hit_dtheta3 = False

        if test_dy_hit_dtheta3:

            R = self.Rot_theta(theta)
            R_inv = np.linalg.inv(R)
            v_R = self.RacketVelocityCartesian(theta)
            vel_aft = R_inv@M_impact@R@(vel_bef - v_R) + v_R

            x_running = np.zeros(6)
            x_running[0:3] = pos.copy().flatten()
            x_running[3:6] = vel_aft.copy().flatten()
            iter = 0
            while x_running[2] > z_table:

                t_to_impact = (x_running[5] + np.sqrt(x_running[5]**2 + 2*9.802*(x_running[2] - z_table))) / 9.802
                if t_to_impact == 0:
                    break

                if t_to_impact > 0.005:
                    T = 0.005
                else:
                    T = t_to_impact

                iter += 1
                x_running = Falling(x_running, T)

            print('falling iterations: ', iter)

            print('INSIDE, vel_aft: ', vel_aft)
            print('INSIDE, theta: ', theta)
            print()

            delta = 1e-4

            print('---- dx_hit_dtheta3 testing')

            theta_alt = theta.copy()
            theta_alt[3] = theta_alt[3] + delta

            R_alt = self.Rot_theta(theta_alt)
            R_inv_alt = np.linalg.inv(R_alt)
            v_R_alt = self.RacketVelocityCartesian(theta_alt)

            vel_aft_alt = R_inv_alt@M_impact@R_alt@(vel_bef - v_R_alt) + v_R_alt

            x_running_alt = np.zeros(6)
            x_running_alt[0:3] = pos.copy().flatten()
            x_running_alt[3:6] = vel_aft_alt.copy().flatten()
            iter = 0
            while x_running_alt[2] > z_table:

                t_to_impact = (x_running_alt[5] + np.sqrt(x_running_alt[5]**2 + 2*9.802*(x_running_alt[2] - z_table))) / 9.802
                if t_to_impact == 0:
                    break

                if t_to_impact > 0.005:
                    T = 0.005
                else:
                    T = t_to_impact

                iter += 1
                x_running_alt = Falling(x_running_alt, T)

            print('falling iterations alt: ', iter)

            diff = ( x_running_alt - x_running ) / delta

            M = self.GetHitPoint(theta, vel_bef, pos)
            M_alt = self.GetHitPoint(theta_alt, vel_bef, pos)

            print('predicted impact in test: ', x_running)
            print('predicted impact in test from GetHitPoint: ', M)

            print('alt predicted impact in test: ', x_running_alt)
            print('alt predicted impact in test from GetHitPoint: ', M_alt)

            print('dy_hit_dtheta3 num: ', diff[1])
            print('dy_hit_dtheta3 num alt: ', (M_alt - M) / delta)
            print('dy_hit_dtheta3 ana: ', dx_hit[1, 3])
            print('diff: ', dx_hit[1, 3] - diff[1])
            print('diff norm: ', np.linalg.norm(dx_hit[1, 3] - diff[1]))
            print()

        return dx_hit

    def GetHitPoint(self, theta, v_bef, p0, z_table = -0.43, T_step_falling=0.005):
        '''
        For physical model: predict landing point based on interception policy, and the state before the racket impact
        '''
        
        gravity     = 9.802
        kd          = 0.1062

        M_impact    = np.array([[0.75,  0.0, 0.0], # simple impact matrix
                                [0.0, -0.75, 0.0],
                                [0.0,  0.0, 0.75]])

        R_inv   = self.Rot_theta(theta)
        R       = np.linalg.inv(R)
        v_R     = self.RacketVelocityCartesian(theta)


        v_aft = R_inv@M_impact@R@(v_bef - v_R) + v_R
        x_pred = np.vstack((p0.reshape(-1, 1), v_aft.reshape(-1, 1)))


        # apply Falling model until Ball is at height of table

        iter = 0

        while x_pred[2] > z_table:
            iter += 1
            v_norm = np.linalg.norm(x_pred[3:6])

            t_to_impact = (x_pred[5] + np.sqrt(x_pred[5]**2 + 2*gravity*(x_pred[2] - z_table))) / gravity
            if t_to_impact == 0:
                break

            if t_to_impact > T_step_falling:
                T = T_step_falling
            else:
                T = t_to_impact

            x_pred = np.array([ x_pred[0] + T * x_pred[3], 
                                x_pred[1] + T * x_pred[4], 
                                x_pred[2] + T * x_pred[5], 
                                x_pred[3] - T * kd * v_norm * x_pred[3], 
                                x_pred[4] - T * kd * v_norm * x_pred[4], 
                                x_pred[5] - T * kd * v_norm * x_pred[5] - T * gravity])

        hit_state = np.array([x_pred[0], x_pred[1], x_pred[3], x_pred[4], x_pred[5]])

        return hit_state.flatten()

    def ModelInput_NN(self, theta):
        '''
        prepare input the NN approach convenietly
        '''

        theta0_norm = (theta[0] - self.normalizers[0][0]) / self.normalizers[0][1]
        theta3_norm = (theta[1] - self.normalizers[1][0]) / self.normalizers[1][1]
        theta0_dot_norm = (theta[2] - self.normalizers[2][0]) / self.normalizers[2][1]

        return torch.DoubleTensor([theta0_norm, theta3_norm, theta0_dot_norm])

    def ModelInput_Physical(self, theta):
        '''
        prepare input the physical model approach convenietly
        '''

        theta0 = theta[0]
        theta1 = theta[1]
        theta2 = theta[2]

        x = math.cos(theta0) * math.sin(theta1) * self.l_1 + math.cos(theta0) * math.sin(theta1 + theta2) * self.l_2
        y = math.sin(theta0) * math.sin(theta1) * self.l_1 + math.sin(theta0) * math.sin(theta1 + theta2) * self.l_2
        z = math.cos(theta1) * self.l_1 + math.cos(theta1 + theta2) * self.l_2
        
        traj_T = self.position.copy(); traj_T = traj_T.T
        
        pos_racket = np.array([-x, -y, z])

        distances = [abs((pos_racket - i)[1]) for i in traj_T]
        k_impact = distances.index(min(distances))

        vel_bef = np.zeros(3)
        for i in range(1, 4):
            vel_bef = vel_bef + (self.position[:, k_impact-i] - self.position[:, k_impact-i-1]) / (self.t_stamp[k_impact-i] - self.t_stamp[k_impact-i-1])
        vel_bef = vel_bef / 3

        pos = traj_T[k_impact]

        return vel_bef.flatten(), pos, k_impact

    def Rot_theta(self, theta):
        
        theta0 = theta[0]
        theta1 = theta[1]
        theta2 = theta[2]
        theta3 = theta[3]

        R1 = np.array([ [math.cos(theta0),   -math.sin(theta0),  0],
                        [math.sin(theta0),   math.cos(theta0),   0], 
                        [0,                  0,                  1]])
        R2 = np.array([ [math.cos(-theta1),  0, math.sin(-theta1)],
                        [0,                  1,                  0], 
                        [-math.sin(-theta1), 0, math.cos(-theta1)]])
        R3 = np.array([ [math.cos(-theta2),   0,  math.sin(-theta2)],
                        [0,                  1,                  0], 
                        [-math.sin(-theta2),  0,  math.cos(-theta2)]])
        R4 = np.array([ [math.cos(theta3),  -math.sin(theta3), 0],
                        [math.sin(theta3),  math.cos(theta3),  0], 
                        [0,                  0,                  1]])

        R = R1@R2@R3@R4
        return np.linalg.inv(R)

    def RacketVelocityCartesian(self, theta):

        theta0 = theta[0]
        theta1 = theta[1]
        theta2 = theta[2]

        theta0_dot = theta[4]
        theta1_dot = theta[5]
        theta2_dot = theta[6]

        x_dot = self.l_1*( math.cos(theta0)*math.cos(theta1)*theta1_dot - math.sin(theta0)*math.sin(theta1)*theta0_dot ) + self.l_2*( math.cos(theta0)*math.cos(theta1 + theta2)*(theta1_dot + theta2_dot) - math.sin(theta0)*math.sin(theta1 + theta2)*theta0_dot)
        y_dot = self.l_1*( math.sin(theta0)*math.cos(theta1)*theta1_dot + math.cos(theta0)*math.sin(theta1)*theta0_dot ) + self.l_2*( math.sin(theta0)*math.cos(theta1 + theta2)*(theta1_dot + theta2_dot) + math.cos(theta0)*math.sin(theta1 + theta2)*theta0_dot)
        z_dot = self.l_1*(-math.sin(theta1)*theta1_dot) + self.l_2*(-math.sin(theta1 + theta2)*(theta1_dot + theta2_dot))

        return np.array([-x_dot, -y_dot, z_dot])

    def impact_detection_uncalibrated(self, pos_loc, z_table = -0.44):
        n = len(pos_loc[0])

        table_impacts = [] # save steps so that each is a step right after impact
        for k in range(2, n-2):

            diff_before_1 = pos_loc[2, k-1] - pos_loc[2, k-2]
            diff_before_2 = pos_loc[2, k] - pos_loc[2, k-1]

            diff_after_1 = pos_loc[2, k+1] - pos_loc[2, k]
            diff_after_2 = pos_loc[2, k+2] - pos_loc[2, k+1]

            if diff_before_1 < 0 and diff_before_2 < 0 and diff_after_1 > 0 and diff_after_2 > 0:
                if pos_loc[2, k] > z_table - 0.1 and pos_loc[2, k] < z_table + 0.1:
                    table_impacts.append(k)

        racket_impact = -1

        for k in range(5, n-5):

            pos_diffs_before = []
            for i in range(0, 4):
                pos_diffs_before.append(pos_loc[1, k-i] - pos_loc[1, k-i-1])

            pos_diffs_after = []
            for i in range(0, 4):
                pos_diffs_after.append(pos_loc[1, k+i+1] - pos_loc[1, k+i])

            # impact if: positions before go down, positions after go up, velocity before is negative enough, velocity after is positive enough
            if max(pos_diffs_before) < 0 and min(pos_diffs_after) > 0:
                racket_impact = k

        return table_impacts, racket_impact

    def impact_detection(self, pos_loc, z_table = -0.43):
        n = len(pos_loc[0])

        table_impacts = [] # save steps so that each is a step right after impact
        for k in range(2, n-2):

            diff_before_1 = pos_loc[2, k-1] - pos_loc[2, k-2]
            diff_before_2 = pos_loc[2, k] - pos_loc[2, k-1]

            diff_after_1 = pos_loc[2, k+1] - pos_loc[2, k]
            diff_after_2 = pos_loc[2, k+2] - pos_loc[2, k+1]

            if diff_before_1 < 0 and diff_before_2 < 0 and diff_after_1 > 0 and diff_after_2 > 0:
                if pos_loc[2, k] > z_table - 0.3 and pos_loc[2, k] < z_table + 0.3:
                    table_impacts.append(k)

        racket_impact = -1

        for k in range(5, n-5):

            pos_diffs_before = []
            for i in range(0, 4):
                pos_diffs_before.append(pos_loc[1, k-i] - pos_loc[1, k-i-1])

            pos_diffs_after = []
            for i in range(0, 4):
                pos_diffs_after.append(pos_loc[1, k+i+1] - pos_loc[1, k+i])

            # impact if: positions before go down, positions after go up, velocity before is negative enough, velocity after is positive enough
            if min(pos_diffs_before) > 0 and max(pos_diffs_after) < 0:
                # racket impacts only: 
                if pos_loc[0, k] > -0.8 and pos_loc[0, k] < -0.2:
                    if pos_loc[1, k] > -0.7 and pos_loc[1, k] < 0.7:
                        if pos_loc[2, k] > -0.3 and pos_loc[2, k] < 0.3:
                            racket_impact = k

        return table_impacts, racket_impact

    def PlotThetaHisotry(self):
        if len(self.theta_history)>1:

            fig, axs = plt.subplots(4, 2)
            theta_hist = np.array(self.theta_history).T
            for j in range(4):
                axs[j, 0].plot(theta_hist[j, :])
                axs[j, 0].set_title('DoF {}'.format(j))
                if j == 3:
                    break
                axs[j, 1].plot(theta_hist[j+3, :])
                axs[j, 1].set_title('DoF {} - Derivative'.format(j))
            fig.suptitle('Theta History')
        else:
            print('only one run, no plot done')
        return