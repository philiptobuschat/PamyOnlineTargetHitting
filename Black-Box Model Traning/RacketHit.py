import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import scipy.optimize

from RealBall import RealBall

# %%

class HitRecording():
    '''
    For data analysis of recordings of ball impacts on racket.
    '''
    def __init__(self, index=None, path=None, load_prints=False, rho_limit=0.83):
        
        self.index=index
        loading = []
        self.load = True
        for file in ['launcher', 'estimator', 'robot', 'feedforward', 'predictor']:
            try:
                f = open(path+'Iterator_{}_{}'.format(self.index, file), 'rb')
                content = []
                while True:
                    try:
                        content.append(pickle.load(f))
                    except:
                        break
                loading.append(content)
            except:
                if load_prints:
                    print('failed to load file {}'.format(file))
                self.load = False

        if self.load:

            self.launcher   = loading[0][0]

            self.estimator  = loading[1]
            self.t_stamp    = self.estimator[0][0]
            self.position   = self.estimator[1]             # in calibrated coordinate system
            self.x_kf       = np.array(self.estimator[2])   # in uncalibrated coordinate system

            # cut zeros at end of recordings
            if self.t_stamp[-1] == 0:
                k_last = len(self.t_stamp) - 1
                while self.t_stamp[k_last] == 0:
                    k_last -= 1
                self.t_stamp    = self.t_stamp[:k_last+1]
                self.position   = self.position[:, :k_last+1]

            self.robot              = loading[2]
            self.theta_list         = np.array(self.robot[0])
            self.reference_list     = np.array(self.robot[1])
            self.robot_t_stamp      = np.array(self.robot[2])

            try: # on some recording pressure input data is available
                self.press_hist         = np.array(self.robot[3])
                self.press_hist_set     = np.array(self.robot[4])
            except:
                self.press_hist     = None
                self.press_hist_set = None

            self.feedforward        = loading[3]
            self.reference_polar    = self.feedforward[0]
            self.velocity_polar     = self.feedforward[1]
            self.acceleration_polar = self.feedforward[2]
            self.reference_angle    = self.feedforward[3]
            
            self.predictor  = loading[4]
            self.hit_points = self.predictor[0]
            self.hit_times  = self.predictor[1]

            # recalibrate positions
            self.A = np.array( [[ -0.99591269,  0.07944872, 0.04296286],
                                [ -0.07685274, -0.99529339, 0.05903159],
                                [  0.04745064,  0.0554885,  0.99733117]]) # from calibration 2022_09_22

            self.CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

            if len(self.t_stamp) != len(self.position[0, :]) or len(self.t_stamp) != len(self.x_kf):
                self.load = False
                if load_prints:
                    print('loading error in t_stamp / position / x_kf')
                return

            if self.x_kf[0, 1, 0] / self.position[1, 0] < 0: # recalibrate KF data if necessary (not needed for early recordings)
                for k in range(len(self.t_stamp)):
                    self.x_kf[k, 0:3, 0] = self.A@(self.x_kf[k, 0:3, 0] + self.CALIBRATION)
                    self.x_kf[k, 3:6, 0] = self.A@self.x_kf[k, 3:6, 0]
                    self.x_kf[k, 6:9, 0] = self.A@self.x_kf[k, 6:9, 0]

            self.GetTableImpacts()

            if len(self.table_impacts) == 0:
                self.load = False
                if load_prints:
                    print('no table impact found')
                return

            self.GetRacketImpact()

            if self.racket_impact == -1:
                self.load = False
                if load_prints:
                    print('no racket impact found in HitRecording')
                return
            
            if not self.load:
                return

            v_R = self.RacketVelocityCartesian(self.racket_impact_x[30:])

            vel_bef = self.racket_impact_vel_bef - v_R
            vel_aft = self.racket_impact_vel_aft - v_R
            vel_comb = (1/np.linalg.norm(vel_aft - vel_bef)) * (vel_aft - vel_bef)

            R_inv = np.linalg.inv(self.Rot_theta(self.racket_impact_x[30:]))
            racket_vec = R_inv@np.array([0, -1, 0])

            self.rho = np.dot(vel_comb, racket_vec)
            if self.rho < rho_limit:
                self.load = False
                if load_prints:
                    print('rho < rho limit ({}): edge hit likely'.format(rho_limit))
                return

    def Falling(self, x, T):
        '''
        free-falling ball model discrete step
        '''
        gravity = 9.802
        kd = 0.1062
        v_norm = np.linalg.norm(x[3:6])
        x_pred = np.array([ x[0] + T * x[3], 
                            x[1] + T * x[4], 
                            x[2] + T * x[5] - 0.5 * T**2 * gravity, 
                            x[3] - T * kd * v_norm * x[3], 
                            x[4] - T * kd * v_norm * x[4], 
                            x[5] - T * kd * v_norm * x[5] - T * gravity])
        return x_pred

    def Rot_theta(self, theta):
        '''
        rotation matrix 
        '''
        theta0 = theta[0];      theta1 = theta[1];      theta2 = theta[2];      theta3 = theta[3]
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
        '''
        racket velocity in calibrated cartesian system from angular space representation
        '''
        l_1=0.3768
        l_2=0.4038
        theta0 = theta[0];      theta1 = theta[1];      theta2 = theta[2]
        theta0_dot = theta[4];  theta1_dot = theta[5];  theta2_dot = theta[6]
        x_dot = l_1*( math.cos(theta0)*math.cos(theta1)*theta1_dot - math.sin(theta0)*math.sin(theta1)*theta0_dot ) + l_2*( math.cos(theta0)*math.cos(theta1 + theta2)*(theta1_dot + theta2_dot) - math.sin(theta0)*math.sin(theta1 + theta2)*theta0_dot)
        y_dot = l_1*( math.sin(theta0)*math.cos(theta1)*theta1_dot + math.cos(theta0)*math.sin(theta1)*theta0_dot ) + l_2*( math.sin(theta0)*math.cos(theta1 + theta2)*(theta1_dot + theta2_dot) + math.cos(theta0)*math.sin(theta1 + theta2)*theta0_dot)
        z_dot = l_1*(-math.sin(theta1)*theta1_dot) + l_2*(-math.sin(theta1 + theta2)*(theta1_dot + theta2_dot))
        return np.array([-x_dot, -y_dot, z_dot])

    def PlotRobotStates(self):
        '''
        plot robot arm DoFs over time and reference trajectory
        '''
        fig, axs = plt.subplots(2, 2)
        for i in range(4):

            axs[i%2, int(i/2)].plot(self.theta_list[:, i])
            axs[i%2, int(i/2)].plot(self.reference_list[:, i], ls='--')
            axs[i%2, int(i/2)].set_title('DoF {}'.format(i+1))
        
        fig.suptitle('index = {}'.format(self.index))
        return

    def GetTableImpacts(self):
        '''
        find table impacts, save for each impact the iteration number following the impact
        '''
        n = len(self.t_stamp)
        self.table_impacts = []
        for k in range(5, n-5):

            pos_diffs_before = []
            vels_before = []

            for i in range(0, 4):
                pos_diffs_before.append(self.position[2, k-i] - self.position[2, k-i-1])
                vels_before.append( pos_diffs_before[-1] / (self.t_stamp[k-i] - self.t_stamp[k-i-1]) )

            pos_diffs_after = []
            vels_after = []

            for i in range(0, 4):
                pos_diffs_after.append(self.position[2, k+i+1] - self.position[2, k+i])
                vels_after.append( pos_diffs_after[-1] / (self.t_stamp[k+i+1] - self.t_stamp[k+i]) )

            # impact if: z-position before goes down, z-position after goes up, z-velocity before is negative enough, z-velocity after is positive enough
            if max(pos_diffs_before) < 0 and min(pos_diffs_after) > 0 and np.mean(vels_before) < -1 and np.mean(vels_after) > 1:
                self.table_impacts.append(k)
        return

    def GetRacketImpact(self, delta_t=0.01):
        '''
        find racket impacts, save for each impact the iteration number following the impact
        '''
        n = len(self.t_stamp)
        self.racket_impact = -1
        for k in range(5, n-5):

            pos_diffs_before = []
            for i in range(0, 4):
                pos_diffs_before.append(self.position[1, k-i] - self.position[1, k-i-1])

            pos_diffs_after = []
            for i in range(0, 4):
                pos_diffs_after.append(self.position[1, k+i+1] - self.position[1, k+i])

            # impact if: y-position before increases, y-position after decreases, position in general close enough to robot arm's origin 
            if min(pos_diffs_before) > 0 and max(pos_diffs_after) < 0:
                if self.position[0, k] > -1 and self.position[0, k] < -0:
                    if self.position[1, k] > -1 and self.position[1, k] < 0.5:
                        if self.position[2, k] > -0.5 and self.position[2, k] < 0.5:
                            self.racket_impact = k

        self.racket_impact_robot = -1

        if self.racket_impact != -1: # impact found, calculate iteration number in robot data corresponding to impact and extract impact parameters
            time_impact = self.t_stamp[self.racket_impact]

            # ball position measurements around racket impact usually not available, due to occluded vision
            # extrapolate ball trajectories before and after to find probable actual impact time/position: 

            if self.table_impacts[0] > self.racket_impact:
                #print('first table impact not detectable')
                self.racket_impact = -1
                return

            if self.racket_impact - self.table_impacts[0] < 5:
                #print('not enough point between table impact and racket impact')
                self.racket_impact = -1
                return

            fit_bef = [np.polyfit(self.t_stamp[self.table_impacts[0]:self.racket_impact], self.position[i, self.table_impacts[0]:self.racket_impact], 2) for i in range(3)]
            fit_aft = [np.polyfit(self.t_stamp[self.racket_impact+1:self.racket_impact+10], self.position[i, self.racket_impact+1:self.racket_impact+10], 2) for i in range(3)]

            fcn_bef = [np.poly1d(fit) for fit in fit_bef]
            fcn_aft = [np.poly1d(fit) for fit in fit_aft]

            distances = []
            t_test = np.linspace(time_impact-0.2, time_impact+0.2, 1000)
            for t in t_test:
                dist_vec = np.array([(fcn_bef[i](t) - fcn_aft[i](t)) for i in range(3)])
                distances.append(np.linalg.norm(dist_vec))

            # estimate of acutal impact time
            self.time_impact_intersection = t_test[distances.index(min(distances))]

            # roll out last state before impact until estimate of actual impact time 
            x_running = self.x_kf[self.racket_impact-1, 0:6, 0].copy()
            t_running = self.t_stamp[self.racket_impact-1].copy()
            while t_running < self.time_impact_intersection:
                if self.time_impact_intersection - t_running > 0.005:
                    T = 0.005
                else:
                    T = self.time_impact_intersection - t_running
                t_running += T
                x_running = self.Falling(x_running, T)
            x_bef = x_running.copy().flatten() # estimated state right before racket impact

            self.racket_impact_position = x_bef[0:3]

            robot_time_dist = [abs(i-self.time_impact_intersection) for i in self.robot_t_stamp]
            self.racket_impact_robot = robot_time_dist.index(min(robot_time_dist))

            if self.racket_impact_robot+1 >= len(self.theta_list[:, 0]):
                self.load = False
                return
                
            # collect inputs and labels for data-driven racket impact model
            # inputs: incoming trajectory, robot state at interception
            # labels: landing point of ball after interceptzon
            self.incoming_traj = self.position[:, self.racket_impact - 10:self.racket_impact].T.flatten()
            self.robot_angles = self.theta_list[self.racket_impact_robot, :]
            self.robot_velocities = (self.theta_list[self.racket_impact_robot+1, :] - self.theta_list[self.racket_impact_robot, :]) / (delta_t)
            self.robot_velocities = self.robot_velocities[0:3] # assume last DoF is stationary so remove it

            self.reference_angles = self.reference_list[self.racket_impact_robot, :]
            self.reference_velocities = (self.reference_list[self.racket_impact_robot+1, :] - self.reference_list[self.racket_impact_robot, :]) / (delta_t)

            self.racket_impact_x = np.hstack((self.incoming_traj, self.robot_angles, self.robot_velocities))

            self.racket_impact_vel_bef = x_bef[3:6]
            self.racket_impact_vel_aft = self.FitVelocityAfterRacketImpact()

            self.racket_impact_y = self.GetHitPoint()

    def FitVelocityAfterRacketImpact(self, xtol=1e-8):
        '''
        find optimal initial ball state after racket impact to describe given outgoing trajectory
        '''
        global t_fit, r_fit

        def traj(p):
            global t_fit, r_fit

            x_traj = [r_fit[0, 0]]
            y_traj = [r_fit[1, 0]]
            z_traj = [r_fit[2, 0]]

            x_curr = np.zeros(6)
            x_curr[0:3] = r_fit[:, 0]
            x_curr[3:6] = p.copy()

            x_save = np.zeros((6, len(t_fit)))
            x_save[:, 0] = x_curr[0:6]

            converge_flag = True

            for k in range(1, len(t_fit)):

                if np.linalg.norm(x_curr) > 1e5:
                    converge_flag = False
                    break

                T_k = t_fit[k] - t_fit[k-1]
                x_next = self.Falling(x_curr, T_k)

                x_traj.append(x_next[0].item())
                y_traj.append(x_next[1].item())
                z_traj.append(x_next[2].item())

                x_curr = x_next.copy()

                x_save[:, k] = x_curr[0:6]
            
            return x_traj, y_traj, z_traj, converge_flag, x_save

        def err(p):
            global t_fit, r_fit

            x_traj, y_traj, z_traj, converge_flag, _ = traj(p)

            if not converge_flag:
                return 1e10

            traj_tot = np.array([x_traj, y_traj, z_traj])

            return np.linalg.norm(r_fit - traj_tot)

        k_end_cut_off = len(self.t_stamp)

        errors_cut_off_x = []
        errors_cut_off_z = []

        for k_end in range(self.racket_impact + 5, len(self.t_stamp)):

            test_px = np.poly1d( np.polyfit(self.t_stamp[self.racket_impact:k_end], self.position[0, self.racket_impact:k_end], 2) )
            test_pz = np.poly1d( np.polyfit(self.t_stamp[self.racket_impact:k_end], self.position[2, self.racket_impact:k_end], 2) )

            err_cut_off_x = 0
            err_cut_off_z = 0

            for j in range(self.racket_impact, k_end):

                err_cut_off_x += (test_px(self.t_stamp[j]) - self.position[0, j])**2
                err_cut_off_z += (test_pz(self.t_stamp[j]) - self.position[2, j])**2

            errors_cut_off_x.append(err_cut_off_x)
            errors_cut_off_z.append(err_cut_off_z)

            err_tot= err_cut_off_x + err_cut_off_z
            if err_tot>1e-3:
                k_end_cut_off = k_end - 5
                break

        t_fit = self.t_stamp[self.racket_impact+1:k_end_cut_off]
        r_fit = self.position[:, self.racket_impact+1:k_end_cut_off]

        if k_end_cut_off-self.racket_impact-1 < 5:
            self.load = False
            return 0.0

        k_start = self.racket_impact

        ball_velocity = np.zeros(3)
        for i in range(1, 4):
            ball_velocity = ball_velocity + ( self.position[:, k_start+1+i] - self.position[:, k_start+i] ) / (self.t_stamp[k_start+1+i] - self.t_stamp[k_start+i])
        ball_velocity = ball_velocity / 3

        p0 = ball_velocity.copy()

        p_opt = scipy.optimize.least_squares(err, p0, xtol=xtol)

        return p_opt.x

    def GetHitPoint(self, z_table = -0.44, T_step_falling=0.005):
        '''
        return either acutal recorded landing point of ball on table after racket hit or calculated point where ball crosses table height. 
        '''

        # transform into uncalibrated system, where table is even and landing point predictions more accurate
        pos_uncal = []
        for k in range(len(self.t_stamp.flatten())):
            pos_uncal.append( np.linalg.inv(self.A)@self.position[:, k] - self.CALIBRATION )
        pos_uncal = np.array(pos_uncal).T

        table_impacts, racket_impact = self.impact_detection_uncalibrated(pos_uncal, z_table = z_table)

        if racket_impact == -1:
            self.load = False
            return

        table_impact = -1
        for i in table_impacts: # use first impact on table after racket impact for the following
            if i > racket_impact:
                table_impact = i
                break

        Ball = RealBall(model = None,
                        center_of_table=np.array([0.80, 1.71, z_table]),
                        x_of_table=np.array([-1, 0]),
                        y_of_table=np.array([0, -1]),
                        height_of_ground= z_table -0.76,
                        height_of_table=0.0)

        ball_position = pos_uncal[:, racket_impact]
        ball_velocity = (pos_uncal[:, racket_impact+1] - pos_uncal[:, racket_impact]) / (self.t_stamp[racket_impact+1] - self.t_stamp[racket_impact])
        Ball.StateEstimation(ball_position, ball_velocity)

        v_meas = []
        v_filt = []

        k_end_cut_off = len(self.t_stamp) - 3
        errors_cut_off_x = []
        errors_cut_off_z = []
        for k_end in range(self.racket_impact + 5, len(self.t_stamp)):
            test_px = np.poly1d( np.polyfit(self.t_stamp[self.racket_impact:k_end], pos_uncal[0, self.racket_impact:k_end], 2) )
            test_pz = np.poly1d( np.polyfit(self.t_stamp[self.racket_impact:k_end], pos_uncal[2, self.racket_impact:k_end], 2) )
            err_cut_off_x = 0
            err_cut_off_z = 0
            for j in range(self.racket_impact, k_end):
                err_cut_off_x += (test_px(self.t_stamp[j]) - pos_uncal[0, j])**2
                err_cut_off_z += (test_pz(self.t_stamp[j]) - pos_uncal[2, j])**2
            errors_cut_off_x.append(err_cut_off_x)
            errors_cut_off_z.append(err_cut_off_z)
            err_tot= err_cut_off_x + err_cut_off_z
            if err_tot>5e-3:
                k_end_cut_off = k_end - 6
                break

        pos_filt = []

        step = 1
        while Ball.x_meas[2] > z_table and step < len(self.t_stamp)-racket_impact: # Kalman Filter data until table height reached, no more data, or detected table impact(possibly slightly above set table height)
            pos_filt.append(Ball.x_meas.copy().flatten())
            ball_meas = pos_uncal[:, racket_impact+step]
            t_step = self.t_stamp[racket_impact+step] - self.t_stamp[racket_impact+step-1]
            Ball.StateEstimation(ball_meas, np.zeros(3), k=step+racket_impact ,step_length=t_step)

            v_filt.append(Ball.x_meas.copy().flatten()[3:6])
            v_meas.append( (pos_uncal[:, racket_impact+step] - pos_uncal[:, racket_impact+step-1]) / t_step)

            if step+racket_impact == table_impact or step+racket_impact == k_end_cut_off:
                break

            step += 1

        if step+racket_impact == table_impact or Ball.x_meas[2] <= z_table: # either impact on table or ball crossed table height limit
            
            x_pred_cal = np.zeros(6)
            x_pred_cal[0:3] = self.A@(Ball.x_meas[0:3].copy().flatten() + self.CALIBRATION)
            x_pred_cal[3:6] = self.A@Ball.x_meas[3:6].copy().flatten()

            return np.array([x_pred_cal[0], x_pred_cal[1], x_pred_cal[3], x_pred_cal[4], x_pred_cal[5]]).flatten()

        x_pred = Ball.x_meas
        x_falling = []
        y_falling = []
        z_falling = []
        pos_pred = []

        while x_pred[2] > z_table: # apply Falling model until Ball is at height of table
            pos_pred.append(x_pred.copy().flatten())
            x_falling.append(x_pred[0])
            y_falling.append(x_pred[1])
            z_falling.append(x_pred[2])
            x_pred = Ball.Falling(x_pred, T=T_step_falling)
            step+=1

        x_pred_cal = np.zeros(6)
        x_pred_cal[0:3] = self.A@(x_pred[0:3].flatten() + self.CALIBRATION)
        x_pred_cal[3:6] = self.A@x_pred[3:6].flatten()

        return np.array([x_pred_cal[0], x_pred_cal[1], x_pred_cal[3], x_pred_cal[4], x_pred_cal[5]]).flatten()

    def impact_detection_uncalibrated(self, pos_loc, z_table = -0.44):
        '''
        find impacts with uncalibrated positions
        '''
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

            if max(pos_diffs_before) < 0 and min(pos_diffs_after) > 0:
                racket_impact = k

        return table_impacts, racket_impact

    def PlotBallStates(self):
        n = len(self.t_stamp)

        pos_x = [self.position[0, k] for k in range(n)]
        pos_y = [self.position[1, k] for k in range(n)]
        pos_z = [self.position[2, k] for k in range(n)]

        v = []

        for k in range(n - 1):
            v_k = (self.position[:, k+1] - self.position[:, k]) / (self.t_stamp[k+1] - self.t_stamp[k])
            v.append(v_k)

        fig = plt.figure( figsize=(12, 12) )
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.set_ylabel('velocity [m/s]')
        ax2.plot(v, label=['x$_{meas}$', 'y$_{meas}$', 'z$_{meas}$'])
        ax1.plot(pos_x, label='x$_{meas}$')
        ax1.plot(pos_y, label='y$_{meas}$')
        ax1.plot(pos_z, label='z$_{meas}$')
        ax1.set_ylabel('position [m]')

        pos_x_kf = [self.x_kf[k, 0, 0] for k in range(n)]
        pos_y_kf = [self.x_kf[k, 1, 0] for k in range(n)]
        pos_z_kf = [self.x_kf[k, 2, 0] for k in range(n)]

        vel_x_kf = [self.x_kf[k, 3, 0] for k in range(n)]
        vel_y_kf = [self.x_kf[k, 4, 0] for k in range(n)]
        vel_z_kf = [self.x_kf[k, 5, 0] for k in range(n)]

        omega_x_kf = [self.x_kf[k, 6, 0] for k in range(n)]
        omega_y_kf = [self.x_kf[k, 7, 0] for k in range(n)]
        omega_z_kf = [self.x_kf[k, 7, 0] for k in range(n)]
        
        ax2.plot(vel_x_kf, label='x$_{KF}$')
        ax2.plot(vel_y_kf, label='y$_{KF}$')
        ax2.plot(vel_z_kf, label='z$_{KF}$')

        ax1.plot(pos_x_kf, label='x$_{KF}$')
        ax1.plot(pos_y_kf, label='y$_{KF}$')
        ax1.plot(pos_z_kf, label='z$_{KF}$')

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)

        ax3.plot(omega_x_kf, label='x$_{KF}$')
        ax3.plot(omega_y_kf, label='y$_{KF}$')
        ax3.plot(omega_z_kf, label='z$_{KF}$')
        ax3.legend()
        ax3.set_ylabel('spin [rad/s]')
        ax3.set_xlabel('k [-]')
    
        ax2.legend()
        ax1.legend()

        fig.suptitle('KF estimate of states')

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

    def Plot3DEstimation( self, y_to_compare=None, y_real=None):

        if y_real is None:
            y_real = self.position

        fig = plt.figure( figsize=(8, 8) )
        ax = plt.subplot(111, projection='3d')

        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)

        if y_to_compare is not None:

            for i in range(len(y_to_compare)):
                ax.scatter(y_to_compare[i][0, :], y_to_compare[i][1, :], y_to_compare[i][2, :], c='red', s=0.5)
                ax.plot3D(y_to_compare[i][0, :], y_to_compare[i][1, :], y_to_compare[i][2, :],'grey', linewidth=0.1)

        ax.scatter(y_real[0, 0:len(self.t_stamp)], y_real[1, 0:len(self.t_stamp)], y_real[2, 0:len(self.t_stamp)], c='black', s=5)
        ax.plot3D(y_real[0, 0:len(self.t_stamp)], y_real[1, 0:len(self.t_stamp)], y_real[2, 0:len(self.t_stamp)],'grey', linewidth=0.1, label=r'real')

        plt.legend(ncol=1, loc='upper center', shadow=True, fontsize=14)
        ax.set_xlabel(r'$x$ in m', fontsize=14)
        ax.set_ylabel(r'$y$ in m', fontsize=14)   
        ax.set_zlabel(r'$z$ in m', fontsize=14)
        self.set_axes_equal(ax)   
