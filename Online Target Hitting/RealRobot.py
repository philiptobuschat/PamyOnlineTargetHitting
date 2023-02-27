import math
import numpy as np
import o80
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from RealJoint import Joint
from RealGenerateMatrices import Filter
import time

class Robot:
    def __init__(self, frontend, dof_list, model_num, model_den,
                 model_num_order, model_den_order, model_ndelay_list,
                 inverse_model_num, inverse_model_den, order_num_list, order_den_list,
                 ndelay_list, anchor_ago_list, anchor_ant_list, strategy_list, pid_list,
                 delta_d_list, delta_y_list, delta_w_list, delta_ini_list, 
                 pressure_min, pressure_max, weight_list,
                 A_list=None, A_bias=None, cnn_model_list=None):
        '''
        This class is used to build the robot.
        frontend: connect to the backend of pamy
        dof_list: the list of dofs that you want to contol
        model_num, model_den, model_ndelay_list: parameters that describe the discrete linear forward 
                                                 model of each dof
        inverse_model_num, inverse_model_den, ndelay_list: parameters that describe the discrete linear
                                                           inverse model of each dof
        ''' 

        # connect to the backend
        self.frontend = frontend
        # all dofs
        self.dof_list = dof_list
        # linear model
        self.model_num = model_num
        self.model_den = model_den
        self.model_num_order = model_num_order
        self.model_den_order = model_den_order
        self.model_ndelay_list = model_ndelay_list
        # inverse model
        self.inverse_model_num = inverse_model_num
        self.inverse_model_den = inverse_model_den
        self.order_num_list = order_num_list
        self.order_den_list = order_den_list
        self.ndelay_list = ndelay_list
        # anchor pressures for each dof
        self.anchor_ago_list = anchor_ago_list
        self.anchor_ant_list = anchor_ant_list
        # control strategies
        self.strategy_list = strategy_list
        # pid controller to return to initial posture
        self.pid_list = pid_list
        # covariance parameters for ILC
        self.delta_d_list = delta_d_list
        self.delta_y_list = delta_y_list
        self.delta_w_list = delta_w_list
        self.delta_ini_list = delta_ini_list
        # set bounds for optimization
        self.pressure_min = pressure_min
        self.pressure_max = pressure_max
        self.weight_list = weight_list
        # define four joints
        self.Joint_list = []
        '''
        Each dof can be seen as independent.
        '''
        for dof in self.dof_list:
            Joint_temp = Joint(self.frontend, self.dof_list[dof], self.anchor_ago_list[dof], self.anchor_ant_list[dof], 
                               self.inverse_model_num[dof, :], self.inverse_model_den[dof, :],
                               self.order_num_list[dof], order_den_list[dof], ndelay_list[dof], 
                               self.pid_list[dof, :], self.strategy_list[dof])
            self.Joint_list.append( Joint_temp )
        
        frequency_backend = 500
        frequency_frontend = 50
        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        self.step_size = 1 / frequency_frontend
        self.iterations_per_command = int( period_frontend / period_backend )  # sychronize the frontend and the backend

        self.pid_for_tracking = np.array([[-13000, 0, -300],
                                          [-80000, 0, -300],
                                          [-5000, -8000, -100],
                                          [3.422187330173758e+04, 1.673663594798479e+05 / 10, 73.238165769446297]])

        self.A_list = A_list 
        self.A_bias = A_bias
        self.cnn_model_list = cnn_model_list

        # length should be long enough
        self.trajectory_temp = np.zeros((4, 1000))
        self.trajectory_history = []
        self.trajectory_real = np.zeros((4, 1000))
        self.p_in_cylinder = np.zeros((3, 1000))
        self.v_in_cylinder = np.zeros((3, 1000))
        self.a_in_cylinder = np.zeros((3, 1000))
        self.p_to_check = np.zeros((3, 1000))
        self.ff = np.zeros((4, 1000))
        self.fb = np.zeros((4, 1000))
    
    def GetOptimizer(self, angle_initial, total_iteration=40, mode_name='none'):
        '''
        y_desired:       relative desired trajectories
        t_stamp:         time stamp for the trajectories
        angle_initial:   initial angles
        total_iteration: the number of iterations for ILC
        mode_name:       training mode
                         'none' - train without feedback inputs
                         'pd'   - train with pd controller

        y_desired + angle_initial = absolute trajectory
        '''
        self.Optimizer_list = []
        for dof in self.dof_list:
            Optimizer = Filter(dof, self.y_desired[dof, :], self.y_desired.shape[1],
                                    self.pressure_min[dof], self.pressure_max[dof],
                                    self.model_num[dof, :], self.model_den[dof, :],
                                    self.model_num_order[dof], self.model_den_order[dof],
                                    self.model_ndelay_list[dof], angle_initial, 
                                    total_iteration=total_iteration )
            Optimizer.GenerateGlobalMatrix(mode_name)
            self.Optimizer_list.append( Optimizer )
        
    def ImportTrajectory(self, y_desired, t_stamp ):
        '''
        This function is used to update the desired trjectories of the Robot.
        The desired trajectories should be relative trajectories.
        '''
        self.y_desired = np.copy( y_desired )
        self.t_stamp = t_stamp
        
    def Feedforward(self, y_list=None, angle_initial_list=None):
        '''
        here y_list should be the relative angle
        '''
        if y_list is None:
           y_list = np.copy( self.y_desired )
        # basic pressure for ago
        u_ago = np.array([])
        # basic pressure for ant
        u_ant = np.array([])
        # feedforward control
        ff = np.array([])
        for dof in self.dof_list:
            (u_ago_temp, u_ant_temp, ff_temp) = self.Joint_list[dof].Feedforward(y_list[dof, :])
            u_ago = np.append(u_ago, u_ago_temp)
            u_ant = np.append(u_ant, u_ant_temp)
            ff = np.append(ff, ff_temp)
        ff = ff.reshape(len(self.dof_list), -1)
        u_ago = u_ago.reshape(len(self.dof_list), -1)
        u_ant = u_ant.reshape(len(self.dof_list), -1)
        return(u_ago, u_ant, ff)

    def Control(self, y_list=None, mode_name_list=["fb+ff", "fb+ff", "fb+ff", "fb+ff"], 
                mode_trajectory="ref",
                frequency_frontend=100, frequency_backend=500,
                ifplot="yes", u_ago=None, u_ant=None, ff=None, echo="no", controller='pid'):
        # import the reference trajectory
        if y_list is None:
            y_list = np.copy( self.y_desired )
        # generate the corresponding time stamp
        t_stamp_u = np.linspace(0, (y_list.shape[1] - 1) / frequency_frontend, y_list.shape[1] )

        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )  # sychronize the frontend and the backend
        
        # read the actual current states
        theta = self.frontend.latest().get_positions()
        theta = np.array( theta )
        theta_dot = self.frontend.latest().get_velocities()
        theta_dot = np.array( theta_dot )
        pressures = self.frontend.latest().get_observed_pressures()
        pressures = np.array( pressures )
        # the actual ago pressures and ant pressures
        pressure_ago = pressures[:, 0]
        pressure_ant = pressures[:, 1]
        # reference trajectory or absolute trajectory
        # reference trajectory for tracking 
        # absolute trajectory for initial posture
        if mode_trajectory == "ref":
            angle_initial = theta
            for dof in self.dof_list:
                print("{}. initial angle is: {:.2f} degree".format(dof, angle_initial[dof] * 180 / math.pi))
        elif mode_trajectory == "abs":
            angle_initial = np.zeros(len(self.dof_list))
        # if u_ago or u_ant or ff is not specified then calculate the feedforward
        if (u_ago is None) or (u_ant is None) or (ff is None):
            (u_ago, u_ant, ff) = self.Feedforward( self.y_desired, angle_initial )

        fb = np.array([])
        
        pid = np.copy( self.pid_list )
        if controller == 'pd':
            pid = np.copy( self.pid_for_tracking )

        angle_delta_pre = np.zeros( len(self.dof_list) ).reshape(len(self.dof_list), -1)
        res_i = 0

        # read the current iteration
        iteration_reference = self.frontend.latest().get_iteration()  
        # set the beginning iteration number
        iteration_begin = iteration_reference + 1500

        iteration = iteration_begin

        self.frontend.add_command(pressure_ago, pressure_ant,
                                  o80.Iteration(iteration_begin-iterations_per_command),
                                  o80.Mode.QUEUE)
        
        self.frontend.pulse()

        # how mang steps to track
        for i in range( y_list.shape[1] ):
            # all following vectors must be column vectors
            angle_delta = (y_list[:, i] + angle_initial - theta).reshape(len(self.dof_list), -1)
            res_d = ( angle_delta - angle_delta_pre ) / t
            res_i += angle_delta * t

            feedback = pid[:, 0].reshape(len(self.dof_list), -1) * angle_delta\
                     + pid[:, 1].reshape(len(self.dof_list), -1) * res_i\
                     + pid[:, 2].reshape(len(self.dof_list), -1) * res_d
            
            angle_delta_pre = np.copy( angle_delta )

            if i == 0:
                fb = np.copy( feedback )
            else:
                fb = np.hstack((fb, feedback))

            pressure_ago = np.array([], dtype=int)
            pressure_ant = np.array([], dtype=int)

            for dof in self.dof_list:
                if mode_name_list[dof] == "ff":
                    diff = ff[dof, i]
                elif mode_name_list[dof] == "fb":
                    diff = fb[dof, i]
                elif mode_name_list[dof] == "fb+ff" or mode_name_list[dof] == "ff+fb":
                    diff = ff[dof, i] + fb[dof, i]

                if self.strategy_list[dof] == 1:
                    if diff > 0:
                        pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] + diff ))
                        pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] ))
                    elif diff < 0:
                        pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] ))
                        pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] - diff ))
                    else:
                        pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] ))
                        pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] ))
                elif self.strategy_list[dof] == 2:
                    pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] + diff ))
                    pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] - diff ))
            
            # do not control the last dof
            pressure_ago[3] = self.anchor_ago_list[3]
            pressure_ant[3] = self.anchor_ant_list[3]

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
    
            self.frontend.pulse()

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)

            observation = self.frontend.pulse_and_wait()
            # update the angles
            theta = observation.get_positions()
            theta = np.array( theta )
            # update the angular velocities
            # theta_dot = observation.get_velocities()
            # theta_dot = np.array( theta_dot )

            iteration += iterations_per_command
        
        iteration_end = iteration
        
        if ifplot == "yes":
            self.PlotFigures(y_list, angle_initial, ff, fb, t_stamp_u, iteration_begin, iteration_end, iterations_per_command)
        
        if echo == "yes":
            position = np.array([])
            iteration = iteration_begin
            pressure_ago = np.array([])
            pressure_ant = np.array([])

            while iteration < iteration_end:
                observation = self.frontend.read(iteration)
                obs_position = np.array( observation.get_positions() )
                obs_pressure = np.array(observation.get_observed_pressures())

                pressure_ago = np.append(pressure_ago, obs_pressure[:, 0])
                pressure_ant = np.append(pressure_ant, obs_pressure[:, 1])

                position = np.append(position, obs_position)
                iteration += iterations_per_command
            
            position = position.reshape(-1, len(self.dof_list)).T
            pressure_ago = pressure_ago.reshape(-1, len(self.dof_list)).T
            pressure_ant = pressure_ant.reshape(-1, len(self.dof_list)).T

            return (position, fb, pressure_ago, pressure_ant)
         
    def PressureInitialization(self, times=1, duration=1):
        for _ in range(times):
            # creating a command locally. The command is *not* sent to the robot yet.
            self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                      o80.Duration_us.seconds(duration),
                                      o80.Mode.QUEUE)
            # sending the command to the robot, and waiting for its completion.
            self.frontend.pulse_and_wait()
        # for dof in self.dof_list:
        #     print("the {}. ago/ant pressure is: {:.2f}/{:.2f}".format(dof, pressures[dof, 0], pressures[dof, 1]) )

    def AngleInitialization(self, angle, tolerance_list=[0.1,0.2,0.1,1.0], 
                            frequency_frontend=100, frequency_backend=500):
        
        pid = np.copy( self.pid_list )
        tolerance_list = np.array(tolerance_list)
        theta = self.frontend.latest().get_positions()
        res_i = 0
        angle_delta_pre = np.zeros( len(self.dof_list) ).reshape(len(self.dof_list), -1)

        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )

        iteration = self.frontend.latest().get_iteration() + 200  # set the iteration when beginning

        self.frontend.add_command(self.anchor_ago_list, self.anchor_ant_list,
                                  o80.Iteration(iteration-iterations_per_command),
                                  o80.Mode.QUEUE)
        
        self.frontend.pulse()
        # do not consider the last dof
        while not (abs((theta[0:3] - angle[0:3])*180/math.pi) < tolerance_list[0:3]).all():
            # all following vectors must be column vectors
            angle_delta = (angle - theta).reshape(len(self.dof_list), -1)
            res_d = ( angle_delta - angle_delta_pre ) / t
            res_i += angle_delta * t

            feedback = pid[:, 0].reshape(len(self.dof_list), -1) * angle_delta\
                     + pid[:, 1].reshape(len(self.dof_list), -1) * res_i\
                     + pid[:, 2].reshape(len(self.dof_list), -1) * res_d
            
            angle_delta_pre = np.copy( angle_delta )

            pressure_ago = np.array([], dtype=int)
            pressure_ant = np.array([], dtype=int)

            for dof in self.dof_list:
                diff = feedback[dof]
                pressure_ago = np.append(pressure_ago, int( self.anchor_ago_list[dof] + diff ))
                pressure_ant = np.append(pressure_ant, int( self.anchor_ant_list[dof] - diff ))
            
            # do not control the last dof
            pressure_ago[3] = self.anchor_ago_list[3]
            pressure_ant[3] = self.anchor_ant_list[3]

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
    
            self.frontend.pulse()

            self.frontend.add_command(pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)

            observation = self.frontend.pulse_and_wait()
            # update the angles
            theta = np.array(observation.get_positions())

            iteration += iterations_per_command

        for dof in self.dof_list:
            print("the {}. expected/actual angle: {:.2f}/{:.2f} degree".format(dof, angle[dof] * 180 / math.pi, theta[dof] * 180 / math.pi))
            print("the error of {}. angle: {:.2f} degree".format(dof, (theta[dof] - angle[dof]) * 180 / math.pi))
        
    def ILC(self, number_iteration, angle_initial, mode_name='none'):
        '''
        only get the feedforward control without exciting the simulation/real system
        u is the basic presssure and ff is the feedforward control
        dimensions of u_ago, u_ant and ff are all dofs * length
        '''
        GLOBAL_INITIAL = np.array([0.000000, -0.514884, -0.513349, -0.172187])

        (u_ago, u_ant, ff) =  self.Feedforward(self.y_desired, angle_initial)
        # avoid too aggressive motion in the first iteration
        # ff = 0.1 * ff
        # %% only feedforward is used when do ILC
        mode_name_list = ["ff", "ff", "ff", "ff"]
        # generate the initial state z0 and the shifted control input
        # generate the initial disturbance
        disturbance = np.zeros((len(self.dof_list), self.y_desired.shape[1]))
        P_list = []
        for dof in self.dof_list:
            # self.Optimizer_list[dof].GenerateVector( ff[dof, :] )
            '''
            Generate covariance matrices Q, P, R for each dof,
            and record P for the first iteration
            '''
            P = self.Optimizer_list[dof].GenerateCovMatrix(self.delta_d_list[dof], 
                                                           self.delta_y_list[dof],
                                                           self.delta_w_list[dof],
                                                           self.delta_ini_list[dof])
            P_list.append( P )

        # initialization
        y_history = []
        ff_history = []
        disturbance_history = []
        P_history = []
        fb_history = []
        ago_history = []
        ant_history = []
        d_lifted_history = []
        P_lifted_history = []
        repeated = []
        P_lifted = [None] * len(self.dof_list)
        d_lifted = np.zeros(disturbance.shape)
        # disturbance_history.append( np.copy( disturbance ) )
        # P_history.append( np.copy( P_list ) )
        y_history.append( np.copy( self.y_desired ) )

        # begin to learn
        for i in range(number_iteration):
            
            ff_history.append( np.copy( ff ) )
            print("Iteration: {}".format(i))

            '''
            read the output of the simulation/real system
            y is the absolute measured angle
            ''' 
            print("Begin to measure...")
            (y, fb, obs_ago, obs_ant) = self.Control(self.y_desired, mode_name_list=mode_name_list, 
                                                     ifplot="no", u_ago=u_ago, u_ant=u_ant, 
                                                     ff=ff, echo="yes",
                                                     controller='pd' )
            print("...measurement completed")

            '''
            calculate the variables for the kalman filter
            ff + u is used to reach the absolute angle
            P and disturbance will be updated using Kalman filter
            '''
            print("Begin to optimize...")
            for dof in self.dof_list:
                (ff_temp, dis_temp, P_temp, dis_lifted_temp, P_lifted_temp) \
                = self.Optimizer_list[dof].Optimization(y[dof, :], 
                                                        ff[dof, :], 
                                                        disturbance[dof, :], 
                                                        P_list[dof], 
                                                        number_iteration=i, 
                                                        weight=self.weight_list[dof],
                                                        mode_name=mode_name )
                
                d_lifted[dof, :] = np.copy( dis_lifted_temp.reshape(1, -1) )
                P_lifted[dof] = np.copy(P_lifted_temp )

                ff[dof, :] = np.copy( ff_temp.reshape(1, -1) )
                disturbance[dof, :] = np.copy( dis_temp.reshape(1, -1) )
                P_list[dof] = np.copy( P_temp )
            print("...optimization completed")

            # record all the results of each iteration
            fb_history.append( np.copy(fb) )
            ago_history.append( np.copy(obs_ago) )
            ant_history.append( np.copy(obs_ant) )
            d_lifted_history.append( np.copy( d_lifted ) )
            P_lifted_history.append( np.copy( P_lifted ) )
            disturbance_history.append( np.copy( disturbance) )
            y_history.append( np.copy( y ) )
            P_history.append( np.copy( P_list ) )

            # set the same initial angle for the next iteration
            print("Begin to initialize...")
            self.AngleInitialization(GLOBAL_INITIAL)
            self.PressureInitialization()
            print("...initialization completed")

        for _ in range(2):
            (y, _, _, _) = self.Control(self.y_desired, mode_name_list=mode_name_list, 
                                        ifplot="no", u_ago=u_ago, u_ant=u_ant, ff=ff, echo="yes", 
                                        controller='pd' )
            
            repeated.append( np.copy( y ))
            self.AngleInitialization(GLOBAL_INITIAL)
            self.PressureInitialization()

        mode_name_list = ['ff+fb', 'ff+fb', 'ff+fb', 'ff+fb']
        (y_pid, _, _, _) = self.Control(self.y_desired, mode_name_list=mode_name_list, 
                                        ifplot="no", u_ago=u_ago, u_ant=u_ant, ff=ff, echo="yes", 
                                        controller='pd' )
        self.AngleInitialization(GLOBAL_INITIAL)
        self.PressureInitialization()

        return(y_history, repeated, ff_history, disturbance_history, P_history, d_lifted_history, P_lifted_history, fb_history, ago_history, ant_history, y_pid)




