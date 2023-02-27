import math
import numpy as np
import time
import matplotlib.pyplot as plt
import o80
from scipy import integrate

class Joint:

    def __init__(self, frontend, dof, anchor_ago, 
                anchor_ant, num, den, order_num, 
                order_den, ndelay, pid, strategy):
                
        self.dof = dof
        self.frontend = frontend
        self.anchor_ago = anchor_ago
        self.anchor_ant = anchor_ant
        self.delay = ndelay
        self.order_num = order_num
        self.order_den = order_den
        self.num = num[0:order_num+1]
        self.den = den[0:order_den+1]
        # self.pressure_list = pressure_list
        # self.section_list = section_list
        self.pid = pid
        self.strategy = strategy

    def SelectSection(self, theta):
        if theta <= self.section_list[0]:
            pressure_steady = self.pressure_list[0]
            return ( pressure_steady)
        if theta >= self.section_list[-1]:
            pressure_steady = self.pressure_list[-1]
            return ( pressure_steady )

        idx = 0
        while theta > self.section_list[idx]:
            idx += 1
    
        l = self.section_list[idx] - self.section_list[idx-1]
        part1 = (self.section_list[idx] - theta) / l
        part2 = (theta - self.section_list[idx-1]) / l
        pressure_steady = self.pressure_list[idx] * part2 + self.pressure_list[idx-1] * part1
        return ( pressure_steady )


    def Feedforward(self, y, angle_initial=None):
        '''
        here y should be the relative angles
        '''
        u_ago = []
        u_ant = []
        ff = []
        for i in range(0, len( y )):

            pressure_steady_ago = self.anchor_ago
            pressure_steady_ant = self.anchor_ant

            sum_num = 0
            for Nr in range(self.order_num + 1):
                a = i + self.delay - Nr
                if a >= len(y):
                    a = len(y) - 1
                if a >= 0:
                    term = self.num[Nr] * (y[a] - y[i])
                else:
                    term = self.num[Nr] * 0.0
                sum_num += term
            
            sum_den = 0
            for Nr in range(1, self.order_den + 1):
                a = i - Nr
                if a >= 0:
                    term = self.den[Nr] * ff[a]
                else:
                    term = self.den[Nr] * 0.0
                sum_den += term

            feedforward = sum_num - sum_den 
            ff.append(feedforward)
            u_ago.append(pressure_steady_ago)
            u_ant.append(pressure_steady_ant)

        u_ago = np.array(u_ago)
        u_ant = np.array(u_ant)
        ff = np.array(ff)

        return(u_ago, u_ant, ff)

    def PlotFigures(self, y, angle_initial, ff, fb, t_stamp_u, iteration_begin, iteration_end, iterations_per_command):
        
        y = np.array(y)

        iteration = iteration_begin
        obs_pressure_ago = []
        obs_pressure_ant = []
        des_pressure_ago = []
        des_pressure_ant = []
        position = []
        velocity = []
        t_stamp = []

        while iteration < iteration_end:
            observation = self.frontend.read(iteration)

            obs_pressure = observation.get_observed_pressures()
            des_pressure = observation.get_desired_pressures()
            obs_position = observation.get_positions()
            obs_velocity = observation.get_velocities()

            obs_pressure_ant.append(obs_pressure[self.dof][0])
            obs_pressure_ago.append(obs_pressure[self.dof][1])

            des_pressure_ant.append(des_pressure[self.dof][0])
            des_pressure_ago.append(des_pressure[self.dof][1])

            position.append(obs_position[self.dof])
            velocity.append(obs_velocity[self.dof])
    
            t_stamp.append( observation.get_time_stamp() * 1e-9 )

            iteration += iterations_per_command


        initial_time = t_stamp[0]
        t_stamp = np.array(t_stamp) - initial_time

        print("begin to draw the plot")
        plt.figure(1)
        line_1, = plt.plot(t_stamp, position , label=r'Observed angle $\theta_{obs}$', linewidth=1)
        line_2, = plt.plot(t_stamp_u, y + angle_initial, label=r'Desired angle $\theta_{des}$', linewidth=0.3)
        #plt.legend(handles = [line_1, line_2], loc='upper right', shadow=True)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Angle $\theta$ in rad')
        plt.show()

        plt.figure(2)
        plt.plot(t_stamp, ff, linewidth=1)
        # plt.legend(handles = [line_1, line_2], loc='upper right', shadow=True)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Feedforward Control')
        plt.show()

        plt.figure(3)
        plt.plot(t_stamp, fb, linewidth=1)
        # plt.legend(handles = [line_1, line_2], loc='upper right', shadow=True)
        plt.xlabel(r'Time $t$ in s')
        plt.ylabel(r'Feedback Control')
        plt.show()

    def Control(self, y=[], mode_name="fb+ff", mode_trajectory="ref",
                frequency_frontend=100, frequency_backend=500,
                ifplot="yes", u_ago=[], u_ant=[], ff=[], echo="no"):

        if len(y) != 0:
            t_stamp_u = np.linspace(0, (len(y) - 1) / frequency_frontend, len(y) )
        elif len(ff) != 0:
            t_stamp_u = np.linspace(0, (len(ff) - 1) / frequency_frontend, len(ff) )

        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )  # sychronize the frontend and the backend
        
        theta = self.frontend.latest().get_positions()
        theta_dot = self.frontend.latest().get_velocities()

        pressures = self.frontend.latest().get_observed_pressures()

        pressure_ago = pressures[self.dof][0]
        pressure_ant = pressures[self.dof][1]

        if mode_trajectory == "ref":
            angle_initial = theta[self.dof]
            print("Initial angle: {:.2f} degree".format(angle_initial * 180 / math.pi))
        elif mode_trajectory == "abs":
            angle_initial = 0.0

        if (len(u_ago) == 0) or (len(ff) == 0) or (len(u_ant)==0):
            (u_ago, u_ant, ff) = self.Feedforward( y, angle_initial )

        error_x = []
        error_y = []
        fb = np.array([])

        iteration_reference = self.frontend.latest().get_iteration()  # read the current iteration
        iteration_begin = iteration_reference + 2000  # set the iteration when beginning
        iteration = iteration_begin

        self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                o80.Iteration(iteration_begin-iterations_per_command),
                                o80.Mode.QUEUE)
        
        self.frontend.pulse()

        res_i = 0
        angle_delta_pre = 0

        for i in range(0, len( t_stamp_u )):
            
            if mode_name != 'ff':
                angle_delta = y[i] + angle_initial - theta[self.dof]


                # error_y.append( angle_delta )
                # error_x.append( t_stamp_u[i] )

                # res_i = integrate.trapz(error_y, error_x)
                # res_d = - theta_dot[self.dof]
                # if i > 0:
                #     res_d = (y[i] - y[i-1])/t + res_d

                res_d = ( angle_delta - angle_delta_pre ) / t
                res_i += angle_delta * t

                feedback = self.pid[0] * angle_delta + self.pid[1] * res_i + self.pid[2] * res_d
                fb = np.append(fb, feedback)

                angle_delta_pre = angle_delta

            # for the real system we use another control strategy
            if mode_name == "ff":
                diff = int( ff[i] ) 
            elif mode_name == "fb":
                diff = int( fb[i] )
            elif mode_name == "fb+ff" or mode_name == "ff+fb":
                diff = int( ff[i] + fb[i] )
            
            if self.strategy == 1:
                if diff > 0:
                    pressure_ago = int( self.anchor_ago + diff )
                    pressure_ant = int( self.anchor_ant )
                elif diff < 0:
                    pressure_ago = int( self.anchor_ago )
                    pressure_ant = int( self.anchor_ant - diff )
                else:
                    pressure_ago = int( self.anchor_ago )
                    pressure_ant = int( self.anchor_ant )
            elif self.strategy == 2:
                pressure_ago = int( self.anchor_ago + diff )
                pressure_ant = int( self.anchor_ant - diff )
            elif self.strategy == 3:
                if diff > 0:
                    pressure_ago = int( self.anchor_ago + diff + 10000 )
                    pressure_ant = int( self.anchor_ant )
                elif diff < 0:
                    pressure_ago = int( self.anchor_ago )
                    pressure_ant = int( self.anchor_ant - diff + 10000 )
                else:
                    pressure_ago = int( self.anchor_ago )
                    pressure_ant = int( self.anchor_ant )

            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
    
            self.frontend.pulse()

            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)

            observation = self.frontend.pulse_and_wait()
            theta = observation.get_positions()
            theta_dot = observation.get_velocities()

            iteration += iterations_per_command
        
        iteration_end = iteration
        
        if ifplot == "yes":
            self.PlotFigures(y, angle_initial, ff, fb, t_stamp_u, iteration_begin, iteration_end, iterations_per_command)
        
        if echo == "yes":
            position = []
            iteration = iteration_begin
            ressure_ago = np.array([])
            pressure_ant = np.array([])
            
            while iteration < iteration_end:
                observation = self.frontend.read(iteration)

                obs_position = np.array(observation.get_positions())
                obs_pressure = np.array(observation.get_observed_pressures())

                pressure_ago = np.append(pressure_ago, obs_pressure[self.dof][0])
                pressure_ant = np.append(pressure_ant, obs_pressure[self.dof][1])

                position.append(obs_position[self.dof])

                iteration += iterations_per_command

            return(position, fb, pressure_ago, pressure_ant)
    

    def PressureInitialization(self, pressure_ago, pressure_ant, times=4, duration=2):
        for _ in range(times):
            # creating a command locally. The command is *not* sent to the robot yet.
            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                o80.Duration_us.seconds(duration),
                                o80.Mode.QUEUE)

            # sending the command to the robot, and waiting for its completion.
            self.frontend.pulse_and_wait()

        observation = self.frontend.latest()
        pressures = observation.get_observed_pressures()
        positions = observation.get_positions()

        print("the {}. angle is: {} rad".format(self.dof, positions[self.dof]))


    def AngleInitialization(self, angle, frequency_frontend=100, frequency_backend=500, tolerance=0.1, T=5):
        theta = self.frontend.latest().get_positions()

        period_backend = 1.0 / frequency_backend  # period of backend
        period_frontend = 1.0 / frequency_frontend  # period of frontend
        t = 1 / frequency_frontend
        iterations_per_command = int( period_frontend / period_backend )

        iteration = self.frontend.latest().get_iteration() + 200  # set the iteration when beginning

        self.frontend.add_command(self.dof, self.anchor_ago, self.anchor_ant,
                                o80.Iteration(iteration-iterations_per_command),
                                o80.Mode.QUEUE)
        
        self.frontend.pulse()

        res_i = 0
        angle_delta_pre = 0

        while abs((theta[self.dof] - angle)*180/math.pi) > tolerance:
            angle_delta = angle - theta[self.dof]
            res_d = ( angle_delta - angle_delta_pre ) / t
            res_i += angle_delta * t
            feedback = self.pid[0] * angle_delta + self.pid[1] * res_i + self.pid[2] * res_d
            angle_delta_pre = angle_delta

            diff = int( feedback )

            pressure_ago = int( self.anchor_ago + diff )
            pressure_ant = int( self.anchor_ant - diff )

            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration),
                                      o80.Mode.QUEUE)
    
            self.frontend.pulse()

            self.frontend.add_command(self.dof, pressure_ago, pressure_ant,
                                      o80.Iteration(iteration + iterations_per_command - 1),
                                      o80.Mode.QUEUE)

            observation = self.frontend.pulse_and_wait()
            theta = observation.get_positions()
            theta_dot = observation.get_velocities()

            iteration += iterations_per_command

        print("Expected: {:.2f} degree".format(angle * 180 / math.pi))
        print("Actual: {:.2f} degree".format(theta[self.dof] * 180 / math.pi) )
        print("Error: {:.2f} degree".format( ( theta[self.dof] - angle ) * 180 / math.pi ))



  
