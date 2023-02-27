'''
This script defined important functions used in the data analysis for online optimization runs
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time
import pickle
import os
import sys
import math
import matplotlib.patches as patches
import tikzplotlib as tpl

from RacketHit import HitRecording

# %%

Desktop = r'/home/philip/Desktop/'
path_iros_plots = r'/home/philip/Desktop/IROS2023/Figures/'

class RepeatedHits():
    '''
    For data analysis of repeated hits and online optimization results
    '''
    def __init__(self, path=Desktop + r'Repeated Hit Statistics/Data Online Learning/', index=None, n=None, leave_out=[], load_individually=False):
        

        self.A = np.array( [[ -0.99591269,  0.07944872, 0.04296286],
                            [ -0.07685274, -0.99529339, 0.05903159],
                            [  0.04745064,  0.0554885,  0.99733117]]) # from calibration 2022_09_22

        self.CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        self.index = index
        self.n = n
        self.leave_out = leave_out

        self.series = 14
        while os.path.exists(Desktop + r'Racket Impact NN/Interception_Data_Continous/Series {}/Series 1/'.format(self.series+1)):
            self.series += 1
        
        self.target = []
        self.impact_hist = []
        self.theta_set_hist = []
        self.theta_real_hist = []

        loads = []
        with open(path+'Run_{}'.format(str(index)), 'rb') as f:
            while True:
                try:
                    loads.append(pickle.load(f))
                except:
                    break

        # load, depending on which variables were saved

        if len(loads)==11 or len(loads)==12:
            with open(path+'Run_{}'.format(str(index)), 'rb') as f:
                target_Init         = pickle.load(f)
                self.target         = pickle.load(f)
                self.theta_set_hist = pickle.load(f)[-n:]
                self.impact_hist    = pickle.load(f)[-n:]
                self.error_hist     = pickle.load(f)[-n:]
                p_target_hist       = pickle.load(f)[-n:]
                v_target_hist       = pickle.load(f)[-n:]
                self.step_lengths   = pickle.load(f)[-n:]
                self.mode           = pickle.load(f)
                use_mean_traj       = pickle.load(f)
                missed              = pickle.load(f)

        if len(loads)==6:
            with open(path+'Run_{}'.format(str(index)), 'rb') as f:
                self.target         = pickle.load(f)
                self.theta_set_hist = pickle.load(f)[-n:]
                self.impact_hist    = pickle.load(f)[-n:]
                self.error_hist     = pickle.load(f)[-n:]
                use_mean_traj       = pickle.load(f)
                missed              = pickle.load(f)

        if len(loads)==4:
            with open(path+'Run_{}'.format(str(index)), 'rb') as f:
                self.impact_hist    = pickle.load(f)[-n:]
                self.error_hist     = pickle.load(f)[-n:]
                global_iterator     = pickle.load(f)
                missed              = pickle.load(f)

        if len(loads)==3:
            with open(path+'Run_{}'.format(str(index)), 'rb') as f:
                self.impact_hist    = pickle.load(f)[-n:]
                self.error_hist     = pickle.load(f)[-n:]
                global_iterator     = pickle.load(f)
                # missed              = pickle.load(f)

        self.theta_set_hist = [self.theta_set_hist[i] for i in range(len(self.theta_set_hist)) if i not in leave_out]
        self.impact_hist = [self.impact_hist[i] for i in range(len(self.impact_hist)) if i not in leave_out]
        self.error_hist = [self.error_hist[i] for i in range(len(self.error_hist)) if i not in leave_out]
        
        self.theta_real_hist = []
        not_loaded = []

        self.impacts_uncal = []

        if load_individually:

            for index_iter in range(index-n+1, index+1):
                Hit = HitRecording(index=index_iter, path=Desktop + r'Racket Impact NN/Interception_Data_Continous/Series {}/Series 1/'.format(self.series), load_prints=True)

                if Hit.load:
                    theta_real_curr = np.concatenate( (Hit.robot_angles, Hit.robot_velocities) )
                    self.theta_real_hist.append(theta_real_curr)
                    impact_cal = Hit.racket_impact_y

                    self.impacts_uncal.append(impact_cal)#np.linalg.inv(self.A)@impact_cal[0:3]-self.CALIBRATION)
                
                else:
                    print('not loaded: ', index_iter, index_iter - (index-n+1))
                    not_loaded.append(index_iter)

                sys.stdout.write('\r')
                sys.stdout.write('Hit loading... {}%'.format(round( (index_iter - (index-n)) / (n) * 100 )))
                sys.stdout.flush()

            print('not loaded: ', not_loaded)
            print()

    def ThetaEvolution(self):

        fig, axes = plt.subplots(2)
        axes[0].plot([theta[0] for theta in self.theta_set_hist])
        axes[1].plot([theta[3] for theta in self.theta_set_hist])
        axes[0].set_ylabel('theta0 [deg]')
        axes[1].set_ylabel('theta3 [deg]')
        axes[1].set_xlabel('iteration [-]')
        fig.suptitle(self.mode)

    def MeanImpactEvolution(self, yscale='linear'):
        '''
        plot the evolution of the mean impact point
        '''

        self.mean_impacts_running_x = []
        self.mean_impacts_running_y = []
        self.std_impacts_running_x = []
        self.std_impacts_running_y = []
        self.mean_error_running = []

        for j in range(self.n):
            mean_x = np.mean([i[0] for i in self.impact_hist[:(j+1)]])
            mean_y = np.mean([i[1] for i in self.impact_hist[:(j+1)]])
            std_x = np.std([i[0] for i in self.impact_hist[:(j+1)]])
            std_y = np.std([i[1] for i in self.impact_hist[:(j+1)]])

            self.mean_impacts_running_x.append(mean_x)
            self.mean_impacts_running_y.append(mean_y)
            self.std_impacts_running_x.append(std_x)
            self.std_impacts_running_y.append(std_y)
            self.mean_error_running.append(np.linalg.norm([mean_x-self.target[0], mean_y-self.target[1]]))

        # plt.figure(figsize=(8, 8))
        # plt.plot(self.mean_impacts_running_x, self.mean_impacts_running_y, marker='x')
        # plt.scatter(self.target[0], self.target[1], marker='+')

        plt.figure(figsize=(8, 8))
        plt.plot(self.mean_error_running, label='mean imapct')
        plt.ylabel('distance error [m]')
        plt.xlabel('iteration [-]')
        plt.yscale(yscale)
        plt.legend()

        plt.figure(figsize=(8, 8))
        plt.plot(self.std_impacts_running_x, label='std x imapct')
        plt.plot(self.std_impacts_running_y, label='std y imapct')
        plt.ylabel('[m]')
        plt.xlabel('iteration [-]')
        plt.legend()

    def ImpactVariance(self):
        '''
        plot target, all impacts and mean impact
        '''

        B = np.array([[-1.0, 0.0, 0.0],
            [ 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]])

        A = np.array(  [[ 0.99591269, -0.07944872, -0.04296286],
                        [ 0.07685274,  0.99529339, -0.05903159],
                        [ 0.04745064,  0.0554885,   0.99733117]]) # from calibration 2022_09_22
        A = B@A

        CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        impact_x = [i[0] for i in self.impact_hist]
        impact_y = [i[1] for i in self.impact_hist]   

        impacts = [np.array([impact_x[i], impact_y[i], -0.44]) for i in range(len(impact_x))]
        
        impacts_uncal = [np.linalg.inv(A)@(i)- CALIBRATION for i in impacts]

        impact_x = [i[0] for i in impacts_uncal]
        impact_y = [i[1] for i in impacts_uncal]

        mean_x = np.mean(impact_x)
        std_x = np.std(impact_x)
        mean_y = np.mean(impact_y)
        std_y = np.std(impact_y)

        plt.figure(figsize=(8,8))
        plt.scatter(impact_x, impact_y, color=self.colors[0], alpha = 0.5, label='std x = {}, std y = {}'.format(round(std_x,2), round(std_y, 2)))

        plt.scatter([mean_x], [mean_y], marker='x', lw=2, s=100, label='mean hit point')

        plt.scatter([self.target[0]], [self.target[1]], color=self.colors[0], marker='+', lw=2, s=100, label='target')

        plt.scatter([mean_x], [mean_y], marker='x', color=self.colors[0], lw=2, s=100)

        ideal_table = np.array([0.141, 1.74, -0.441]) # from calibration 2022_09_22
        height_of_table = 0.765  # m
        height_of_ground = ideal_table[2] - height_of_table
        center_of_table = ideal_table[0:2]
        half_width_of_table = 1.525/2
        half_length_of_table = 2.74/2
        rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label='table')
        
        plt.gca().add_patch(rect)
        plt.legend(loc='lower left', fontsize=16)
        # plt.title('Different Initial Guesses')
        plt.xlabel('$x$ [m]', fontsize=18)
        plt.ylabel('$y$ [m]', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.ylim([-3.15, -1.55])
        # plt.xlim([-0.9, 0.8])
        # plt.ylim([1.5, 3.5])
        # plt.xlim([-1, 1])
        plt.ylim([0, 4])
        plt.xlim([-2, 2])

    def ThetaRealvsError(self):
        '''
        plot correlations between real theta and error
        '''

        fig, axs = plt.subplots(4, figsize=(8, 8))
        axs[0].plot(self.error_hist)
        axs[0].set_ylabel('[m^2]')

        axs[1].plot([i[0] for i in self.theta_real_hist])
        axs[1].plot([i[0] for i in self.theta_set_hist])
        axs[1].set_ylabel('theta 0 [rad]')

        axs[2].plot([i[3] for i in self.theta_real_hist])
        axs[2].plot([i[3] for i in self.theta_set_hist])
        axs[2].set_ylabel('theta 3 [rad]')

        axs[3].plot([i[4] for i in self.theta_real_hist])
        axs[3].plot([i[4] for i in self.theta_set_hist])
        axs[3].set_ylabel('theta 0 dot [rad/s]')

    def ErrorHist(self):
        '''
        plot evolution of distance error
        '''

        error_list = [self.error_hist[i]**0.5 for i in range(len(self.error_hist)) if i not in self.leave_out]

        plt.figure(figsize=(8,8))
        plt.plot(error_list, lw=1.5, color='black')
        plt.scatter(range(len(error_list)), error_list, marker='x', lw=2, s=60, color='black')
        plt.xlabel('iteration $i$ [-]', fontsize=16)
        plt.ylabel('distance error $\epsilon$ [m]', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.figure(figsize=(8, 8))
        plt.hist(error_list, bins=25)




class ListRepeatedHits():
    '''
    For data analysis of sets of repeated hits and online optimization results
    '''

    def __init__(self, list_runs=None, leave_out=[], n=None):
        self.list_runs = list_runs
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.leave_out = leave_out
        print('number of hits = ', [len(Run.error_hist) for Run in list_runs])

        # self.n = len(self.list_runs[0].error_hist)
        self.n = n


        B = np.array([[-1.0, 0.0, 0.0],
            [ 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]])

        A = np.array(  [[ 0.99591269, -0.07944872, -0.04296286],
                        [ 0.07685274,  0.99529339, -0.05903159],
                        [ 0.04745064,  0.0554885,   0.99733117]]) # from calibration 2022_09_22
        self.A = B@A

        self.CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        return
    
    def ErrorHist(self):
        '''
        plot evolution of distance error with error band
        '''

        mean_list = []
        std_list = []

        for i in range(self.n):
            error_list_curr = [(Run.error_hist[i])**0.5 for Run in self.list_runs]
            # error_list_curr = [(Run.error_hist[i]) for Run in self.list_runs]
            mean_list.append(np.mean(error_list_curr))
            std_list.append(np.std(error_list_curr))

        plt.figure(figsize=(8,8))
        plt.plot(mean_list, color=self.colors[0], lw=1.5)
        plt.fill_between(range(self.n), [mean_list[i] - std_list[i] for i in range(self.n)], [mean_list[i] + std_list[i] for i in range(self.n)], alpha=0.2)
        plt.xlabel('iteration [-]')
        # plt.xticks([0, 1, 2, 3, 4])
        plt.ylabel('distance error [m]')
        
        # plt.savefig(Desktop + r'Results/Results 2022_10_10/different_initial_guesses_repeatedly_error_hist')

        plt.figure(figsize=(8, 8))
        for j in range(len(self.list_runs)):
            plt.plot([(self.list_runs[j].error_hist[i])**0.5 for i in range(self.n)])

    def ThetaEvolution(self):

        fig, axes = plt.subplots(2)
        for Run in self.list_runs:

            axes[0].plot([theta[0]*180/math.pi for theta in Run.theta_set_hist], label=Run.mode)
            axes[1].plot([theta[3]*180/math.pi for theta in Run.theta_set_hist], label=Run.mode)

        axes[0].legend()
        axes[0].set_ylabel('theta0 [rad]')
        axes[1].set_ylabel('theta3 [rad]')
        axes[1].set_xlabel('iteration [-]')
        fig.suptitle('Policy Evolution')

    def ErrorHist_NNvsPhysical(self):
        '''
        plot evolution of distance error with error band
        '''

        print('mode: ', [Run.mode for Run in self.list_runs])

        mean_list_NN = []
        std_list_NN = []
        for i in range(self.n):
            error_list_curr = [(Run.error_hist[i])**0.5 for Run in self.list_runs if Run.mode=='NN']
            mean_list_NN.append(np.mean(error_list_curr))
            std_list_NN.append(np.std(error_list_curr))

        mean_list_Physical = []
        std_list_Physical = []
        for i in range(self.n):
            error_list_curr = [(Run.error_hist[i])**0.5 for Run in self.list_runs if Run.mode=='Physical']
            mean_list_Physical.append(np.mean(error_list_curr))
            std_list_Physical.append(np.std(error_list_curr))

        plt.figure(figsize=(8,8))
        plt.plot(mean_list_NN, color=self.colors[0], lw=1.5)
        plt.fill_between(range(self.n), [mean_list_NN[i] - std_list_NN[i] for i in range(self.n)], [mean_list_NN[i] + std_list_NN[i] for i in range(self.n)], color=self.colors[0], alpha=0.2)
        plt.plot(mean_list_Physical, color=self.colors[1], lw=1.5)
        plt.fill_between(range(self.n), [mean_list_Physical[i] - std_list_Physical[i] for i in range(self.n)], [mean_list_Physical[i] + std_list_Physical[i] for i in range(self.n)], color=self.colors[1], alpha=0.2)
        plt.xlabel('iteration [-]')
        plt.ylabel('distance error [m]')
        
        plt.figure(figsize=(8, 8))
        for j in range(len(self.list_runs)):
            plt.plot([(self.list_runs[j].error_hist[i])**0.5 for i in range(self.n)])

    def ImpactHist(self):
        '''
        plot impacts 
        '''

        mean_list = []
        std_list = []

        for i in range(self.n):
            error_list_curr = [(Run.error_hist[i])**0.5 for Run in self.list_runs]
            mean_list.append(np.mean(error_list_curr))
            std_list.append(np.std(error_list_curr))

        plt.figure(figsize=(8,8))
        plt.xlabel('impact x [m]')
        plt.ylabel('impact y [m]')

        for h, Run in enumerate(self.list_runs):


            impact_x = [i[0] for i in Run.impact_hist]
            impact_y = [i[1] for i in Run.impact_hist]    

            mean_x = np.mean(impact_x)
            std_x = np.std(impact_x)
            mean_y = np.mean(impact_y)
            std_y = np.std(impact_y)

            plt.scatter(impact_x, impact_y, alpha = 0.5, label='std x = {}, std y = {}'.format(round(std_x,2), round(std_y, 2)), color=self.colors[h])
            for i in range(len(impact_x)):
                if i in self.leave_out:
                    plt.scatter(impact_x[i], impact_y[i], alpha=1, color='red')
            plt.scatter([mean_x], [mean_y], marker='x', lw=2, s=100, label='mean hit point')
            plt.scatter([Run.target[0]], [Run.target[1]], color=self.colors[h], marker='+', lw=2, s=100, label='target')

    def ImpactHist_WithTable(self):
        '''
        plot impacts in drawing with table
        '''

        B = np.array([[-1.0, 0.0, 0.0],
            [ 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]])

        A = np.array(  [[ 0.99591269, -0.07944872, -0.04296286],
                        [ 0.07685274,  0.99529339, -0.05903159],
                        [ 0.04745064,  0.0554885,   0.99733117]]) # from calibration 2022_09_22
        A = B@A

        CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        mean_list = []
        std_list = []

        print(len(self.list_runs))
        print(self.n)

        for i in range(self.n):
            error_list_curr = [(Run.error_hist[i])**0.5 for Run in self.list_runs]
            mean_list.append(np.mean(error_list_curr))
            std_list.append(np.std(error_list_curr))

        ideal_table = np.array([0.141, 1.74, -0.441]) # from calibration 2022_09_22
        height_of_table = 0.765  # m
        height_of_ground = ideal_table[2] - height_of_table # m, add a value distance to make sure the table impact is triggered in the state estimation
        center_of_table = ideal_table[0:2]
        half_width_of_table = 1.525/2
        half_length_of_table = 2.74/2
        rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label='table')
        
        plt.figure(figsize=(8, 8))
        plt.legend(loc='lower left', fontsize=16)
        # plt.title('Different Initial Guesses')
        plt.xlabel('$x$ [m]', fontsize=18)
        plt.ylabel('$y$ [m]', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.ylim([-3.15, -1.55])
        # plt.xlim([-0.9, 0.8])
        # plt.ylim([1.5, 3.5])
        # plt.xlim([-1, 1])
        plt.ylim([0, 4])
        plt.xlim([-2, 2])

        for h, Run in enumerate(self.list_runs):

            # impact_x = [i[0] for i in Run.impact_hist]
            # impact_y = [i[1] for i in Run.impact_hist]    
            # impact_z = [i[2] for i in Run.impact_hist]   

            # print('first impact cal: ', Run.impact_hist[0]) 

            # impacts = [np.array( [impact_x[i], impact_y[i], impact_z[i]] ) for i in range(len(impact_x))]
            # impacts_uncal = [np.linalg.inv(A)@(i)- CALIBRATION for i in impacts]

            # impact_x = [i[0] for i in impacts_uncal]
            # impact_y = [i[1] for i in impacts_uncal]

            # print('first impact uncal: ', impacts_uncal[0]) 

            print(Run.impacts_uncal)

            impacts = [A@(Run.impacts_uncal[i][0:3]+CALIBRATION) for i in range(len(Run.impacts_uncal)) if i not in self.leave_out]

            impact_x = [i[0] for i in impacts]
            impact_y = [i[1] for i in impacts]

            print('leave out: ', self.leave_out)

            print(len(impact_x), impact_x)
            print(len(impact_y), impact_y)

            target_uncal = np.array([0.164, 2.79, -0.44]) # mark 2

            mean_x = np.mean(impact_x)
            std_x = np.std(impact_x)
            mean_y = np.mean(impact_y)
            std_y = np.std(impact_y)

            plt.scatter(impact_x, impact_y, alpha = 0.5, color='black')
            for i in range(len(impact_x)):
                if i in self.leave_out:
                    plt.scatter(impact_x[i], impact_y[i], alpha=1, color='red')
            # ax.scatter([mean_x], [mean_y], marker='x', lw=2, s=100, label='mean hit point', color='black')

            for impact_iter in range(len(impact_x)):
                if impact_iter == 0:
                    plt.scatter(impact_x[0], impact_y[0], color='black', marker='+', s=100, label='initial guess', zorder=2)
                    plt.scatter(impact_x[1:-1], impact_y[1:-1], color='black', label='landing point history', s=10)
                else:
                    plt.scatter(impact_x[0], impact_y[0], color='black', marker='+', s=100)

                plt.plot(impact_x, impact_y, color='grey', zorder=1, alpha=0.2)
                plt.scatter(impact_x[1:], impact_y[1:], color='black', s=10)

        plt.scatter(target_uncal[0], target_uncal[1], marker='x', lw=2, s=100, label='target', color='black', linewidth=3, zorder=2)
        plt.gca().add_patch(rect)
        plt.legend(loc='lower left', fontsize=16)

    def InitialGuessConvergence(self, n_hits=5):
        '''
        plot first iterations from different initial guesses
        '''
        
        plt.figure(figsize=(8, 8))

        for h, Run in enumerate(self.list_runs):

            impact_x = [impact[0] for i, impact in enumerate(Run.impact_hist) if i<n_hits]
            impact_y = [impact[1] for i, impact in enumerate(Run.impact_hist) if i<n_hits]    

            plt.plot(impact_x, impact_y, alpha = 0.8, color=self.colors[h])
        plt.scatter([Run.target[0]], [Run.target[1]], color='black', marker='+', lw=2, s=100, label='target')
        plt.xlabel('impact x [m]')
        plt.ylabel('impact y [m]')

    def InitialGuessConvergence_Averaged(self, n_hits=5):
        '''
        plot first iterations from different initial guesses onto the same target
        '''

        # # find all used initial guesses
        # ig_list = []
        # for Run in self.list_runs:
        #     included = False
        #     for ig in ig_list:
        #         if np.array_equal(ig, Run.theta_set_hist[0]):
        #             included = True; break
        #     if not included:    
        #         ig_list.append(Run.theta_set_hist[0].copy())

        ig_list = [
            np.array([ -0.05, 0.709, 1.56, -0.05, 6, 0., 0.]),
            np.array([ 0.1, 0.709, 1.56, 0.2, 6, 0., 0.]),
            np.array([ 0.4, 0.709, 1.56, 0.2, 6, 0., 0.]),
            np.array([ 0.7, 0.709, 1.56, 0.0, 6, 0., 0.]),
            np.array([ 0.6, 0.709, 1.56, -0.2, 6, 0., 0.]),
            np.array([ 0.1, 0.709, 1.56, -0.25, 6, 0., 0.])
        ]

        def uncal(i):
            x = i[0]
            y = i[1]
            z = -0.44
            pos_cal = np.array([x, y, z])
            return np.linalg.inv(self.A)@pos_cal - self.CALIBRATION
        
        plt.figure(figsize=(8, 8))

        # for each initial guess: calculate mean impact points per iteration
        mean_impact_per_iteration_per_ig = []
        for ig_iter, ig in enumerate(ig_list):
            impact_hists = []
            for Run in self.list_runs:
                if np.array_equal(ig, Run.theta_set_hist[0]):
                    impact_hists.append(Run.impact_hist)
            print(len(impact_hists), [len(i) for i in impact_hists])
            mean_impact_x_per_iteration = []
            mean_impact_y_per_iteration = []
            for j in range(5):
                mean_impact_x_per_iteration.append(np.mean( [ uncal(i[j])[0] for i in impact_hists ] ))
                mean_impact_y_per_iteration.append(np.mean( [ uncal(i[j])[1] for i in impact_hists ] ))
            if ig_iter == 0:
                plt.scatter(mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0], color='black', marker='+', s=100, label=r'initial guess')
                
                plt.scatter(mean_impact_x_per_iteration[1:-1], mean_impact_y_per_iteration[1:-1], color='black', label=r'landing point history', s=10)
               
            else:
                plt.scatter(mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0], color='black', marker='+', s=100)
                

            plt.plot(mean_impact_x_per_iteration, mean_impact_y_per_iteration, color='grey') 
            plt.scatter(mean_impact_x_per_iteration[1:], mean_impact_y_per_iteration[1:], color='black', s=10)


            for i in range(5):
                print('{} {}'.format(mean_impact_y_per_iteration[i], mean_impact_x_per_iteration[i]))
            
        target = self.list_runs[0].target

        print('target: ', uncal(target))

        plt.scatter(uncal(target)[0], uncal(target)[1], marker='x', s=100, label=r'target', color='black', linewidths=3)
        
        ideal_table = np.array([0.141, 1.74, -0.441]) # from calibration 2022_09_22
        height_of_table = 0.765  # m
        height_of_ground = ideal_table[2] - height_of_table # m, add a value distance to make sure the table impact is triggered in the state estimation
        center_of_table = ideal_table[0:2]
        half_width_of_table = 1.525/2
        half_length_of_table = 2.74/2
        rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label=r'table')
        plt.gca().add_patch(rect)
        plt.legend(loc='lower left', fontsize=12)
        # plt.title('Different Initial Guesses')
        plt.xlabel(r'$x$ [m]', fontsize=14)
        plt.ylabel(r'$y$ [m]', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.ylim([-3.15, -1.55])
        # plt.xlim([-0.9, 0.8])
        plt.ylim([0, 4])
        plt.xlim([-2, 2])

        # plt.show()
        tpl.save(path_iros_plots + 'Results/Different_initial_guesses_tex.tex', axis_width=r'0.45\textwidth', axis_height = r'0.25\textwidth')
        # plt.savefig(Desktop + r'Results/Results 2022_10_10/different initial guesses')

        # plt.figure(figsize=(8, 8))

        # # for each initial guess: calculate mean impact points per iteration
        # mean_impact_per_iteration_per_ig = []
        # for ig_iter, ig in enumerate(ig_list):
        #     impact_hists = []
        #     for Run in self.list_runs:
        #         if np.array_equal(ig, Run.theta_set_hist[0]):
        #             impact_hists.append(Run.impact_hist)
        #     print(len(impact_hists), [len(i) for i in impact_hists])
        #     mean_impact_x_per_iteration = []
        #     mean_impact_y_per_iteration = []
        #     for j in range(5):
        #         mean_impact_x_per_iteration.append(np.mean( [ uncal(i[j])[0] for i in impact_hists ] ))
        #         mean_impact_y_per_iteration.append(np.mean( [ uncal(i[j])[1] for i in impact_hists ] ))
        #     if ig_iter == 0:
        #         plt.scatter(mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0], color='black', marker='+', s=100, label=r'initial guess')
                
        #         plt.scatter(mean_impact_x_per_iteration[1:-1], mean_impact_y_per_iteration[1:-1], color='black', label=r'landing point history', s=10)
               
        #     else:
        #         plt.scatter(mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0], color='black', marker='+', s=100)
                

        #     plt.plot(mean_impact_x_per_iteration, mean_impact_y_per_iteration, color='grey') 
        #     plt.scatter(mean_impact_x_per_iteration[1:], mean_impact_y_per_iteration[1:], color='black', s=10)
            
        # target = self.list_runs[0].target

        # plt.scatter(uncal(target)[0], uncal(target)[1], marker='x', s=100, label='target', color='black', linewidths=3)
        
        # ideal_table = np.array([0.141, 1.74, -0.441]) # from calibration 2022_09_22
        # height_of_table = 0.765  # m
        # height_of_ground = ideal_table[2] - height_of_table # m, add a value distance to make sure the table impact is triggered in the state estimation
        # center_of_table = ideal_table[0:2]
        # half_width_of_table = 1.525/2
        # half_length_of_table = 2.74/2
        # rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label='table')
        # plt.gca().add_patch(rect)
        # plt.legend(loc='lower left', fontsize=12)
        # # plt.title('Different Initial Guesses')
        # plt.xlabel('$x$ [m]', fontsize=14)
        # plt.ylabel('$y$ [m]', fontsize=14)
        # plt.xticks(fontsize=12)
        # plt.yticks(fontsize=12)
        # # plt.ylim([-3.15, -1.55])
        # # plt.xlim([-0.9, 0.8])
        # plt.ylim([0, 4])
        # plt.xlim([-2, 2])

        # # plt.show()
        # tpl.save(path_iros_plots + 'Results/Different_initial_guesses_tex.tex', axis_width=r'0.45\textwidth', axis_height = r'0.25\textwidth')
        # # plt.savefig(Desktop + r'Results/Results 2022_10_10/different initial guesses')

    def TargetConvergence_Averaged(self, n_hits=5):
        '''
        plot first iterations from same initial guess onto different target
        '''
                
        B = np.array([[-1.0, 0.0, 0.0],
                    [ 0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0]])

        A = np.array(  [[ 0.99591269, -0.07944872, -0.04296286],
                        [ 0.07685274,  0.99529339, -0.05903159],
                        [ 0.04745064,  0.0554885,   0.99733117]]) # from calibration 2022_09_22
        A = B@A

        CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        def uncal(i):
            x = i[0]
            y = i[1]
            z = -0.44
            pos_cal = np.array([x, y, z])
            return np.linalg.inv(self.A)@pos_cal - self.CALIBRATION

        target_pos_1 = np.array([0.747, 2.78, -0.438]) # mark 1
        target_pos_2 = np.array([0.164, 2.79, -0.44]) # mark 2
        target_pos_3 = np.array([-0.413, 2.78, -0.432]) # mark 3
        target_pos_4 = np.array([0.747, 2.78-1.0, -0.438]) # mark 1 - 1.0m in y
        target_pos_5 = np.array([0.164, 2.79-1.0, -0.44]) # mark 2 - 1.0m in y
        target_pos_6 = np.array([-0.413, 2.78-1.0, -0.432]) # mark 3 - 1.0m in z

        targets_pos = [target_pos_1, target_pos_2, target_pos_3, target_pos_4, target_pos_5, target_pos_6]
        target_vel = np.zeros(3)

        targets = [np.hstack(((A@(i+CALIBRATION))[0:2], target_vel)) for i in targets_pos]


        
        plt.figure(figsize=(8, 8))

        enforce_same_initial_landing_point = False

        if enforce_same_initial_landing_point:
            # for each initial guess: calculate mean impact points per iteration
            mean_initial_impact = [np.mean([uncal(Run.impact_hist[0])[0] for Run in self.list_runs]), np.mean([uncal(Run.impact_hist[0])[1] for Run in self.list_runs])]
            mean_impact_per_iteration_per_target = []
            plt.scatter(mean_initial_impact[0], mean_initial_impact[1], marker='+', s=100, color='black', label='initial guess', zorder=2)
            for target_iter, target in enumerate(targets):
                impact_hists = []
                for Run in self.list_runs:
                    if np.array_equal(target, Run.target):
                        impact_hists.append(Run.impact_hist)
                        print('impact hist shape: ', np.array(impact_hists[-1]).shape)
                print(len(impact_hists), [len(i) for i in impact_hists])
                mean_impact_x_per_iteration = [mean_initial_impact[0]]
                mean_impact_y_per_iteration = [mean_initial_impact[1]]
                for j in range(1, 5):
                    mean_impact_x_per_iteration.append(np.mean( [ uncal(i[j])[0] for i in impact_hists ] ))
                    mean_impact_y_per_iteration.append(np.mean( [ uncal(i[j])[1] for i in impact_hists ] ))

                if target_iter == 0: # just for legend
                    plt.scatter(mean_impact_x_per_iteration[1:-1], mean_impact_y_per_iteration[1:-1], color='grey', label='landing point history', s=10)                
                    plt.scatter(uncal(target)[0], uncal(target)[1], color='black', marker='x', s=100, linewidth=3, label='target')
                else: # without label for all after first
                    plt.scatter(uncal(target)[0], uncal(target)[1], color='black', marker='x', s=100, linewidth=3)

                print('target: ', uncal(target))

                plt.plot(mean_impact_x_per_iteration, mean_impact_y_per_iteration, color='grey', zorder=-1)
                # plt.scatter(mean_impact_x_per_iteration[1:], mean_impact_y_per_iteration[1:], color='black', s=10)
                plt.scatter(mean_impact_x_per_iteration, mean_impact_y_per_iteration, color='black', s=10)
            circ = patches.Circle((mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0]), radius=0.15, linewidth=1, linestyle='--', edgecolor='black', facecolor='white', alpha=1, zorder=1)
            plt.gca().add_patch(circ)

        else:
            # for each initial guess: calculate mean impact points per iteration
            for target_iter, target in enumerate(targets):
                impact_hists = []
                for Run in self.list_runs:
                    if np.array_equal(target, Run.target):
                        impact_hists.append(Run.impact_hist)
                        print('impact hist shape: ', np.array(impact_hists[-1]).shape)
                print(len(impact_hists), [len(i) for i in impact_hists])
                mean_impact_x_per_iteration = []
                mean_impact_y_per_iteration = []
                for j in range(0, 5):
                    mean_impact_x_per_iteration.append(np.mean( [ uncal(i[j])[0] for i in impact_hists ] ))
                    mean_impact_y_per_iteration.append(np.mean( [ uncal(i[j])[1] for i in impact_hists ] ))

                if target_iter == 0: # just for legend
                    plt.scatter(mean_impact_x_per_iteration[1:-1], mean_impact_y_per_iteration[1:-1], color='grey', label='landing point history', s=10)                
                    plt.scatter(uncal(target)[0], uncal(target)[1], color='black', marker='x', s=100, linewidth=3, label='target')
                    plt.scatter(mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0], color='black', marker='+', s=10, label=r'initial guess')
                else: # without label for all after first
                    plt.scatter(uncal(target)[0], uncal(target)[1], color='black', marker='x', s=100, linewidth=3)
                    plt.scatter(mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0], color='black', marker='+', s=10)

                print('target: ', uncal(target))
                print('initial guess: ', mean_impact_x_per_iteration[0], mean_impact_y_per_iteration[0])


                for i in range(5):
                    print('{} {}'.format(mean_impact_y_per_iteration[i], mean_impact_x_per_iteration[i]))

                plt.plot(mean_impact_x_per_iteration, mean_impact_y_per_iteration, color='grey', zorder=-1)
                plt.scatter(mean_impact_x_per_iteration[1:], mean_impact_y_per_iteration[1:], color='black', s=10)


        ideal_table = np.array([0.141, 1.74, -0.441]) # from calibration 2022_09_22
        height_of_table = 0.765  # m
        height_of_ground = ideal_table[2] - height_of_table # m, add a value distance to make sure the table impact is triggered in the state estimation
        center_of_table = ideal_table[0:2]
        half_width_of_table = 1.525/2
        half_length_of_table = 2.74/2
        rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label='table')
        plt.gca().add_patch(rect)        
        plt.legend(loc='lower left', fontsize=12)
        # plt.title('Different Targets')
        plt.xlabel('$x$ [m]', fontsize=14)
        plt.ylabel('$y$ [m]', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.ylim([-3.15, -1.55])
        # plt.xlim([-0.9, 0.8])
        plt.ylim([0, 4])
        plt.xlim([-2, 2])

        # tpl.save(path_iros_plots + 'Results/Different_targets')
        # plt.savefig(Desktop + r'Results/Results 2022_11_23/different targets')

    def InitialGuessConvergence(self, n_hits=5):
        '''
        plot first iterations from different initial guesses
        '''

        plt.figure(figsize=(8, 8))

        for h, Hit in enumerate(self.list_runs):

            impact_x = [impact[0] for i, impact in enumerate(Hit.impact_hist) if i<n_hits]
            impact_y = [impact[1] for i, impact in enumerate(Hit.impact_hist) if i<n_hits]    

            plt.plot(impact_x, impact_y, alpha = 0.8, color=self.colors[h])
        plt.scatter([Hit.target[0]], [Hit.target[1]], color='black', marker='+', lw=2, s=100, label='target')
        plt.xlabel('impact x [m]')
        plt.ylabel('impact y [m]')

    def MeanImpactEvolution_NNvsPhysical(self, yscale='linear'):
        '''
        plot the evolution of the mean impact point
        '''

        def uncal(i):
            x = i[0]
            y = i[1]
            z = -0.44
            pos_cal = np.array([x, y, z])
            return np.linalg.inv(self.A)@pos_cal - self.CALIBRATION

        self.mean_impacts_running_x_Physical = []
        self.mean_impacts_running_y_Physical = []
        self.mean_impacts_running_Physical = []
        self.std_impacts_running_x_Physical = []
        self.std_impacts_running_y_Physical = []
        self.mean_error_running_Physical = []
        self.std_error_running_Physical = []

        self.mean_impacts_running_x_NN = []
        self.mean_impacts_running_y_NN = []
        self.mean_impacts_running_NN = []
        self.std_impacts_running_x_NN = []
        self.std_impacts_running_y_NN = []
        self.mean_error_running_NN = []
        self.std_error_running_NN = []


        for Run in self.list_runs:
            for j in range(self.n):
                
                mean_x = np.mean([i[0] for i in Run.impact_hist[:(j+1)]])
                mean_y = np.mean([i[1] for i in Run.impact_hist[:(j+1)]])
                std_x = np.std([i[0] for i in Run.impact_hist[:(j+1)]])
                std_y = np.std([i[1] for i in Run.impact_hist[:(j+1)]])
                std_dist_error = np.std([np.linalg.norm(i) for i in Run.impact_hist[:(j+1)]])

                if Run.mode=='NN':
                    
                    self.mean_impacts_running_NN.append(np.array([mean_x, mean_y]))
                    self.mean_impacts_running_x_NN.append(mean_x)
                    self.mean_impacts_running_y_NN.append(mean_y)
                    self.std_impacts_running_x_NN.append(std_x)
                    self.std_impacts_running_y_NN.append(std_y)
                    self.mean_error_running_NN.append(np.linalg.norm([mean_x-Run.target[0], mean_y-Run.target[1]]))
                    self.std_error_running_NN.append(std_dist_error)

                if Run.mode=='Physical':

                    self.mean_impacts_running_Physical.append(np.array([mean_x, mean_y]))
                    self.mean_impacts_running_x_Physical.append(mean_x)
                    self.mean_impacts_running_y_Physical.append(mean_y)
                    self.std_impacts_running_x_Physical.append(std_x)
                    self.std_impacts_running_y_Physical.append(std_y)
                    self.mean_error_running_Physical.append(np.linalg.norm([mean_x-Run.target[0], mean_y-Run.target[1]]))
                    self.std_error_running_Physical.append(std_dist_error)

        mean_imp_hist = True
        if mean_imp_hist:
            plt.figure(figsize=(8, 8))
            plt.plot([uncal(i)[0] for i in self.mean_impacts_running_NN], [uncal(i)[1] for i in self.mean_impacts_running_NN], marker='x', alpha=0.5, label='NN')
            plt.plot([uncal(i)[0] for i in self.mean_impacts_running_Physical], [uncal(i)[1] for i in self.mean_impacts_running_Physical], marker='x', alpha=0.5, label='Physical')
            # plt.scatter(self.list_runs[0].target[0], self.list_runs[0].target[1], marker='+')
            plt.scatter(uncal(self.list_runs[0].target)[0], uncal(self.list_runs[0].target)[1], marker='x', s=100, label='target', color='black', linewidths=3)
            print('target: ', uncal(self.list_runs[0].target)[0], uncal(self.list_runs[0].target)[1])
            plt.xlabel('$x_\mathrm{landing}$ [m]', fontsize=14)
            plt.ylabel('$y_\mathrm{landing}$ [m]', fontsize=14)
            plt.ylim([3.3, 2.3])
            plt.xlim([-0.8, 0.2])
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            tpl.save(path_iros_plots + 'Results/Mean_landing_point_history')
            # plt.title('long run - mean impact')
            # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_mean_impact')


        err_mean_imp = True
        if err_mean_imp:
            plt.figure(figsize=(8, 8))
            plt.plot(self.mean_error_running_NN, label='NN')
            plt.plot(self.mean_error_running_Physical, label='Physical')
            plt.ylabel('distance error $\epsilon_i$ [m]', fontsize=14)
            plt.xlabel('iteration $i$ [-]', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.yscale(yscale)
            plt.legend(fontsize=12)
            tpl.save(path_iros_plots + 'Results/Mean_distance_error')
            # plt.title('long run - error of mean impact')
            # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_mean_impact_error_flat')

        var_per_coord = False
        if var_per_coord:
            plt.figure(figsize=(8, 8))
            plt.plot(self.std_impacts_running_x_NN, label='NN - x imapct')
            plt.plot(self.std_impacts_running_y_NN, label='NN - y imapct')
            plt.plot(self.std_impacts_running_x_Physical, label='Physical - x imapct')
            plt.plot(self.std_impacts_running_y_Physical, label='Physical - y imapct')
            plt.ylabel('standard deviation of distance error[m]')
            plt.xlabel('iteration $i$ [-]')
            plt.legend()
            plt.title('long run - variance evolution')
            # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_variance_evolution_1')

        var = True
        if var:
            plt.figure(figsize=(8, 8))
            plt.plot(self.std_error_running_NN, label='NN')
            plt.plot(self.std_error_running_Physical, label='Physical')
            plt.ylabel('standard deviation $\sigma_i$ [m]', fontsize=14)
            plt.xlabel('iteration $i$ [-]', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            tpl.save(path_iros_plots + 'Results/Spread')
            # plt.title('long run - variance evolution')
            # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_variance_evolution_2')

    def MeanImpactEvolution(self, yscale='linear', mode='Physical'):
        '''
        plot the evolution of the mean impact point
        '''

        self.mean_impacts_running_x = []
        self.mean_impacts_running_y = []
        self.std_impacts_running_x = []
        self.std_impacts_running_y = []
        self.mean_error_running = []
        self.std_error_running = []

        fig = plt.figure(figsize=(8, 8))
        for Run in self.list_runs:
            std_err_running = []
            cov_sqrt = []
            for j in range(self.n):


                landing_points = np.array([i[0:2] for i in Run.impact_hist[:(j+1)]])
                cov_sqrt.append(np.linalg.norm(np.cov(landing_points.T))**0.5)
                    
                mean_x = np.mean([i[0] for i in Run.impact_hist[:(j+1)]])
                mean_y = np.mean([i[1] for i in Run.impact_hist[:(j+1)]])
                std_x = np.std([i[0] for i in Run.impact_hist[:(j+1)]])
                std_y = np.std([i[1] for i in Run.impact_hist[:(j+1)]])
                std_dist_error = np.std([np.linalg.norm(i) for i in Run.impact_hist[:(j+1)]])

                self.mean_impacts_running_x.append(mean_x)
                self.mean_impacts_running_y.append(mean_y)
                self.std_impacts_running_x.append(std_x)
                self.std_impacts_running_y.append(std_y)
                self.mean_error_running.append(np.linalg.norm([mean_x-Run.target[0], mean_y-Run.target[1]]))
                self.std_error_running.append(std_dist_error)

                std_err_running.append(std_dist_error)

            plt.plot(cov_sqrt, color='gray')
            plt.plot(det_cov_sqrt, color='blue')
            #plt.plot(std_err_running, label='{}: std dist error'.format(mode))
            plt.ylabel('standard deviation $\sigma_i$ [m]')
            plt.xlabel('iteration $i$ [-]')
            plt.ylim([0, 0.45])
            
            # tpl.save(path_iros_plots + 'Results/Pure_Variance')
            # plt.legend()

        return
        plt.figure(figsize=(8, 8))
        plt.plot(self.mean_impacts_running_x, self.mean_impacts_running_y, marker='x', alpha=0.5, label='NN')
        plt.scatter(self.list_runs[0].target[0], self.list_runs[0].target[1], marker='+')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        # plt.ylim([-3.1-0.2, -2.6+0.3])
        plt.ylim([-2.8-0.6, -2.8+0.6])
        # plt.xlim([-0.1, 0.9])
        plt.xlim([-0.5, 0.7])
        plt.legend()
        plt.title('long run - mean impact')#, {} outliers removed'.format(len(self.leave_out)))
        # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_mean_impact')

        plt.figure(figsize=(8, 4))
        plt.plot(self.mean_error_running, label=mode)
        plt.ylabel('distance error [m]')
        plt.xlabel('iteration [-]')
        plt.yscale(yscale)
        plt.legend()
        plt.title('long run - error of mean impact')#, {} outliers removed'.format(len(self.leave_out)))
        # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_mean_impact_error_flat')

        plt.figure(figsize=(8, 8))
        plt.plot(self.std_impacts_running_x, label='{}: x-coord imapct'.format(mode))
        plt.plot(self.std_impacts_running_y, label='{}: y-coord imapct'.format(mode))
        plt.ylabel('standard deviation of distance error [m]')
        plt.xlabel('iteration [-]')
        plt.ylim([0, 0.45])
        plt.legend()
        plt.title('long run - variance evolution per axis')#, {} outliers removed'.format(len(self.leave_out)))
        # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_variance_evolution_1')

        plt.figure(figsize=(8, 8))
        plt.plot(self.std_error_running, label='{}: std dist error'.format(mode))
        plt.ylabel('standard deviation of distance error [m]')
        plt.xlabel('iteration [-]')
        plt.ylim([0, 0.45])
        plt.legend()
        plt.title('long run - variance evolution')#, {} outliers removed'.format(len(self.leave_out)))
        # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_variance_evolution_2')

    def MeanImpactEvolution_multiple(self, yscale='linear', modes=['Physical']):
        '''
        plot the evolution of the mean impact point
        '''
        target_pos_2 = np.array([0.164, 2.79, -0.44]) # mark 2
        target_x =  target_pos_2[0]
        target_y =  target_pos_2[1]

        B = np.array([[-1.0,  0.0,  0.0],
              [ 0.0, -1.0,  0.0],
              [ 0.0,  0.0,  1.0]])

        A = np.array(  [[ 0.99591269, -0.07944872, -0.04296286],
                        [ 0.07685274,  0.99529339, -0.05903159],
                        [ 0.04745064,  0.0554885,   0.99733117]]) # from calibration 2022_09_22
        A = B@A

        CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        self.mean_error_running = []

        for Run in self.list_runs:
            mean_err_running_curr = []
            for j in range(self.n):
                mean_x = np.mean([i[0] for i in Run.impact_hist[:(j+1)]])
                mean_y = np.mean([i[1] for i in Run.impact_hist[:(j+1)]])

                mean_cal = np.array([mean_x, mean_y, -0.44])
                mean_uncal = np.linalg.inv(A)@mean_cal - CALIBRATION

                mean_err_running_curr.append(np.linalg.norm([mean_uncal[0]-target_x, mean_uncal[1]-target_y]))
            self.mean_error_running.append(mean_err_running_curr)

        plt.figure(figsize=(8, 8))
        for r, Run in enumerate(self.list_runs):
            print('RUN ', r, len(self.mean_error_running[r]))

            plt.plot(self.mean_error_running[r], label=modes[r], linewidth=2)

            plt.ylabel('distance error $\epsilon_i$ [m]', fontsize=14)
            plt.xlabel('iteration $i$ [-]', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.yscale(yscale)
            plt.legend(fontsize=12)

            # plt.savefig('/home/philip/Desktop/Results/Results 2022_10_10/Long_mean_impact_error_flat')

    def ImpactHist_Combined(self, label=[None]*10):
        '''
        plot target, all impacts and mean impact
        '''

        B = np.array([[-1.0, 0.0, 0.0],
            [ 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]])

        A = np.array(  [[ 0.99591269, -0.07944872, -0.04296286],
                        [ 0.07685274,  0.99529339, -0.05903159],
                        [ 0.04745064,  0.0554885,   0.99733117]]) # from calibration 2022_09_22
        A = B@A

        CALIBRATION = np.array([ 0.06727473,  0.01105588, -0.08797134]) # from calibration 2022_09_22

        # plt.figure(figsize=(8,8))
        fig, axs = plt.subplots(2, 2, figsize=(8,8))

        for r, Run in enumerate(self.list_runs):

            ax = axs[int(r/2), r%2]

            impact_x = [i[0] for i in Run.impact_hist]
            impact_y = [i[1] for i in Run.impact_hist]   

            # if len(impact_x) > 25:
            #     impacts = [np.array([impact_x[i], impact_y[i], -0.44]) for i in range(len(impact_x)) if i%int(len(impact_x)/25)==0]

            # else:
            #     impacts = [np.array([impact_x[i], impact_y[i], -0.44]) for i in range(len(impact_x))]
        
            if r == 3:
                impacts = [np.array([impact_x[i], impact_y[i], -0.44]) for i in range(170, 195)]
            else:
                impacts = [np.array([impact_x[i], impact_y[i], -0.44]) for i in range(25)]
            
            impacts_uncal = [np.linalg.inv(A)@(i)- CALIBRATION for i in impacts]

            impact_x = [i[0] for i in impacts_uncal]
            impact_y = [i[1] for i in impacts_uncal]

            mean_x = np.mean(impact_x)
            std_x = np.std(impact_x)
            mean_y = np.mean(impact_y)
            std_y = np.std(impact_y)


            ax.scatter(impact_x, impact_y, color=self.colors[r], alpha = 0.8, label=label[r], lw=0.5)#, label='std x = {}, std y = {}'.format(round(std_x,2), round(std_y, 2)))

            # plt.scatter([mean_x], [mean_y], marker='x', lw=2, s=100, label='mean hit point')

            # plt.scatter([mean_x], [mean_y], marker='x', color=self.colors[r], lw=2, s=100)

            if r == 3:
                ax.scatter([self.target[0]], [self.target[1]], color='black', marker='+', lw=2, s=100, label='target')
            else:
                ax.scatter([self.target[0]], [self.target[1]], color='black', marker='+', lw=2, s=100)

            ideal_table = np.array([0.141, 1.74, -0.441]) # from calibration 2022_09_22
            height_of_table = 0.765  # m
            height_of_ground = ideal_table[2] - height_of_table
            center_of_table = ideal_table[0:2]
            half_width_of_table = 1.525/2
            half_length_of_table = 2.74/2
            if r == 3:
                rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5, label='table')
            else:
                rect = patches.Rectangle((ideal_table[0] - half_width_of_table, ideal_table[1] - half_length_of_table), half_width_of_table*2, half_length_of_table*2, linewidth=1, linestyle='--', edgecolor='black', facecolor='none', alpha =0.5)


            ax.add_patch(rect)
            # ax.legend(loc='lower left', fontsize=12)
            # plt.title('Different Initial Guesses')
            ax.set_xlabel('$x$ [m]', fontsize=14)
            ax.set_ylabel('$y$ [m]', fontsize=14)
            ax.tick_params(size=12)
            # ax.tick_params(fontsize=12)
            # plt.ylim([-3.15, -1.55])
            # plt.xlim([-0.9, 0.8])
            # plt.ylim([1.5, 3.5])
            # plt.xlim([-1, 1])
            ax.set_ylim([0, 4])
            ax.set_xlim([-2, 2])

            if r in [0, 1]:
                ax.set_xticks([])
                ax.set_xlabel('')

            if r in [1, 3]:
                ax.set_yticks([])
                ax.set_ylabel('')
        
        fig.legend(loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1))