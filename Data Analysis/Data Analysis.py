'''
This script is used for the data analysis of individual online optimization runs and multiple runs
We can either use the high level data saved for a run or load each interception individually, 
which is controlled by the load_individually setting
'''

import numpy as np
import matplotlib.pyplot as plt
import os

import tikzplotlib as tpl

from FunctionsDataAnalysis import RepeatedHits, ListRepeatedHits

Desktop = r'/home/philip//Desktop/'

individual          = False

long_physical_vs_nn = True
different_target    = False
different_initial   = False
pure_variance       = False
video               = False
human_benchmark     = False



# ---- individual online optimization runs

if individual:
    for index in [6393]:
        print(index)
        RepHits = RepeatedHits(index=index, n=200, load_individually=True)
        # RepHits.MeanImpactEvolution(yscale='log')
        # RepHits.ImpactVariance()
        # RepHits.ThetaRealvsError()
        # RepHits.ErrorHist()

        plt.show()

different_targets_nn        = [5889, 5929, 6010]
different_targets_physical  = [6413, 6485, 6505, 6525, 6558]
different_ig_nn             = [6062, 6082, 6123, 6143, 6183]
different_ig_physical       = [7759, 7782, 7803, 7841, 7865]

repeated_opt_physical       = [6602, 6623, 6643, 6664, 6685, 6715]

different_ig_physical_aggressive_repeated = [i for i in range(7895, 8068)]
different_ig_physical_aggressive_repeated = different_ig_physical_aggressive_repeated + [8174, 8179, 8189, 8203]

different_target_physical_aggressive_repeated = [8605, 8610, 8615, 8620, 8631, 8636,
                                                    8647, 8652, 8657, 8667, 
                                                    8672, 8677, 8682, 8687, 8692, 
                                                    8697, 8707, 8712, 8726, 
                                                    8732, 8737, 8742, 8747, 8752,
                                                    8757, 8762, 8770, 8775, 8780]

video_run = [12528]

humans_target_2 = [8922, 9027, 9102]


# ---- LONG RUN PHYSICAL VS NN

if long_physical_vs_nn:
    list_runs = []
    for index in [6393, 7413]:
        if os.path.exists(Desktop + r'Repeated Hit Statistics/Data Online Learning/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(index=index, n=200, leave_out=[], load_individually=False))
            # list_runs[-1].ThetaEvolution()
        else:
            print('index not found: ', index)

    # plt.show()

    ListRepeated = ListRepeatedHits(list_runs, leave_out=[])
    ListRepeated.n = 200

    # ListRepeated.ErrorHist()
    # ListRepeated.ImpactHist_WithTable()
    # ListRepeated.ErrorHist_NNvsPhysical()
    ListRepeated.ThetaEvolution()
    ListRepeated.MeanImpactEvolution_NNvsPhysical(yscale='log')

    # ListRepeated.InitialGuessConvergence_Averaged()
    # ListRepeated.TargetConvergence_Averaged()
    # ListRepeated.MeanImpactEvolution(yscale='log', mode='Physical')

    plt.show()


# ---- PURE VARIANCE

# policies used:  9737: [0.25  0.709 1.56  0.    6.    0.    0.   ]
#                11452: [0.32  0.709 1.56  0.05  6.    0.    0.   ]
#                12384: [ 0.5  0.709 1.56  -0.1  6.    0.    0.   ]

if pure_variance:
    list_runs = []
    for index in [9737, 11452, 12384]:
        if os.path.exists(Desktop + r'Repeated Hit Statistics/Data Online Learning/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(index=index, n=200, leave_out=[], load_individually=False))
        else:
            print('index not found: ', index)

    ListRepeated = ListRepeatedHits(list_runs, leave_out=[])
    ListRepeated.n = 200

    # ListRepeated.ErrorHist()
    # ListRepeated.ImpactHist_WithTable()
    # ListRepeated.ErrorHist_NNvsPhysical()
    # ListRepeated.MeanImpactEvolution_NNvsPhysical(yscale='log')
    # ListRepeated.InitialGuessConvergence_Averaged()
    # ListRepeated.TargetConvergence_Averaged()
    ListRepeated.MeanImpactEvolution(yscale='log', mode='Physical')
    plt.show()


# ---- DIFFERENT INITIAL GUESSES

if different_initial:
    list_runs = []
    for index in different_ig_physical_aggressive_repeated:
        if os.path.exists(Desktop + r'Repeated Hit Statistics/Data Online Learning/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(index=index, n=5, leave_out=[], load_individually=False))
        else:
            print('index not found: ', index)

    ListRepeated = ListRepeatedHits(list_runs, leave_out=[])

    # ListRepeated.ErrorHist()
    # ListRepeated.ImpactHist_WithTable()
    # ListRepeated.ErrorHist_NNvsPhysical()
    # ListRepeated.MeanImpactEvolution_NNvsPhysical(yscale='log')
    ListRepeated.InitialGuessConvergence_Averaged()
    # ListRepeated.TargetConvergence_Averaged()
    # ListRepeated.MeanImpactEvolution(yscale='log', mode='Physical')
    # plt.show()


# ---- DIFFERENT TARGETS

if different_target:
    list_runs = []
    for index in different_target_physical_aggressive_repeated:
        if os.path.exists(Desktop + r'Repeated Hit Statistics/Data Online Learning/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(index=index, n=5, leave_out=[], load_individually=False))
        else:
            print('index not found: ', index)

    ListRepeated = ListRepeatedHits(list_runs, leave_out=[])

    # ListRepeated.ErrorHist()
    # ListRepeated.ImpactHist_WithTable()
    # ListRepeated.ErrorHist_NNvsPhysical()
    # ListRepeated.MeanImpactEvolution_NNvsPhysical(yscale='log')
    # ListRepeated.InitialGuessConvergence_Averaged()
    ListRepeated.TargetConvergence_Averaged()
    # ListRepeated.MeanImpactEvolution(yscale='log', mode='Physical')
    plt.show()


# ---- VIDEO RUN

if video:
    list_runs = []
    for index in video_run:
        if os.path.exists(Desktop + r'Repeated Hit Statistics/Data Online Learning/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(index=index, n=10, leave_out=[], load_individually=True))
        else:
            print('index not found: ', index)

    ListRepeated = ListRepeatedHits(list_runs, leave_out=[], n=10)

    list_runs[0].ErrorHist()

    # ListRepeated.ErrorHist()
    ListRepeated.ImpactHist_WithTable()
    # ListRepeated.ErrorHist_NNvsPhysical()
    # ListRepeated.MeanImpactEvolution_NNvsPhysical(yscale='log')
    # ListRepeated.InitialGuessConvergence_Averaged()
    # ListRepeated.TargetConvergence_Averaged()
    # ListRepeated.MeanImpactEvolution(yscale='log', mode='Human Play')
    plt.show()

# HUMAN BENCHMARK
human = False
if human:
    list_runs = []
    for index in [6393]+humans_target_2:
        if os.path.exists(Desktop + r'Human Play Data/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(path=Desktop + r'Human Play Data/' , index=index, n=25, leave_out=[], load_individually=False))
            list_runs[-1].target = np.array([0.164, 2.79, -0.44]) # mark 2
            # list_runs[-1].ImpactVariance()
        elif os.path.exists(Desktop + r'Repeated Hit Statistics/Data Online Learning/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(path=Desktop + r'Repeated Hit Statistics/Data Online Learning/' , index=index, n=200, leave_out=[], load_individually=False))
            list_runs[-1].target = np.array([0.164, 2.79, -0.44]) # mark 2
            # list_runs[-1].ImpactVariance()
        else:
            print('index not found: ', index)

    ListRepeated = ListRepeatedHits(list_runs, leave_out=[])
    ListRepeatedHits.n = 25

    # list_runs[0].ErrorHist()

    # ListRepeated.ErrorHist()
    # ListRepeated.ImpactHist_WithTable()
    # ListRepeated.ErrorHist_NNvsPhysical()
    # ListRepeated.MeanImpactEvolution_NNvsPhysical(yscale='log')
    # ListRepeated.InitialGuessConvergence_Averaged()
    # ListRepeated.TargetConvergence_Averaged()
    ListRepeated.MeanImpactEvolution_multiple(yscale='log', modes=['robot', 'amateur 1', 'amateur 2', 'advanced amateur 1'])
    plt.show()


# human_comparison = [9102, 6393]
# human_comparison = [8922, 9027, 9102, 6393]
human_comparison = [9027, 9102, 6393, 6393]

if human_benchmark:
    list_runs = []
    for index in human_comparison:
        if os.path.exists(Desktop + r'Human Play Data/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(path=Desktop + r'Human Play Data/' , index=index, n=25, leave_out=[], load_individually=False))
            list_runs[-1].target = np.array([0.164, 2.79, -0.44]) # mark 2
            # list_runs[-1].ImpactVariance()
        if os.path.exists(Desktop + r'Repeated Hit Statistics/Data Online Learning/Run_{}'.format(index)):
            list_runs.append(RepeatedHits(path=Desktop + r'Repeated Hit Statistics/Data Online Learning/' , index=index, n=200, leave_out=[], load_individually=False))
            list_runs[-1].target = np.array([0.164, 2.79, -0.44]) # mark 2
            # list_runs[-1].ImpactVariance()

    ListRepeated = ListRepeatedHits(list_runs, leave_out=[])
    ListRepeated.target = np.array([0.164, 2.79, -0.44]) # mark 2

    ListRepeated.ImpactHist_Combined(label=['amateur', 'advanced amateur', 'robot iterations 0-24', 'robot iterations 175-199'])
    plt.show()