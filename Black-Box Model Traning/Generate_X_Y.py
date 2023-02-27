'''
Extract useful data from whole dataset and build inputs and labels for learning-based landing
point prediction
'''

import pickle
import os
import sys

from RacketHit import HitRecording

# %%

path            = os.path.dirname(__file__)
path_checkpoint = path + '/checkpoint'
path_data       = ''                            # path of folder containing the interception data

X           = []
Y           = []
X_simpler   = []
Y_simpler   = []
used_split  = []

loaded      = 0
index0      = -1
n           = 12600

if os.path.exists(path_checkpoint):
    with open(path_checkpoint, 'rb') as f:
        X = pickle.load(f)
        Y = pickle.load(f)
        X_simpler = pickle.load(f)
        Y_simpler = pickle.load(f)
        index0 = pickle.load(f)
        loaded = pickle.load(f)
        used_split = pickle.load(f)

for index in range(index0+1, n):

    Hit = HitRecording(index=index, path=path_data)
    if Hit.load:
        used_split.append(True)  

        loaded += 1
        X_simpler.append([Hit.racket_impact_x[i] for i in [30, 33, 34]])
        Y_simpler.append(Hit.racket_impact_y[:2])

        X.append(Hit.racket_impact_x)
        Y.append(Hit.racket_impact_y)

    else:
        used_split.append(False)

    sys.stdout.write('\r')
    sys.stdout.write('Hits loading... {}% - {}/{}'.format(round( index / n * 100 ), index, loaded))
    sys.stdout.flush()

    if index % 100 == 0 and index > 0:
        with open(path_checkpoint, 'wb') as f:
            pickle.dump(X, f)
            pickle.dump(Y, f)
            pickle.dump(X_simpler, f)
            pickle.dump(Y_simpler, f)
            pickle.dump(index, f)
            pickle.dump(loaded, f)
            pickle.dump(used_split, f)

with open(path + '/RacketImpact_X_and_Y_simpler', 'wb') as f:
    pickle.dump(X_simpler, f)
    pickle.dump(Y_simpler, f)
    pickle.dump(used_split, f)

with open(path + '/RacketImpact_X_and_Y', 'wb') as f:
    pickle.dump(X, f)
    pickle.dump(Y, f)
    pickle.dump(used_split, f)

