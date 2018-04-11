"""
Developed by the CRP #4 team. 
CRP #4 was held in Clermont Ferrand, France in August 2017.

Runs the entire AL pipeline for the static, full light curve analysis.
"""

from actsnclass import loadData
from actsnclass import randomiseData

from actsnclass import learnLoop
from actsnclass.actConfig import ConfigureStrategy

from actsnclass.analysis_functions import save_results

from actsnclass.actConfig import initialDataSetup
from actsnclass.actConfig import initialQuerySetup

import numpy as np

# Set data paths
trainDataPath = '../data/train_fitparameters.dat'                      # path to training data matrix (light curve fit parameters)
trainLabelPath = '../data/train_labels.dat'                            # path to training labels
cvDataPath = '../data/query_fitparameters.dat'                         # path to query data matrix (light curve fit parameters)
cvLabelPath = '../data/query_labels.dat'                               # path to query labels
vanillaDataPath = '../data/pseudotrain_fitparameters.txt'              # path to pseudo-train data matrix (light curve fit parameters)
vanillaLabelPath = '../data/pseudotrain_labels.dat'                    # path to pseudo-train labels
rawDataPath = '../../data/SIMGEN_PUBLIC_DES/'                          # path to raw data

# Store user choices
user_input = {}

# Data preparation options
user_input['useOnlyFirstHalf'] = False                         # use True if you wish to ignore part of collumns
user_input['useFixedRandomState'] = True                       # use the same random seeds throughout the analysis
user_input['queryFraction'] = 0.8                              # fraction of data available for query (use >0 only when query/target are not separated before hand)
user_input['trainFraction'] = 1.0                              # fraction of training to be used
user_input['vanilla'] = True                                   # use True if you wish to run the canonical strategy

# Active Learning options
user_input['batchSize'] = 5                    # this is only used for batch-mode
user_input['queryQuota'] = 1000                # total number of queries

# Random Forest options
user_input['mainTreeSize'] = 1000              # Number of trees in the random forest              
user_input['treesInRFCommittee'] = 15          # Only used for QBC2

# Active learning configuration , options are:
#
#'random_RF'      : random sampling with random forest
#'unc_RF '        : uncertainty sampling with random forest
#'qbc_noble'      : query by committee, QBC2
#'qbc_robSmall'   : query by committee, QBC1
#'batch_random'   : batch random sampling
#'batch_nlunc'    : batch N-least uncertain
#'batch_semi'     : batch semi-supervised

user_input['choice'] = 'unc_RF'                                                       # query strategy
user_input['outputFolder'] = user_input['choice'] + '/'                               # folder to dump results
user_input['diagnosticFile'] = 'diag_' + user_input['choice'] + '_fullsample.csv'     # output diagnostic file 
user_input['queryFile'] = 'queries_' + user_input['choice']  + '_fullsample.csv'      # output query objs file


if user_input['vanilla']:
    user_input['choice'] = 'random_RF'
    user_input['outputFolder'] = user_input['choice'] + '/'
    user_input['diagnosticFile'] = 'diag_' + user_input['choice'] + '_fullsample_canonical.csv'
    user_input['queryFile'] = 'queries_' + user_input['choice'] + '_fullsample_canonical.csv'


    if 'canonical' not in user_input['diagnosticFile']:
        raise ValueError('Please correct name of output file!')

    # Load new train data
    op1 = open(vanillaDataPath, 'r')
    lin1 = op1.readlines()
    op1.close()

    vanillaFeatures = np.array([[float(item) for item in elem.split()] for elem in lin1[1:]])

    op2 = open(vanillaLabelPath, 'r')
    lin2 = op2.readlines()
    op2.close()

    vanillaLabels = [elem.split() for elem in lin2[1:]]


print user_input['diagnosticFile']

# Build the chosen strategy
myStrategy = ConfigureStrategy(user_input)

# load data
(trainingSet, trainingLabels, querySet, queryLabels, pubSet, pubLabels) = \
                         loadData(trainDataPath, trainLabelPath, cvDataPath,
                   cvLabelPath, cutFirstHalf=user_input['useOnlyFirstHalf'])

# randomise data
randData = randomiseData(trainingSet, trainingLabels, querySet,
       queryLabels, queryFraction=user_input['queryFraction'],
                        fixRandomState=user_input['useFixedRandomState'],
                               trainFraction=user_input['trainFraction'])
if user_input['vanilla']:
    randData['queryPoolFeatures'] = vanillaFeatures
    randData['queryPoolLabels'] = np.array(vanillaLabels)


# create ideal labeler                                                                        
(trainData, fullLabels, labeler) = initialDataSetup(randData['trainFeatures'],
                                                      randData['trainLabels'],
                                                randData['queryPoolFeatures'],
                                                  randData['queryPoolLabels'],
                                                                  SNLabel='0') 

queryStrategy = initialQuerySetup(trainData, myStrategy['queryStrategyID'],
                                                 myStrategy['queryParams'], 
                                         user_input['useFixedRandomState'])



(accuracies, effs, purs, foms, queryNames, queryClasses, queryLabels, predictedClasses) = \
                                               learnLoop(myStrategy['quota'],
                                                                   trainData, 
                                                                  fullLabels,
                                                   randData['validFeatures'], 
                             (randData['validLabels'][:,2]=='0').astype(int), 
                                                                     labeler, 
                                                 myStrategy['mainModelList'], 
                                                               queryStrategy,
                                               myStrategy['queryStrategyID'])


print 'Complete code'
print user_input['diagnosticFile']
print 'random seed = ' + str(user_input['useFixedRandomState'])
print 'eff = ' + str(effs[0])
print 'pur = ' + str(purs[0])
print 'fom = ' + str(foms[0])

# Output options
save_results(accuracies, effs, purs, foms, queryNames, queryClasses,
                               queryLabels, user_input, rawDataPath, Ia_label=1)




