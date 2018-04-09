from snactclass import loadData
from snactclass import randomiseData

from snactclass import learnLoop
from snactclass.actConfig import ConfigureStrategy

from snactclass.analysis_functions import save_results

from snactclass.actConfig import initialDataSetup
from snactclass.actConfig import initialQuerySetup

# Set data paths
trainDataPath = '../snactclass/data/train_lightcurves.txt'
trainLabelPath = '../snactclass/data/train_labels.txt'
cvDataPath = '../snactclass/data/crossValidation_lightcurves.txt'
cvLabelPath = '../snactclass/data/crossValidation_labels.txt'
rawDataPath = '../data/SIMGEN_PUBLIC_DES/'     # this is too big,
                                       # but it is in the other repository

# Store user choices
user_input = {}

# Data preparation options
user_input['useOnlyFirstHalf'] = False                # use only first 6 points in the light curve
user_input['useFixedRandomState'] = True              # Fix the random seed
user_input['validationDataProportion'] = 0.2          # fraction of photometric data separated  for validation
                                                      # (this is not available for query)
user_input['trainFraction'] = 1.0                     # fraction of training data to be used 

# Active Learning options
user_input['batchSize'] = 10                          # number of queries in each batch
user_input['queryQuota'] = 500                        # total  number of queries

# Random Forest options
user_input['mainTreeSize'] = 1000            # number of trees in RF
user_input['treesInRFCommittee'] = 15        # Only used with runSetupID 2

# Active learning configuration , options are:
#
#'random_RF'        : random sampling with random forest
#'unc_RF '        : uncertainty sampling with random forest
#'qbc_noble'      : query by committee, a'la Noble
#'qbc_robSmall'   : query by committee, a'la Robert, small committee
#'qbc_robBig'     : query by committee, a'la Robert, big committee
#'batch_random'   : batch random sampling, a'la Jim
#'batch_nlunc'    : batch N-least uncertain, a'la Jim
#'batch_semi'     : batch semi-supervised, a'la Jim
user_input['choice'] = 'batch_nlunc'
user_input['outputFolder'] = user_input['choice'] + '/'
user_input['diagnosticFile'] = 'diag_' + user_input['choice'] + '_fullSample.csv'
user_input['queryFile'] = 'queries_' + user_input['choice'] + '_fullSample.csv'

# Build the chosen strategy
myStrategy = ConfigureStrategy(user_input)

# load data
(trainingSet, trainingLabels, crossValidationSet, crossValidationLabels) = \
                loadData(trainDataPath, trainLabelPath, cvDataPath, cvLabelPath)

# randomise data
randData = randomiseData(trainingSet, trainingLabels, crossValidationSet,
         crossValidationLabels, 1-user_input['validationDataProportion'],
                                       user_input['useFixedRandomState'],
                               trainFraction=user_input['trainFraction'])


# create ideal labeler                                                                        
(trainData, fullLabels, labeler) = initialDataSetup(randData['trainFeatures'],
                                                      randData['trainLabels'],
                                                randData['queryPoolFeatures'],
                                                  randData['queryPoolLabels'],
                                                                  SNLabel='0') 

# set up query strategy
queryStrategy = initialQuerySetup(trainData, myStrategy['queryStrategyID'],
                                                 myStrategy['queryParams'], 
                                         user_input['useFixedRandomState'])



# perform AL loop
(accuracies, effs, purs, foms, queryNames, queryClasses, queryLabels) = \
                                               learnLoop(myStrategy['quota'],
                                                                   trainData, 
                                                                  fullLabels,
                                                   randData['validFeatures'], 
                             (randData['validLabels'][:,2]=='0').astype(int), 
                                                                     labeler, 
                                                 myStrategy['mainModelList'], 
                                                               queryStrategy,
                                               myStrategy['queryStrategyID'])


# Output options
save_results(accuracies, effs, purs, foms, queryNames, queryClasses,
                               queryLabels, user_input, rawDataPath)




