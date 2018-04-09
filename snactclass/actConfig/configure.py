"""Created by Emille E. O. Ishida in 6 Sep 2017.

Configuration function to choose between different active learning settings.

- ConfigureStrategy
    Build dictionary with possible configurations.
"""

from initialSetup import initialDataSetup
from initialSetup import initialModelSetup,initialQuerySetup

def ConfigureStrategy(chosenInput):
    """
    Build dictionary with possible configurations. 

    input: chosenInput, dict - user choices
           keywords must include:
               choice, str - one of the following
                 'rand_RF': random sampling with random forest
                 'unc_RF ': uncertainty sampling with random forest
                 'qbc_noble': query by committee, a'la Noble
                 'qbc_robSmall' : query by committee, a'la Robert, 
                                                      small committee
                 'qbc_robBig': query by committee, a'la Robert, 
                                                   big committee
                 'batch_random': batch random sampling, a'la Jim
                 'batch_nlunc': batch N-least uncertain, a'la Jim
                 'batch_semi': batch semi-supervised, a'la Jim
               useOnlyFirstHalf, boolean - rather to apply epoch cuts
               useFixedRandomState, boolean - rather to fix random seed
               validationDataProportion, float - fraction of data separated 
                                                 for validation
               batchSize, int - number of SNe query per batch mode
               queryQuota, int - total number of queries
               mainTreeSize, int - number of trees in Random Forest
               treesInRFCommittee, int - number of RF class in the committee

    output: final_config, dict - parameters for chosen configuration
            keywords are:
                runSetupID, int - numerical identification of configuration
                output_name, str - root of output file names
                modelID, int - model identification
                modelParams, tuple - parameters of the model
                queryStrategyID, int - query strategy identification
                queryParams, tuple or None - parameters of query strategy
                quota, int - number of iterations calling the AL routine 
    """

    config = {}
    config['random_RF'] = {}
    config['random_RF']['runSetupID'] = 0
    config['random_RF']['output_name'] = 'random_RF'
    config['random_RF']['modelID'] = 0
    config['random_RF']['modelParams'] = (chosenInput['mainTreeSize'],)
    config['random_RF']['queryStrategyID'] = 0
    config['random_RF']['queryParams'] = None
    config['random_RF']['quota'] = chosenInput['queryQuota']

    config['unc_RF'] = {}
    config['unc_RF']['runSetupID'] = 1
    config['unc_RF']['output_name'] = 'unc_RF'
    config['unc_RF']['modelID'] = 0
    config['unc_RF']['modelParams'] = (chosenInput['mainTreeSize'],)
    config['unc_RF']['queryStrategyID'] = 1
    config['unc_RF']['quota'] = chosenInput['queryQuota']

    config['qbc_noble'] = {}
    config['qbc_noble']['runSetupID'] = 2
    config['qbc_noble']['output_name'] = 'qbc_noble'
    config['qbc_noble']['modelID'] = 1
    config['qbc_noble']['modelParams'] = (chosenInput['mainTreeSize'], \
                                          chosenInput['treesInRFCommittee'])
    config['qbc_noble']['queryStrategyID'] = 2
    config['qbc_noble']['quota'] = chosenInput['queryQuota']

    config['qbc_robSmall'] = {}
    config['qbc_robSmall']['runSetupID'] = 3
    config['qbc_robSmall']['output_name'] = 'qbc_robSmall'
    config['qbc_robSmall']['modelID'] = 2
    config['qbc_robSmall']['modelParams'] = (None,)
    config['qbc_robSmall']['queryStrategyID'] = 2
    config['qbc_robSmall']['quota'] = chosenInput['queryQuota']

    config['qbc_robBig'] = {}
    config['qbc_robBig']['runSetupID'] = 4
    config['qbc_robBig']['output_name'] = 'qbc_robBig'
    config['qbc_robBig']['modelID'] = 3
    config['qbc_robBig']['modelParams'] = (None,)
    config['qbc_robBig']['queryStrategyID'] = 2
    config['qbc_robBig']['quota'] = chosenInput['queryQuota']

    config['batch_random'] = {}
    config['batch_random']['runSetupID'] = 5
    config['batch_random']['output_name'] = 'batch_random'
    config['batch_random']['modelID'] = 0
    config['batch_random']['modelParams'] = (chosenInput['mainTreeSize'],)
    config['batch_random']['queryStrategyID'] = 3
    config['batch_random']['queryParams'] = (chosenInput['batchSize'],)
    config['batch_random']['quota'] = \
                      chosenInput['queryQuota']/chosenInput['batchSize']

    config['batch_nlunc'] = {}
    config['batch_nlunc']['runSetupID'] = 6
    config['batch_nlunc']['output_name'] = 'batch_nlunc'
    config['batch_nlunc']['modelID'] = 0
    config['batch_nlunc']['modelParams'] = (chosenInput['mainTreeSize'],)
    config['batch_nlunc']['queryStrategyID'] = 4
    config['batch_nlunc']['quota'] = \
                       chosenInput['queryQuota']/chosenInput['batchSize']

    config['batch_semi'] = {}
    config['batch_semi']['runSetupID'] = 7
    config['batch_semi']['output_name'] = 'batch_semi'
    config['batch_semi']['modelID'] = 0
    config['batch_semi']['modelParams'] = (chosenInput['mainTreeSize'],)
    config['batch_semi']['queryStrategyID'] = 5
    config['batch_semi']['quota'] = \
                       chosenInput['queryQuota']/chosenInput['batchSize']

    # Create list of models to be trained
    config['random_RF']['mainModelList'] = \
                initialModelSetup(config['random_RF']['modelID'],
                                  config['random_RF']['modelParams'],
                                  chosenInput['useFixedRandomState'])
    config['unc_RF']['mainModelList'] = \
                initialModelSetup(config['unc_RF']['modelID'],
                                  config['unc_RF']['modelParams'],
                                  chosenInput['useFixedRandomState'])
    config['qbc_noble']['mainModelList'] = \
                initialModelSetup(config['qbc_noble']['modelID'],
                                  config['qbc_noble']['modelParams'],
                                  chosenInput['useFixedRandomState'])
    config['qbc_robSmall']['ModelList'] = \
          initialModelSetup(config['qbc_robSmall']['modelID'], 
                            config['qbc_robSmall']['modelParams'],
                            chosenInput['useFixedRandomState'])
    config['qbc_robSmall']['mainModelList'] = \
          initialModelSetup(0, (chosenInput['mainTreeSize'],),
                                chosenInput['useFixedRandomState'])
    config['qbc_robBig']['ModelList'] = \
          initialModelSetup(config['qbc_robBig']['modelID'], 
                            config['qbc_robBig']['modelParams'],
                            chosenInput['useFixedRandomState'])
    config['qbc_robBig']['mainModelList'] = \
          initialModelSetup(0, (chosenInput['mainTreeSize'],), 
                           chosenInput['useFixedRandomState'])
    config['batch_random']['mainModelList'] = \
                initialModelSetup(config['batch_random']['modelID'],
                                  config['batch_random']['modelParams'],
                                  chosenInput['useFixedRandomState'])
    config['batch_nlunc']['mainModelList'] = \
                initialModelSetup(config['batch_nlunc']['modelID'],
                                  config['batch_nlunc']['modelParams'],
                                  chosenInput['useFixedRandomState'])
    config['batch_semi']['mainModelList'] = \
                initialModelSetup(config['batch_semi']['modelID'],
                                  config['batch_semi']['modelParams'],
                                  chosenInput['useFixedRandomState'])

    # determine query parameters
    config['unc_RF']['queryParams'] = (config['unc_RF']['mainModelList'][0],)
    config['qbc_noble']['queryParams'] = (config['qbc_noble']['mainModelList'],)
    config['qbc_robSmall']['queryParams'] = (config['qbc_robSmall']['ModelList'],)
    config['qbc_robBig']['queryParams'] = (config['qbc_robBig']['ModelList'],)
    config['batch_nlunc']['queryParams'] = \
        (config['batch_nlunc']['mainModelList'][0], chosenInput['batchSize'])
    config['batch_semi']['queryParams'] = \
        (config['batch_semi']['mainModelList'][0], chosenInput['batchSize'])

    # Select choice
    final_config = config[chosenInput['choice']]

    if chosenInput['useOnlyFirstHalf']:
        final_config['queryFile'] = chosenInput['outputFolder'] + 'query_' + \
                                                   final_config['output_name'] + '_halfLC.csv'
        final_config['diagnosticFile'] = chosenInput['outputFolder'] + \
                                   'diagnostic_' + final_config['output_name'] + '_halfLC.csv'
    else:
        final_config['queryFile'] = chosenInput['outputFolder'] + 'query_' + \
                                               final_config['output_name'] + '_fullLC_v2.csv'
        final_config['diagnosticFile'] = chosenInput['outputFolder'] + \
                               'diagnostic_' + final_config['output_name'] + '_fullLC_v2.csv'

    return final_config
