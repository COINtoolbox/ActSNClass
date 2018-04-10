"""Created by CRP #4 team between 20-27 Aug 2017.

Simple function to randomize objects in a given sample.

- randomiseData
    Separates the complete cross-validation set into query and validation.
"""

import numpy as np

def randomiseData(trainFeatures, trainLabels, cvFeatures,
                  cvLabels, queryFraction=0.8,
                  fixRandomState=False, trainFraction=1.0,
                  pubFeatures=None, pubLabels=None):
    """Separates the complete target set into query and validation sets, 
       with the appropriate randomization.

    input: trainFeatures, array - complete training sample features
           trainLabels, list - complete training sample lables
           cvFeatures, array - complete cross-validation sample features
           cvLabels, list - complete cross-validation sample labels
           queryFraction, float - fraction of cv sample to be used to query
                                  (optional, default is 0.8)
           fixRandomState, boolean - rather to use a fixed random seed
                                     (optional, default is False)
           trainFraction, float - fraction of training set to be used
                                  (optional, default is 1.0)
           pubFeatures - complete publication sample features
                            (this is only used if queryFraction == 0,
                             this correspond to publication option when 
                             cv sample is static, default is None)
           pubLabeles - complete publication sample labels
                            (this is only used if queryFraction == 0,
                             this correspond to publication option when 
                             cv sample is static, default is None)

    output: data, dict - separated train, query and validation sets
              keywords: trainFeatures, trainLabels, 
                        queryPoolFeatures, queryPoolLabels, 
                        validFeatures, validLabels
    """
    
    if (fixRandomState):
        np.random.seed(42)
    
    # Count the number of objects in each set
    nTest = len(cvLabels)
    nTrain = len(trainLabels)

    # Get indexes for test set
    inds = np.arange(nTest)

    # Randomize indexes for test set
    np.random.shuffle(inds)

    if queryFraction > 0:
        # Separate query set indexes
        testInds = inds[:int(nTest * queryFraction)]

        # Separate validation set indexes
        validInds = inds[int(nTest * queryFraction):]

        # Get query and validation data
        queryPoolFeatures, queryPoolLabels = cvFeatures[testInds], cvLabels[testInds]
        validFeatures, validLabels = cvFeatures[validInds], cvLabels[validInds]

    else:
        # complete query is given by the cv sample
        # complete validation is given by the publication sample
        queryPoolFeatures, queryPoolLabels = cvFeatures, cvLabels
        validFeatures, validLabels = pubFeatures, pubLabels
        

    # Now shuffle up the trainingInputs too
    inds = range(nTrain)
    np.random.shuffle(inds)

    # Separate the useful trainig set indexes
    trainInds = np.array([inds[i] for i in range(int(nTrain * trainFraction))])

    # Get training data
    trainFeatures = trainFeatures[trainInds]
    trainLabels = trainLabels[trainInds]

    # Organize output
    data = {}
    data['trainFeatures'] = trainFeatures
    data['trainLabels'] = trainLabels
    data['queryPoolFeatures'] = queryPoolFeatures
    data['queryPoolLabels'] = queryPoolLabels
    data['validFeatures'] = validFeatures
    data['validLabels'] = validLabels

    return data

