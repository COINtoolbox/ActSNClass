"""Created by CRP #4 team between 20-27 Aug 2017.

Simple function to read fitted SN light curves.

- loadData
    Load supernova data from concatenated matrix.
"""

import numpy as np

def loadData(trainDataPath, trainLabelPath, cvDataPath, cvLabelPath,
             cutFirstHalf=False, firstHalfParameters=None,
             pubDataPath=None, pubLabelPath=None):
    """
    Load supernova data from concatenated matrix.

    input: trainDataPath, str - complete path to train feature file
           trainLabelPath, str - complete path to train label file
           cvDataPath, str - complete path to cross-validation feature file
           cvLabelPath, str - complete path to cross-validation label file
           cutFirstHalf, boolean - rather to use partial light curves
                                   (optional, default is False)
           firstHalfParameters, tuple or None - column indexes for epoch cut
                                                in each filter
                                                (optional, default is None)
           trainFraction, float - fraction of training set to use
                                  (optional, default is 1) 
           pubDataPath, str - complete path to publication sample feature file
                              default is None - not to be used during analysis
           pubLabelPath, str - complete path to publication sample label file
                               default is None - not to be used during analysis

    output: tuple - trainingSet, array
                    trainingLables, list
                    crossValidationSet, array
                    crossValidationLabels, list
    """

    # Read training set
    trainingSet=np.loadtxt(trainDataPath)
    trainingLabels=np.genfromtxt(trainLabelPath, dtype="|U16")

    # Read cross-validation set
    crossValidationSet=np.loadtxt(cvDataPath)
    crossValidationLabels=np.genfromtxt(cvLabelPath, dtype="|U16")

    if pubDataPath is not None:
        # Read Publication set
        pubSet=np.loadtxt(pubDataPath)
        pubLabels=np.genfromtxt(pubLabelPath, dtype="|U16")

    else:
        pubSet = None
        pubLabels = None
    
    # Select partial light curve
    if cutFirstHalf:
    
        if (firstHalfParameters is None):
            firstHalfParameters = (4,33,11)
        
        halfCurveIndex = []
        for i in range(firstHalfParameters[0]):
            halfCurveIndex += list(range(0 + i*firstHalfParameters[1],
                                         firstHalfParameters[2] + 
                                         i * firstHalfParameters[1]))

        trainingSet = trainingSet[:,halfCurveIndex]
        crossValidationSet = crossValidationSet[:,halfCurveIndex]
        pubSet = pubSet[:,halfCurveIndex]
    
    return (trainingSet,trainingLabels,crossValidationSet,
                  crossValidationLabels, pubSet, pubLabels)
