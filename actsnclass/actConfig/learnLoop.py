import numpy as np

from actsnclass.analysis_functions.diagnostics import efficiency, purity, fom

def learnLoop(quota, trainData, fullLabels, validationFeatures,
              validationClasses, labeler, mainModelList, queryStrategy,
              queryStrategyID):
    
    accuracies=[]
    efficiencies=[]
    purities=[]
    foms=[]
    
    queryNames=[]
    queryClasses=[]
    queryLabels=[]
    
    #Could optimise this based on queryStrategyID, but not vital since it only runs once
    [model.train(trainData) for model in mainModelList]
    
    predictedClasses = np.array([model.predict(validationFeatures) for model in mainModelList])
    predictedClasses = np.array(np.sum(predictedClasses, axis=0) > len(mainModelList)/2.0, dtype=np.int32)
    
    accuracies.append(np.mean(predictedClasses == validationClasses))
    efficiencies.append(efficiency(predictedClasses,validationClasses,Ia_flag=1))
    purities.append(purity(predictedClasses,validationClasses,Ia_flag=1))
    foms.append(fom(predictedClasses,validationClasses,Ia_flag=1))
    
    
    for i in range(quota):
        # Standard usage of libact objects
        if i % 5 == 0:
            print('Queries '+str(i))
  
        #UncertaintySampling fits the models here, inefficient
        query_ids = queryStrategy.make_query()
        
        if (queryStrategyID==3 or queryStrategyID==4 or queryStrategyID==5):
            #These are the batch query strategies
            
            labels=[]
            for query_id in query_ids:
                labels.append(labeler.label(trainData.data[query_id][0]))
            
                queryNames.append(fullLabels[query_id,0])
                queryClasses.append(fullLabels[query_id,2])
                queryLabels.append(labels[-1])

            queryStrategy.update_batch(query_ids, labels)  
            #The update_batch fits the models for N-least certain and semisupervised
            #Random does not have an associated model

        else:            
            label=labeler.label(trainData.data[query_ids][0])
            
            #The dataset update fits the models for QueryByCommittee via a callback function
            #Random does not have an associated model, UncertaintySampling isn't updated
            trainData.update(query_ids, label)
        
            queryNames.append(fullLabels[query_ids,0])
            queryClasses.append(fullLabels[query_ids,2])
            queryLabels.append(label)            
            
            #queryStrategy.update(query_id, label)
        
        
        #UncertaintySampling needs to be fitted here with the updated data, inefficient as it is also fitted in the make_query
        #Query by committee might have a separate main model, in that case it has to be fitted too
        #Random methods need the model fitted too
        if (queryStrategyID==0 or queryStrategyID==1 or queryStrategyID==3 
            or (queryStrategyID==2 and len(mainModelList)<queryStrategy.n_students)):

            [model.train(trainData) for model in mainModelList]
        
        predictedClasses = np.array([model.predict(validationFeatures) for model in mainModelList])
        predictedClasses = np.array(np.sum(predictedClasses, axis=0) > len(mainModelList)/2.0, dtype=np.int32)
        
        accuracies.append(np.mean(predictedClasses == validationClasses))
        efficiencies.append(efficiency(predictedClasses,validationClasses,Ia_flag=1))
        purities.append(purity(predictedClasses,validationClasses,Ia_flag=1))
        foms.append(fom(predictedClasses,validationClasses,Ia_flag=1))

    return (accuracies, efficiencies, purities, foms, queryNames, queryClasses, queryLabels, predictedClasses)
