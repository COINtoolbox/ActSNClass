"""Created by CRP #4 team between 20-27 August 2017.

Implementation of batch mode queries.
Uses the structure from libact - https://github.com/ntucllab/libact

- BatchQuery
    Batch query class
- RandomBatchQuery
    Random selection of batchSize objects
- LeastCertainBatchQuery
    Chooses the n objects with highest uncertainties
- SemiSupervisedBatchQuery
    Chooses the elements of the batch as detailed in
    XXX et al., 2017
"""

import libact
import copy
from libact.base.interfaces import QueryStrategy
from libact.base.dataset import Dataset
from libact.query_strategies import UncertaintySampling
from libact.utils import seed_random_state
from libact.base.interfaces import ContinuousModel, ProbabilisticModel

import numpy as np

class BatchQuery(QueryStrategy):

    def __init__(self, *args, **kwargs):
        super(BatchQuery, self).__init__(*args, **kwargs)
        
        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        batch_size = kwargs.pop('batch_size', None)
        if (batch_size is None):
            raise TypeError(
                "__init__() missing required keyword-only argument: 'batch_size'"
            )
        
        self.batch_size_=batch_size
        
    def update(self, entry_id, label):
        pass

    def update_batch(self, entry_ids, labels):
        
        if len(entry_ids)==len(labels):
            
            for i in range(len(entry_ids)):
                self.dataset.update(entry_ids[i], labels[i])
            
        else:
            raise ValueError(
                "Id and label numbers not matching in update_batch"
            )
            
class RandomBatchQuery(BatchQuery):

    def __init__(self, *args, **kwargs):
        super(RandomBatchQuery, self).__init__(*args, **kwargs)
    
            
    def make_query(self):    
        
        dataset = self.dataset
        unlabeled_entry_ids, _ = zip(*dataset.get_unlabeled_entries())

        indx = self.random_state_.choice(len(unlabeled_entry_ids),
                                    size=self.batch_size_,replace=False)
        query_ids=[unlabeled_entry_ids[item] for item in indx]
        
        return query_ids

        
class LeastCertainBatchQuery(BatchQuery):

    def __init__(self, *args, **kwargs):
        super(LeastCertainBatchQuery, self).__init__(*args, **kwargs)
    
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
            
        if not isinstance(self.model, ContinuousModel) and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ContinuousModel or ProbabilisticModel"
            )
        
        self.model.train(self.dataset)
            
    def update_batch(self, entry_ids, labels):
        super(LeastCertainBatchQuery, self).update_batch(entry_ids, labels)
        
        self.model.train(self.dataset)
        
    def make_query(self):
    
        unlabeled_entry_ids, unlabeled_entry_features = zip(*self.dataset.get_unlabeled_entries())
    
        probs = self.model.predict_proba(unlabeled_entry_features)
        
        # why is there a- 0.5 here???
        indx = np.argsort(np.abs(probs[:,0]-0.5))[:self.batch_size_]
        queryIds = [unlabeled_entry_ids[item] for item in indx]
        
        return queryIds

        
class SemiSupervisedBatchQuery(BatchQuery):

    def __init__(self, *args, **kwargs):
        super(SemiSupervisedBatchQuery, self).__init__(*args, **kwargs)
    
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
            )
            
        if not isinstance(self.model, ContinuousModel) and \
                not isinstance(self.model, ProbabilisticModel):
            raise TypeError(
                "model has to be a ContinuousModel or ProbabilisticModel"
            )
        
        self.model.train(self.dataset)    

    def update_batch(self, entry_ids, labels):
        super(SemiSupervisedBatchQuery, self).update_batch(entry_ids,
                                                            labels)
        
        self.model.train(self.dataset)    
    
    def make_query(self):
        
        tempDataset = copy.deepcopy(self.dataset)    
        tempModel = copy.deepcopy(self.model)

        queryStrat = UncertaintySampling(tempDataset,
                                         model=tempModel) #Model is fit here
        queryIds = []    
        
        for j in range(self.batch_size_):
            queryId = queryStrat.make_query() #Model is also fit here
            queryIds.append(queryId)
            
            features = tempDataset.get_entries()[queryId][0]
            
            probs = tempModel.predict_proba(features.reshape(1, -1))
            
            # hard coded flag for positive answer - need to improve
            if self.random_state_.rand() < probs[0][0]:
                label = 0
            else:
                label = 1
            
            tempDataset.update(queryId,label)
            
            # tempModel.train(tempDataset) #This is not needed, 
            # since the make_query of UncertaintySampling fits
            
        return queryIds
