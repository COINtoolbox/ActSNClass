"""Created by CRP #4 Team between 20-27 Aug 2017
"""

import pandas as pd
import numpy as np

class MLResult(object):
    
    _required_keys = ["fom", "efficiency", "purity", "accuracy"]
    
    def __init__(self, snids, fom, efficiency, purity, accuracy, **kwargs):
    
        table = {}
        
        n_entries = len(snids)
        # Create number of queries vector
        indx = range(0, n_entries)
        
        #table['snids'] = snids
        table['fom'] = fom
        table['efficiency'] = efficiency
        table['purity'] = purity
        table['accuracy'] = accuracy

        diag_rows = np.vstack((indx, snids, accuracy, efficiency, purity, fom))
        diag_columns = diag_rows.transpose()
        
        for key, value in kwargs.iteritems():
            
            table[key] = value
            
        for key in MLResult._required_keys:
            
            assert len(table[key]) == n_entries, '%s has the wrong length'%key
            
        for key in kwargs.keys():
            
            assert len(table[key]) == n_entries, '%s has the wrong length'%key

        self._table = pd.DataFrame(diag_columns, columns=['nqueries', 'snids', 'accuracy', 
                                                'eff', 'pur', 'fom'])
        self._table['nqueries'] = self._table['nqueries'].astype(int)      
        
    def save(self,file_name):
       
        self._table.to_csv(file_name, 
                           index=False, header=['nqueries', 'snids', 'accuracy', 'eff',
                                        'pur', 'fom'])
        
    @property
    def results(self):

        return self._table

        
    @classmethod
    def from_save(file_name):
        return pd.DataFrame.from_csv(file_name, header=True)
            
            
            
            
        
        
        
    
