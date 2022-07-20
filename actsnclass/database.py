# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 10 August 2019
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import pandas as pd

from actsnclass.classifiers import random_forest
from actsnclass.query_strategies import uncertainty_sampling, random_sampling
from actsnclass.metrics import get_snpcc_metric


__all__ = ['DataBase']


class DataBase:
    """DataBase object, upon which the active learning loop is performed.

    Attributes
    ----------
    classprob: np.array()
        Classification probability for all objects, [pIa, pnon-Ia].
    data: pd.DataFrame
        Complete information read from features files.
    features: pd.DataFrame()
        Feature matrix to be used in classification (no metadata).
    features_names: list
        Header for attribute `features`.
    metadata: pd.DataFrame
        Features matrix which will not be used in classification.
    metadata_names: list
        Header for metadata.
    metrics_list_names: list
        Values for metric elements.
    predicted_class: np.array()
        Predicted classes - results from ML classifier.
    queried_sample: np.array()
        Complete information of queried objects.
    queryable_ids: np.array()
        Flag for objects available to be queried.
    test_features: pd.DataFrame
        Features matrix for the test sample.
    test_metadata: pd.DataFrame()
        Metadata for the test sample
    test_labels: np.array()
        True classification for the test sample.
    train_features: pd.DataFrame()
        Features matrix for the train sample.
    train_metadata: pd.DataFrame()
        Metadata for the training sample.
    train_labels: np.array()
        Classes for the training sample.

    Methods
    -------
    load_bazin_features(path_to_bazin_file: str)
        Load Bazin features from file
    load_features(path_to_file: str, method: str)
        Load features according to the chosen feature extraction method.
    build_samples(initial_training: str or int, nclass: int)
        Separate train and test samples.
    classify(method: str)
        Apply a machine learning classifier.
    evaluate_classification(metric_label: str)
        Evaluate results from classification.
    make_query(strategy: str, batch: int) -> list
        Identify new object to be added to the training sample.
    update_samples(query_indx: list)
        Add the queried obj(s) to training and remove them from test.
    save_metrics(loop: int, output_metrics_file: str)
        Save current metrics to file.
    save_queried_sample(queried_sample_file: str, loop: int, full_sample: str)
        Save queried sample to file.

    Examples
    --------
    >>> from actsnclass import DataBase

    Define the necessary paths

    >>> path_to_bazin_file = 'results/Bazin.dat'
    >>> metrics_file = 'results/metrics.dat'
    >>> query_file = 'results/query_file.dat'

    Initiate the DataBase object and load the data.
    >>> data = DataBase()
    >>> data.load_features(path_to_bazin_file, method='Bazin')

    Separate training and test samples and classify

    >>> data.build_samples(initial_training='original', nclass=2)
    >>> data.classify(method='RandomForest')
    >>> print(data.classprob)          # check predicted probabilities
    [[0.461 0.539]
    [0.346print(data.metrics_list_names)           # check metric header
    ['acc', 'eff', 'pur', 'fom']

    >>> print(data.metrics_list_values)          # check metric values
    [0.5975434599574068, 0.9024767801857585,
    0.34684684684684686, 0.13572404702012383] 0.654]
    ...
    [0.398 0.602]
    [0.396 0.604]]

    Calculate classification metrics

    >>> data.evaluate_classification(metric_label='snpcc')
    >>>

    Make query, choose object and update samples

    >>> indx = data.make_query(strategy='UncSampling', batch=1)
    >>> data.update_samples(indx)

    Save results to file

    >>> data.save_metrics(loop=0, output_metrics_file=metrics_file)
    >>> data.save_queried_sample(loop=0, queried_sample_file=query_file,
    >>>                          full_sample=False)
    """

    def __init__(self):
        self.classprob = np.array([])
        self.data = pd.DataFrame()
        self.features = pd.DataFrame([])
        self.features_names = []
        self.metadata = pd.DataFrame()
        self.metadata_names = []
        self.metrics_list_names = []
        self.metrics_list_values = []
        self.predicted_class = np.array([])
        self.queried_sample = []
        self.queryable_ids = np.array([])
        self.test_features = np.array([])
        self.test_metadata = pd.DataFrame()
        self.test_labels = np.array([])
        self.train_features = np.array([])
        self.train_metadata = pd.DataFrame()
        self.train_labels = np.array([])

    def load_bazin_features(self, path_to_bazin_file: str, screen=False):
        """Load Bazin features from file.

        Populate properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_bazin_file: str
            Complete path to Bazin features file.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        """

        # read matrix with Bazin features
        self.data = pd.read_csv(path_to_bazin_file, sep=' ', index_col=False)

        # list of features to use
        self.features_names = ['gA', 'gB', 'gt0', 'gtfall', 'gtrise', 'rA',
                               'rB', 'rt0', 'rtfall', 'rtrise', 'iA', 'iB',
                               'it0', 'itfall', 'itrise', 'zA', 'zB', 'zt0',
                               'ztfall', 'ztrise']

        self.metadata_names = ['id', 'redshift', 'type', 'code', 
                               'orig_sample', 'queryable']          

        self.features = self.data[self.features_names].values

        if 'queryable' not in self.data.keys():
            self.data['queryable'] = True
            
        self.metadata = self.data[self.metadata_names]
        
        if screen:
            print('Loaded ', self.metadata.shape[0], ' samples!')

    def load_features(self, path_to_file: str, method='Bazin',
                      screen=False):
        """Load features according to the chosen feature extraction method.

        Populates properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_file: str
            Complete path to features file.
        method: str (optional)
            Feature extraction method. The current implementation only
            accepts method=='Bazin'
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        """

        if method == 'Bazin':
            self.load_bazin_features(path_to_file, screen=screen)
        else:
            raise ValueError('Only Bazin features are implemented! '
                             '\n Feel free to add other options.')

    def build_samples(self, initial_training: "str or int", nclass=2,
                      screen=False, queryable=False):
        """Separate train and test samples.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        initial_training : str or int
            Choice of initial training sample.
            If 'original': begin from the train sample flagged in the file
            If int: choose the required number of samples at random,
            ensuring that at least half are SN Ia.
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        queryable: bool (optional)
            If True build also queryable sample for time domain analysis.
            Default is False.
        """

        # separate original training and test samples
        if initial_training == 'original':
            train_flag = self.metadata['orig_sample'] == 'train'
            train_data = self.features[train_flag]
            self.train_features = train_data
            self.train_metadata = self.metadata[train_flag]

            test_flag = self.metadata['orig_sample'] == 'test'
            test_data = self.features[test_flag]
            self.test_features = test_data
            self.test_metadata = self.metadata[test_flag]

            if queryable:
                queryable_flag = self.metadata['queryable'].values
                self.queryable_ids = self.metadata[queryable_flag]['id'].values
            else:
                self.queryable_ids = self.test_metadata['id'].values

            if nclass == 2:
                train_ia_flag = self.train_metadata['type'] == 'Ia'
                self.train_labels = train_ia_flag.astype(int)

                test_ia_flag = self.test_metadata['type'] == 'Ia'
                self.test_labels = test_ia_flag.astype(int)
            else:
                raise ValueError("Only 'Ia x non-Ia' are implemented! "
                                 "\n Feel free to add other options.")

        elif isinstance(initial_training, int):

            if initial_training > 0:
                # initialize the temporary label holder
                train_indexes = np.random.choice(np.arange(0, self.metadata.shape[0]),
                                             size=initial_training, replace=False)
                temp_labels = self.metadata['type'].values[train_indexes]

                # make sure there are the correct ratio of Ias and non-Ias
                while sum(temp_labels == 'Ia') != max(1, initial_training // 2):
                    # this is an array of 5 indexes
                    train_indexes = np.random.choice(np.arange(0, self.metadata.shape[0]),
                                                      size=initial_training,
                                                      replace=False)
                    temp_labels = self.metadata['type'].values[train_indexes]

                if screen:
                    print('\n temp_labels = ', temp_labels, '\n')

                # set training
                train_flag = self.metadata['type'].values[train_indexes] == 'Ia'
                self.train_labels = train_flag.astype(int)
                self.train_features = self.features[train_indexes]
                self.train_metadata = pd.DataFrame(self.metadata.values[train_indexes],
                                                   columns=self.metadata_names)
            else:
                train_indexes = []

            # set test set as all objs apart from those in training
            test_indexes = np.array([i for i in range(self.metadata.shape[0])
                                     if i not in train_indexes])
            test_ia_flag = self.metadata['type'].values[test_indexes] == 'Ia'
            self.test_labels = test_ia_flag.astype(int)
            self.test_features = self.features[test_indexes]
            self.test_metadata = self.metadata.iloc[test_indexes]

            if queryable:
                test_flag = np.array([item in test_indexes 
                                      for item in range(self.metadata.shape[0])])
                queryable_flag = self.metadata['queryable'].values
                combined_flag = np.logical_and(test_flag, queryable_flag)
                self.queryable_ids = self.metadata[combined_flag]['id'].values
            else:
                self.queryable_ids = self.test_metadata['id'].values

        else:
            raise ValueError('"Initial training" should be '
                             '"original" or integer.')

        if screen:
            print('Training set size: ', self.train_metadata.shape[0])
            print('Test set size: ', self.test_metadata.shape[0])

    def classify(self, method='RandomForest', screen=False, n_est=1000, seed=42,
                max_depth=None):
        """Apply a machine learning classifier.

        Populate properties: predicted_class and class_prob

        Parameters
        ----------
        method: str (optional)
            Chosen classifier.
            The current implementation on accepts `RandomForest`.
        n_est: int (optional)
            Number of trees. Default is 1000.
        screen: bool (optional) 
            If True, print debug comments on screen.
            Default is False.
        seed: int (optional)
            Random seed. Default is 42.
        max_depth: None or int (optional)
            The maximum depth of the tree. Default is None.
        """

        if screen:
            print('\n Inside classify: ')
            print('   ... train_features: ', self.train_features.shape)
            print('   ... train_labels: ', self.train_labels.shape)
            print('   ... test_features: ', self.test_features.shape)

        if method == 'RandomForest':
            self.predicted_class,  self.classprob = \
                random_forest(self.train_features, self.train_labels,
                              self.test_features, nest=n_est, seed=seed,
                              max_depth=max_depth)

        else:
            raise ValueError('Only RandomForest classifier is implemented!'
                             '\n Feel free to add other options.')

    def evaluate_classification(self, metric_label='snpcc', screen=False):
        """Evaluate results from classification.

        Populate properties: metric_list_names and metrics_list_values.

        Parameters
        ----------
        metric_label: str
            Choice of metric. Currenlty only `snpcc` is accepted.
        """

        if metric_label == 'snpcc':
            self.metrics_list_names, self.metrics_list_values = \
                get_snpcc_metric(list(self.predicted_class),
                                 list(self.test_labels))
        else:
            raise ValueError('Only snpcc metric is implemented!'
                             '\n Feel free to add other options.')

        if screen:
            print('\n Metrics: ', self.metrics_list_values)

    def make_query(self, strategy='UncSampling', batch=1,
                   seed=42, screen=False) -> list:
        """Identify new object to be added to the training sample.

        Parameters
        ----------
        seed: int (optional)
            Random seed. Default is 42.
        strategy: str (optional)
            Strategy used to choose the most informative object.
            Current implementation accepts 'UncSampling' and
            'RandomSampling'. Default is `UncSampling`.

        batch: int (optional)
            Number of objects to be chosen in each batch query.
            Default is 1.
        screen: bool (optional)
            If true, display on screen information about the
            displacement in order and classificaion probability due to
            constraints on queryable sample.

        Returns
        -------
        query_indx: list
            List of indexes identifying the objects to be queried in decreasing
            order of importance.
            If strategy=='RandomSampling' the order is irrelevant.
        """
        if screen:
            print('\n Inside make_query: ')
            print('       ... classprob: ', self.classprob.shape[0])
            print('       ... queryable_ids: ', self.queryable_ids.shape[0])
            print('       ... test_ids: ', self.test_metadata.shape[0])

        if strategy == 'UncSampling':
            query_indx = uncertainty_sampling(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata['id'].values,
                                              batch=batch, screen=screen)

        elif strategy == 'RandomSampling':
            query_indx = random_sampling(queryable_ids=self.queryable_ids,
                                         test_ids=self.test_metadata['id'].values,
                                         batch=batch, seed=seed, screen=screen)

            if screen:
                print('\n    classprob: ', self.classprob[query_indx[0]])

        else:
            raise ValueError('Invalid strategy. Only "UncSampling" and '
                             '"RandomSampling are implemented! \n '
                             'Feel free to add other options. ')

        for n in query_indx:
            if self.test_metadata['id'].values[n] not in self.queryable_ids:
                raise ValueError('Chosen object is not available for query!')

        return query_indx

    def update_samples(self, query_indx: list, loop: int, epoch=0, screen=False):
        """Add the queried obj(s) to training and remove them from test.

        Update properties: train_headers, train_features, train_labels,
        test_labels, test_headers and test_features.

        Parameters
        ----------
        query_indx: list
            List of indexes identifying objects to be moved.
        loop: int
            Store number of loop when this query was made.
        epoch: int (optional)
            Initial epoch since survey started. Default is 0.
        screen: bool (optional)
            Print debug auxiliary information on screen. Default is False.
        """
   
        all_queries = []

        data_copy = self.test_metadata.copy()
        query_ids = [data_copy['id'].values[item] for item in query_indx]

        # check if there are repeated ids
        for name in self.train_metadata['id'].values:
            if name in self.test_metadata['id'].values:
                raise ValueError('Before update! Object ', name, 
                                 ' found in test and training sample!')
                
        # update queryable ids
        query_flag = np.array([item not in query_indx 
                               for item in range(self.queryable_ids.shape[0])])
        
        self.queryable_ids = self.queryable_ids[query_flag]

        while len(query_indx) > 0:

            obj = query_indx[0]

            # add object to the query sample
            query_header = self.test_metadata.values[obj]
            query_features = self.test_features[obj]
            line = [epoch]
            for item in query_header:
                line.append(item)
            for item1 in query_features:
                line.append(item1)

            # add object to the training sample
            new_header = pd.DataFrame([query_header], columns=self.metadata_names)
            self.train_metadata = pd.concat([self.train_metadata, new_header], axis=0,
                                            ignore_index=True)
            self.train_features = np.append(self.train_features,
                                            np.array([self.test_features[obj]]),
                                            axis=0)
            self.train_labels = np.append(self.train_labels,
                                          np.array([self.test_labels[obj]]),
                                          axis=0)

            # remove queried object from test sample
            self.test_metadata = self.test_metadata.drop(self.test_metadata.index[obj])
            self.test_labels = np.delete(self.test_labels, obj, axis=0)
            self.test_features = np.delete(self.test_features, obj, axis=0)

            # update queried samples
            all_queries.append(line)

            # update ids order
            query_indx.remove(obj)

            new_query_indx = []

            for item in query_indx:
                if item < obj:
                    new_query_indx.append(item)
                else:
                    new_query_indx.append(item - 1)

            query_indx = new_query_indx
            
            if screen: 
                print('  query_indx: ', query_indx)
        
        # update query sample
        self.queried_sample.append(all_queries)

        if screen:
            print('query_ids: ', query_ids)
            print('queried sample: ', self.queried_sample[-1][-1][1])
       
        for name in query_ids:
            if name in self.test_metadata['id'].values:
                raise ValueError('Queried object ', name, ' is still in test sample!')

            if name not in self.train_metadata['id'].values:
                raise ValueError('Queried object ', name, ' not in training!')

        # check if there are repeated ids
        for name in self.train_metadata['id'].values:
            if name in self.test_metadata['id'].values:
                raise ValueError('After update! Object ', name, ' found in test and training samples!')

    def save_metrics(self, loop: int, output_metrics_file: str, epoch: int, batch=1):
        """Save current metrics to file.

        If loop == 0 the 'output_metrics_file' will be created or overwritten.
        Otherwise results will be added to an existing 'output_metrics file'.

        Parameters
        ----------
        loop: int
            Number of learning loops finished at this stage.
        output_metrics_file: str
            Full path to file to store metrics results.
        batch: int
            Number of queries in each loop.
        epoch: int
            Days since the beginning of the survey.
        """

        # add header to metrics file
        if not os.path.exists(output_metrics_file) or loop == 0:
            with open(output_metrics_file, 'w') as metrics:
                metrics.write('loop ')
                for name in self.metrics_list_names:
                    metrics.write(name + ' ')
                for j in range(batch):
                    metrics.write('query_id' + str(j + 1) + ' ')
                metrics.write('\n')

        # write to file
        with open(output_metrics_file, 'a') as metrics:
            metrics.write(str(epoch) + ' ')
            for value in self.metrics_list_values:
                metrics.write(str(value) + ' ')
            for j in range(batch):
                metrics.write(str(self.queried_sample[loop][j][1]) + ' ')
            metrics.write('\n')

    def save_queried_sample(self, queried_sample_file: str, loop: int,
                            full_sample=False, batch=1):
        """Save queried sample to file.

        Parameters
        ----------
        queried_sample_file: str
            Complete path to output file.
        loop: int
            Number of learning loops finished at this stage.
        full_sample: bool (optional)
            If true, write down a complete queried sample stored in
            property 'queried_sample'. Otherwise append 1 line per loop to
            'queried_sample_file'. Default is False.
        """

        if full_sample:
            full_header = self.metadata_names + self.features_names
            query_sample = pd.DataFrame(self.queried_sample, columns=full_header)
            query_sample.to_csv(queried_sample_file, sep=' ', index=False)

        elif isinstance(loop, int):
            if not os.path.exists(queried_sample_file) or loop == 0:
                # add header to query sample file
                full_header = self.metadata_names + self.features_names
                with open(queried_sample_file, 'w') as query:
                    query.write('day ')
                    for item in full_header:
                        query.write(item + ' ')
                    query.write('\n')

            # save query sample to file
            with open(queried_sample_file, 'a') as query:
                for batch in range(batch):
                    for elem in self.queried_sample[loop][batch]:
                        query.write(str(elem) + ' ')
                    query.write('\n')


def main():
    return None


if __name__ == '__main__':
    main()
