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
from actsnclass.metrics import efficiency, purity, fom, accuracy


__all__ = ['DataBase']


class DataBase:
    def __init__(self):
        self.features = None
        self.header = None
        self.data = None
        self.train_features = None
        self.train_labels = None
        self.train_header = None
        self.test_features = None
        self.test_labels = None
        self.test_header = None
        self.classprob = None
        self.header_names = None
        self.feature_names = None
        self.predicted_class = None
        self.eff = None
        self.purity = None
        self.fom = None
        self.accuracy = None
        self.query_sample = []
        self.metrics_list_names = ['acc', 'eff', 'pur', 'fom']
        self.metrics_list_values = None
        self.queryable_ids = None

    def load_bazin_features(self, path_to_bazin_file):
        """Load Bazin features from file.

        Populate properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_bazin_file: str
            Complete path to Bazin features file.
        """

        # read matrix with Bazin features
        self.data = pd.read_csv(path_to_bazin_file, sep=' ', index_col=False)

        # list of features to use
        self.feature_names = ['gA', 'gB', 'gt0', 'gtfall', 'gtrise', 'rA',
                              'rB', 'rt0', 'rtfall', 'rtrise', 'iA', 'iB',
                              'it0', 'itfall', 'itrise', 'zA', 'zB', 'zt0',
                              'ztfall', 'ztrise']

        self.header_names = ['id', 'redshift', 'type', 'code', 'sample']

        self.features = self.data[self.feature_names]
        self.header = self.data[self.header_names]

        print('Loaded ', self.header.shape[0], ' samples!')

    def load_features(self, path_to_file, method='Bazin'):
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
        """

        if method == 'Bazin':
            self.load_bazin_features(path_to_file)
        else:
            raise ValueError('Only Bazin features are implemented! '
                             '\n Feel free to add other options.')

    def build_samples(self, initial_training='original', nclass=2):
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
                    ensuring that at least half are SN Ia
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        """

        # separate original training and test samples
        if initial_training == 'original':
            train_flag = self.header['sample'] == 'train'
            train_data = self.features[train_flag]
            self.train_features = train_data.values
            self.train_header = self.header[train_flag]

            test_flag = np.logical_or(self.header['sample'] == 'test',
                                      self.header['sample'] == 'queryable')
            test_data = self.features[test_flag]
            self.test_features = test_data.values
            self.test_header = self.header[test_flag]

            if 'queryable' in self.header['sample'].values:
                queryable_flag = self.header['sample'].values == 'queryable'
                self.queryable_ids = self.header[queryable_flag]['id'].values
            else:
                self.queryable_ids = self.header['id'].values

            if nclass == 2:
                train_ia_flag = self.train_header['type'] == 'Ia'
                self.train_labels = np.array([int(item) for item in train_ia_flag])

                test_ia_flag = self.test_header['type'] == 'Ia'
                self.test_labels = np.array([int(item) for item in test_ia_flag])
            else:
                raise ValueError("Only 'Ia x non-Ia' are implemented! "
                                 "\n Feel free to add other options.")

        elif isinstance(initial_training, int):

            # initialize the temporary label holder
            train_indexes = np.array([1])
            temp_labels = np.array([self.header['type'].values[train_indexes]])

            # make sure half are Ias and half are non-Ias
            while sum(temp_labels == 'Ia') != initial_training // 2 + 1:
                train_indexes = np.random.randint(low=0,
                                                  high=self.header.shape[0],
                                                  size=initial_training)
                temp_labels = self.header['type'].values[train_indexes]

            # set training
            train_ia_flag = self.header['type'].values[train_indexes] == 'Ia'
            self.train_labels = np.array([int(item) for item in train_ia_flag])
            self.train_features = self.features.values[train_indexes]
            self.train_header = pd.DataFrame(self.header.values[train_indexes],
                                             columns=self.header_names)

            # set test set as all objs apart from those in training
            test_indexes = np.array([i for i in range(self.header.shape[0])
                                     if i not in train_indexes])
            test_ia_flag = self.header['type'].values[test_indexes] == 'Ia'
            self.test_labels = np.array([int(item) for item in test_ia_flag])
            self.test_features = self.features.values[test_indexes]
            self.test_header = pd.DataFrame(self.header.values[test_indexes],
                                            columns=self.header_names)

        else:
            raise ValueError('"Initial training" should be '
                             '"original" or integer.')

        print('Training set size: ', self.train_header.shape[0])
        print('Test set size: ', self.test_header.shape[0])

    def classify(self, method='RandomForest'):
        """Apply a machine learning classifier.

        Populate properties: predicted_class and class_prob

        Parameters
        ----------
        method: str (optional)
            Chosen classifier.
            The current implementation on accepts `RandomForest`.
        """

        if method == 'RandomForest':
            results = random_forest(self.train_features, self.train_labels,
                                    self.test_features)
            self.predicted_class = results[0]
            self.classprob = results[1]

        else:
            raise ValueError('Only RandomForest classifier is implemented!'
                             '\n Feel free to add other options.')

    def evaluate_classification(self):
        """Evaluate results from classification.

        Populate properties: eff, purity, fom, accuracy and
        metrics_list_values.
        """

        self.eff = efficiency(self.predicted_class, self.test_labels)
        self.purity = purity(self.predicted_class, self.test_labels)
        self.fom = fom(self.predicted_class, self.test_labels)
        self.accuracy = accuracy(self.predicted_class, self.test_labels)

        self.metrics_list_values = [self.accuracy, self.eff, self.purity,
                                    self.fom]

    def make_query(self, strategy='UncSampling', batch=1):
        """Identify new object to be added to the training sample.

        Parameters
        ----------
        strategy: str (optional)
            Strategy used to choose the most informative object.
            Current implementation accepts 'UncSampling' and
            'RandomSampling'. Default is `UncSampling`.

        batch: int (optional)
            Number of objects to be chosen in each batch query.
            Default is 1.

        Returns
        -------
        query_indx: list
            List of indexes identifying the objects to be queried in decreasing
            order of importance.
            If strategy=='RandomSampling' the order is irrelevant.
        """

        if strategy == 'UncSampling':
            query_indx = uncertainty_sampling(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_header['id'].values,
                                              batch=batch)
            return query_indx

        elif strategy == 'RandomSampling':
            query_indx = random_sampling(queryable_ids=self.queryable_ids,
                                         test_ids=self.test_header['id'].values,
                                         batch=batch)

            for n in query_indx:
                if self.test_header['id'].values[n] not in self.queryable_ids:
                    raise ValueError('Chosen object is not available for query!')

            return query_indx

        else:
            raise ValueError('Invalid strategy. Only "UncSampling" and '
                             '"RandomSampling are implemented! \n '
                             'Feel free to add other options. ')

    def update_samples(self, query_indx):
        """Add the queried obj(s) to training and remove them from test.

        Update properties: train_headers, train_features, train_labels,
        test_labels, test_headers and test_features.

        Parameters
        ----------
        query_indx: list
            List of indexes identifying objects to be moved.
        """

        for obj in query_indx:
            # add object to the query sample
            query_header = self.test_header.values[obj]
            query_features = self.test_features[obj]
            self.query_sample.append(np.append(query_header, query_features, axis=0))

            # add object to the training sample
            new_header = pd.DataFrame([query_header], columns=self.header_names)
            self.train_header = pd.concat([self.train_header, new_header], axis=0,
                                          ignore_index=True)
            self.train_features = np.append(self.train_features,
                                            np.array([self.test_features[obj]]),
                                            axis=0)
            self.train_labels = np.append(self.train_labels,
                                          np.array([self.test_labels[obj]]),
                                          axis=0)

            # remove queried object from test sample
            self.test_header = self.test_header.drop(self.test_header.index[obj])
            self.test_labels = np.delete(self.test_labels, obj, axis=0)
            self.test_features = np.delete(self.test_features, obj, axis=0)

    def save_metrics(self, loop, output_metrics_file):
        """Save current metrics to file.

        If loop == 0 a the output_metrics_file will be created or overwritten.
        Otherwise results will be added to an existing output_metrics file.

        Parameters
        ----------
        loop: int
            Number of learning loops at this stage.
        output_metrics_file: str
            Full path to file to store metrics results.
        """

        # add header to metrics file
        if not os.path.exists(output_metrics_file) or day == 0:
            with open(output_metrics_file, 'w') as metrics:
                metrics.write('day ')
                for name in self.metrics_list_names:
                    metrics.write(name + ' ')
                metrics.write('query_id' + '\n')

        # write to file
        with open(output_metrics_file, 'a') as metrics:
            metrics.write(str(day) + ' ')
            for value in self.metrics_list_values:
                metrics.write(str(value) + ' ')
            metrics.write(str(self.query_sample[day][0]) + '\n')

    def save_query_sample(self, query_sample_file, day=0, full_sample=False):
        """Save query sample to file"""

        if full_sample:
            full_header = self.header_names + self.feature_names
            query_sample = pd.DataFrame(self.query_sample, columns=full_header)
            query_sample.to_csv(query_sample_file, sep=' ', index=False)

        elif isinstance(day, int):
            if not os.path.exists(query_sample_file) or day == 0:
                # add header to query sample file
                full_header = self.header_names + self.feature_names
                with open(query_sample_file, 'w') as query:
                    for item in full_header:
                        query.write(item + ' ')
                    query.write('\n')

            else:
                # save query sample to file
                with open(query_sample_file, 'a') as query:
                    for elem in self.query_sample[day]:
                        query.write(str(elem) + ' ')
                    query.write('\n')


def main():
    return None


if __name__ == '__main__':
    main()
