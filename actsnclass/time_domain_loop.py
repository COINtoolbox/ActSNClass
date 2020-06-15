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

__all__ = ['time_domain_loop', 'get_original_training']

import numpy as np
import pandas as pd

from actsnclass import DataBase


def get_original_training(path_to_features, method='Bazin', screen=False):
    """Read original full light curve training sample

    Parameters
    ----------
    path_to_features: str
        Complete path to file holding full light curve features.
    method: str (optional)
        Feature extraction method. Only option implemented is "Bazin".
    screen: bool (optional)
        If true, show on screen comments on the dimensions of some key elements.

    Returns
    -------
    snactclass.DataBase
        Information about the original full light curve analys.
    """

    data = DataBase()
    data.load_features(path_to_features, method=method, screen=screen)
    data.build_samples(initial_training='original', screen=screen)

    return data


def time_domain_loop(days: list,  output_metrics_file: str,
                     output_queried_file: str,
                     path_to_features_dir: str, strategy: str,
                     batch=1, canonical = False,  classifier='RandomForest',
                     features_method='Bazin', path_to_canonical="",
                     path_to_full_lc_features="", queryable=True,
                     screen=True, training='original'):
    """Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    days: list
        List of 2 elements. First and last day of observations since the
        beginning of the survey.
    output_metrics_file: str
        Full path to output file to store metrics for each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    path_to_features_dir: str
        Complete path to directory holding features files for all days.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    canonical: bool (optional)
        If True, restrict the search to the canonical sample.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    path_to_canonical: str (optional)
        Path to canonical sample features files.
        It is only used if "strategy==canonical".
    path_to_full_lc_features: str (optional)
        Path to full light curve features file.
        Only used if training is a number.
    queryable: bool (optional)
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    """

    # initiate object
    data = DataBase()

    # load features for the first day
    path_to_features = path_to_features_dir + 'day_' + str(int(days[0])) + '.dat'
    data.load_features(path_to_features, method=features_method,
                       screen=screen)

    # change training
    if training == 'original':
        data.build_samples(initial_training=0, screen=screen, queryable=queryable)
        full_lc_features = get_original_training(path_to_features=path_to_full_lc_features,
                                                 screen=screen)
        ini_train_ids = full_lc_features.train_metadata['id'].values
        data.train_metadata = full_lc_features.train_metadata
        data.train_labels = full_lc_features.train_labels
        data.train_features = full_lc_features.train_features

        # remove repeated ids
        test_flag = np.array([item not in ini_train_ids
                              for item in data.test_metadata['id'].values])
        data.test_metadata = data.test_metadata[test_flag]
        data.test_labels = data.test_labels[test_flag]
        data.test_features = data.test_features[test_flag]

    else:
        data.build_samples(initial_training=int(training), screen=screen,
                           queryable=queryable)
        ini_train_ids = []

    # get list of canonical ids
    if canonical:
        canonical = DataBase()
        canonical.load_features(path_to_file=path_to_canonical)
        data.queryable_ids = canonical.queryable_ids


    for night in range(int(days[0]), int(days[-1]) - 1):

        if screen:
            print('Processing night: ', night)
            print('    ... train: ', data.train_metadata.shape[0])
            print('    ... test: ', data.test_metadata.shape[0])
            print('    ... queryable_ids: ', data.queryable_ids.shape[0])

        # cont loop
        loop = night - int(days[0])

        # classify
        data.classify(method=classifier, screen=screen)

        # calculate metrics
        data.evaluate_classification()

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch, screen=screen)

        # update training and test samples
        data.update_samples(indx, loop=loop, screen=screen)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=night)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop,
                                 full_sample=False)

        # load features for next day
        path_to_features2 = path_to_features_dir + 'day_' + str(night + 1) + '.dat'

        data_tomorrow = DataBase()
        data_tomorrow.load_features(path_to_features2, method=features_method,
                                    screen=False)
        data_tomorrow.build_samples(initial_training=0, screen=screen)

        # remove training samples from new test
        for obj in data.train_metadata['id'].values:
            if obj in data_tomorrow.test_metadata['id'].values:

                indx_tomorrow = list(data_tomorrow.test_metadata['id'].values).index(obj)

                if obj not in ini_train_ids:
                    # remove old features from training
                    indx_today = list(data.train_metadata['id'].values).index(obj)
                    data.train_metadata = data.train_metadata.drop(data.train_metadata.index[indx_today])
                    data.train_labels = np.delete(data.train_labels, indx_today, axis=0)
                    data.train_features = np.delete(data.train_features, indx_today, axis=0)
                
                    # update new features of the training with new obs                
                    flag = np.arange(0, data_tomorrow.test_metadata.shape[0]) == indx_tomorrow
                    data.train_metadata = pd.concat([data.train_metadata, data_tomorrow.test_metadata[flag]], axis=0,
                                                     ignore_index=True)
                    data.train_features = np.append(data.train_features,
                                                    data_tomorrow.test_features[flag], axis=0)
                    data.train_labels = np.append(data.train_labels,
                                                  data_tomorrow.test_labels[flag], axis=0)  

                # remove from new test sample
                data_tomorrow.test_metadata = data_tomorrow.test_metadata.drop(data_tomorrow.test_metadata.index[indx_tomorrow])
                data_tomorrow.test_labels = np.delete(data_tomorrow.test_labels, indx_tomorrow, axis=0)
                data_tomorrow.test_features = np.delete(data_tomorrow.test_features, indx_tomorrow, axis=0)

        # use new test data
        data.test_metadata = data_tomorrow.test_metadata
        data.test_labels = data_tomorrow.test_labels
        data.test_features = data_tomorrow.test_features

        if strategy == 'canonical':
            data.queryable_ids = canonical.queryable_ids
        else:
            data.queryable_ids = data_tomorrow.queryable_ids
        
        if  queryable:
            queryable_flag = data_tomorrow.test_metadata['queryable'].values
            data.queryable_ids = data_tomorrow.test_metadata['id'].values[queryable_flag]
        else:
            data.queryable_ids = data_tomorrow.metadata['id'].values[~train_flag]

        # check if there are repeated ids
        for name in data.train_metadata['id'].values:
            if name in data.test_metadata['id'].values:
                raise ValueError('End of time_domain_loop: ' + \
                                 'Object ', name, ' found in test and training sample!')

        if screen:
            print('\n End of time domain loop:')
            print('    ... train: ', data.train_metadata.shape[0])
            print('    ... test: ', data.test_metadata.shape[0])
            print('    ... queryable: ', data.queryable_ids.shape[0], '\n')


def main():
    return None


if __name__ == '__main__':
    main()
