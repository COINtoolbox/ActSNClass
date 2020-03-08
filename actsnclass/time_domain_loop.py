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
    data.build_samples(initial_training='original')

    return data


def time_domain_loop(days: list,  output_metrics_file: str,
                     output_queried_file: str,
                     path_to_features_dir: str, strategy: str,
                     fname_pattern: list,
                     batch=1, canonical = False,  classifier='RandomForest',
                     continue=False, features_method='Bazin', path_to_canonical="",
                     path_to_full_lc_features="", queryable=True,
                     query_thre=1.0, screen=True, survey='PLAsTiCC', training='original'):
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
    fname_pattern: str
        List of strings. Set the pattern for filename, except day of 
        survey. If file name is 'day_1_vx.dat' -> ['day_', '_vx.dat']
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    canonical: bool (optional)
        If True, restrict the search to the canonical sample.
    continue: bool (optional)
        If True, read the initial states of previous runs from file.
        Default is False.
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
    query_thre: float (optional)
        Percentile threshold for query. Default is 1. 
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    survey: str (optional)
        Name of survey to be analyzed. Accepts 'DES' or 'LSST'.
        Default is DES.
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
    path_to_features = path_to_features_dir + fname_pattern[0]  + \
                       str(int(days[0])) + fname_pattern[1]
    data.load_features(path_to_features, method=features_method,
                       screen=screen, survey=survey)

    # change training
    if training == 'original':
        data.build_samples(initial_training='original', screen=screen)
        full_lc_features = get_original_training(path_to_features=path_to_full_lc_features)
        data.train_metadata = full_lc_features.train_metadata
        data.train_labels = full_lc_features.train_labels
        data.train_features = full_lc_features.train_features

    else:
        data.build_samples(initial_training=int(training), screen=screen, continue=continue)

    # get list of canonical ids
    if canonical:
        canonical = DataBase()
        canonical.load_features(path_to_file=path_to_canonical)
        data.queryable_ids = canonical.queryable_ids

    for night in range(int(days[0]), int(days[-1]) - 1):

        if screen:
            print('Processing night: ', night)

        # cont loop
        loop = night - int(days[0])

        # classify
        data.classify(method=classifier)

        # calculate metrics
        data.evaluate_classification()

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch, queryable=queryable,
                               query_thre=query_thre)

        # update training and test samples
        data.update_samples(indx, loop=loop)

        # save metrics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_metrics_file,
                          batch=batch, epoch=night)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop,
                                 full_sample=False)

        # load features for next day
        path_to_features2 = path_to_features_dir + fname_pattern[0] + \
                            str(night + 1) + fname_pattern[1]

        data_tomorrow = DataBase()
        data_tomorrow.load_features(path_to_features2, method=features_method,
                                    screen=screen, survey=survey)

        # identify objects in the new day which must be in training
        train_flag = np.array([item in data.train_metadata['id'].values 
                              for item in data_tomorrow.metadata['id'].values])
   
        # use new data  
        data.train_metadata = data_tomorrow.metadata[train_flag]
        data.train_features = data_tomorrow.features.values[train_flag]
        data.test_metadata = data_tomorrow.metadata[~train_flag]
        data.test_features = data_tomorrow.features.values[~train_flag]

        # new labels
        data.train_labels = np.array([int(item  == 'Ia') for item in 
                                     data.train_metadata['type'].values])
        data.test_labels = np.array([int(item == 'Ia') for item in 
                                    data.test_metadata['type'].values])

        if strategy == 'canonical':
            data.queryable_ids = canonical.queryable_ids

        if  queryable:
            queryable_flag = data_tomorrow.metadata['queryable'].values
            queryable_test_flag = np.logical_and(~train_flag, queryable_flag)
            data.queryable_ids = data_tomorrow.metadata['id'].values[queryable_test_flag]
        else:
            data.queryable_ids = data_tomorrow.metadata['id'].values[~train_flag]

        if screen:
            print('Training set size: ', data.train_metadata.shape[0])
            print('Test set size: ', data.test_metadata.shape[0])
            print('Queryable set size: ', len(data.queryable_ids))


def main():
    return None


if __name__ == '__main__':
    main()
