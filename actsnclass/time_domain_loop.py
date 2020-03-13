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

__all__ = ['time_domain_loop']

import numpy as np

from actsnclass import DataBase

def time_domain_loop(days: list,  output_metrics_file: str,
                     output_queried_file: str,
                     path_to_features_dir: str, strategy: str,
                     fname_pattern: list,
                     batch=1, canonical = False,  classifier='RandomForest',
                     cont=False, first_loop=20, features_method='Bazin', nclass=2,
                     Ia_frac=0.5, output_fname="", path_to_canonical="",
                     path_to_full_lc_features="", path_to_train="",
                     path_to_queried="", queryable=True,
                     query_thre=1.0, save_samples=False, sep_files=False,
                     screen=True, survey='LSST', initial_training='original'):
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
        Currently 'RandomForest', 'GradientBoostedTrees',
        'KNN', 'MLP', 'SVM' and 'NB' are implemented.
        Default is 'RandomForest'.
    first_loop: int (optional)
        First day of the survey already calculated in previous runs.
        Only used if initial_training == 'previous'.
        Default is 20.
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    Ia_frac: float in [0,1] (optional)
        Fraction of Ia required in initial training sample.
        Default is 0.5.
    nclass: int (optional)
        Number of classes to consider in the classification
        Currently only nclass == 2 is implemented.
    path_to_canonical: str (optional)
        Path to canonical sample features files.
        It is only used if "strategy==canonical".
    path_to_full_lc_features: str (optional)
        Path to full light curve features file.
        Only used if training is a number.
    path_to_train: str (optional)
        Path to initial training file from previous run.
        Only used if initial_training == 'previous'.
    path_to_queried: str(optional)
        Path to queried sample from previous run.
        Only used if initial_training == 'previous'.
    queryable: bool (optional)
        If True, allow queries only on objects flagged as queryable.
        Default is True.
    query_thre: float (optional)
        Percentile threshold for query. Default is 1. 
    save_samples: bool (optional)
        If True, save training and test samples to file.
        Default is False.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    sep_files: bool (optional)
        If True, consider train and test samples separately read 
        from independent files. Default is False.
    survey: str (optional)
        Name of survey to be analyzed. Accepts 'DES' or 'LSST'.
        Default is LSST.
    initial_training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        eilf 'previous': read training and queried from previous run.
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    output_fname: str (optional)
        Complete path to output file where initial training will be stored.
        Only used if save_samples == True.
    """

    # initiate object
    data = DataBase()

    # load features for the first day
    path_to_features = path_to_features_dir + fname_pattern[0]  + \
                       str(int(days[0])) + fname_pattern[1]
    data.load_features(path_to_features, method=features_method,
                       screen=screen, survey=survey)

    # constructs training, test and queryable
    data.build_samples(initial_training=initial_training, nclass=nclass,
                      screen=screen, Ia_frac=Ia_frac,
                      queryable=queryable, save_samples=save_samples, 
                      sep_files=sep_files, survey=survey, 
                      output_fname=output_fname, path_to_train=path_to_train,
                      path_to_queried=path_to_queried, method=features_method)

    # get list of canonical ids
    if canonical:
        canonical = DataBase()
        canonical.load_features(path_to_file=path_to_canonical)
        data.queryable_ids = canonical.queryable_ids

    for night in range(int(days[0]), int(days[-1]) - 1):

        if screen:
            print('Processing night: ', night)

        # cont loop
        if initial_training == 'previous':
            loop = night - first_loop
        else:
            loop = night - int(days[0])

        if data.test_metadata.shape[0] > 0:
            # classify
            data.classify(method=classifier)

            # calculate metrics
            data.evaluate_classification()
        
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
