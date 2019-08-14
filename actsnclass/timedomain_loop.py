# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 13 August 2019
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

__all__ = ['get_queryable_tomorrow', 'set_samples_config',
           'timedomain_loop']

import numpy as np

from actsnclass import DataBase


def get_queryable_tomorrow(info_today: DataBase, info_tomorrow: DataBase):
    """Update queryable property for today with the sample info for tomorrow.

    Parameters
    ----------
    info_today: DataBase
        DataBase with the status of observations today.
    info_tomorrow
        DataBase with the status of observations tomorrow.

    Returns
    -------
    info_today
        Updated DataBase where the queryable_ids property indicates
        which objects can be queried tomorrow.
    """

    # for each element in today's test sample
    for i in range(info_today.test_metadata.shape[0]):

        # get id
        snid = info_today.test_metadata['id'].values[i]

        # check if it is queryable tomorrow
        query_tomorrow = np.isin(snid, info_tomorrow.queryable_ids)

        # change queryable status today
        if query_tomorrow and not np.isin(snid, info_today.queryable_ids):
            info_today.queryable_ids = \
                np.append(info_today.queryable_ids, np.array([snid]), axis=0)

        elif not query_tomorrow and np.isin(snid, info_today.queryable_ids):
            loc = list(info_today.queryable_ids).index(snid)
            info_today.queryable_ids = np.delete(info_today.queryable_ids,
                                                 loc, axis=0)

    return info_today


def set_samples_config(train_option, today: DataBase,
                       path_to_full_features: str, features_method='Bazin'):
    """Configure training and queryable samples according to user choice.

    Parameters
    ----------
    train_option: int or str
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia.
    today: actsnclass.DataBase
        DataBase objet holding data for today.
    path_to_full_features: str
        Complete path to full light curve extracted features.
    features_method: str (optional)
        Feature extraction method. The current implementation only
        accepts method=='Bazin'.

    Returns
    -------
    today: actsnclass.DataBase
        Updated information about the training and queryable samples
        for today using queries available for tomorrow.
    full_train_ids: np.array
        All ids in the original training sample.
    """
    if train_option == 'original':
        # read_full light curve data
        full_data = DataBase()
        full_data.load_features(path_to_file=path_to_full_features,
                                method=features_method, screen=False)
        full_data.build_samples(initial_training='original')

        # get ids of objs in the original training sample
        full_train_ids = full_data.train_metadata['id'].values

        # identify queryable objects which belong to the original training
        queryable_flag = \
            np.array([False if np.isin(item, full_train_ids)
                     else True for item in today.queryable_ids])

        today.queryable_ids = today.queryable_ids[queryable_flag]

        # set the training sample to be the original one
        today.train_labels = full_data.train_labels
        today.train_features = full_data.train_features
        today.train_metadata = full_data.train_metadata

    else:
        # separate training and test samples
        today.build_samples(initial_training=train_option)
        full_train_ids = None

    return today, full_train_ids


def time_domain_loop(training, days: list, output_diag_file: str,
                     output_queried_file: str,
                     path_to_data_dir: str, strategy: str,
                     batch=1, classifier='RandomForest',
                     features_method='Bazin', nclass=2,
                     path_to_full_features='', screen=False):
    """Perform the active learning loop in time domain.

    Trains the machine learning model in day X and queries in day X+1.
    Features files for each individual day should follow the standards
    of the `actsnclass.build_time_domain` module.

    Parameters
    ----------
    training: str or int
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
    days: list
        List of 2 elements, corresponding to first and last days since
        the beginning of the survey to simulate. [first_day, last_day]
    output_diag_file: str
        Full path to output file to store diagnostics of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    path_to_data_dir: str
        Path to directory containing time domain data.
        The files for storing data from each day must have the format
        'day_X.dat' where X corresponds to days since the beginning of the survey.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    nclass: int (optional)
        Number of classes in the training sampel.
        Default is 2.
    path_to_full_features: str (optional)
        Path to full light curve features files.
        This is only used if training == 'original'.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    """

    # name of features file for today
    features_today = path_to_data_dir + 'day_' + str(days[0]) + '.dat'

    # load features
    original_data = DataBase()
    original_data.load_features(features_today, method=features_method,
                                screen=False)
    original_data.build_samples(initial_training='original', nclass=nclass)

    # get full LC original training
    info_today, fulldata_train_ids = \
        set_samples_config(training, original_data,
                           path_to_full_features=path_to_full_features,
                           features_method=features_method)

    # cont loop
    loop = 0

    for epoch in range(days[0], days[-1] - 1):

        if screen:
            print('Processing... ', epoch)

        # classify
        info_today.classify(method=classifier)

        # calculate metrics
        info_today.evaluate_classification()

        # name of features file for tomorrow
        features_tomorrow = path_to_data_dir + 'day_' + str(epoch + 1) + '.dat'

        # read data for  next day
        info_tomorrow = DataBase()
        info_tomorrow.load_features(features_tomorrow, method=features_method,
                                    screen=False)
        info_tomorrow.build_samples(initial_training='original')

        # update 'sample' property with values for tomorrow
        update_today = get_queryable_tomorrow(info_today, info_tomorrow)

        # make query
        indx = update_today.make_query(strategy=strategy, batch=batch)

        # update training and test samples
        update_today.update_samples(indx, loop=loop, epoch=epoch)

        # save diagnostics for current state
        update_today.save_metrics(loop=loop, output_metrics_file=output_diag_file,
                                  batch=batch, epoch=epoch)

        # save query sample to file
        update_today.save_queried_sample(output_queried_file, loop=loop,
                                         full_sample=False)

        # update loop count
        loop = loop + 1

        # check if samples in test today not in original training
        if training == 'original':
            test_surv_flag = \
                np.array([False if item in fulldata_train_ids else True
                         for item in update_today.test_metadata['id'].values])

            meta_data2 = update_today.test_metadata[test_surv_flag]
            label2 = update_today.test_labels[test_surv_flag]
            features2 = update_today.test_features[test_surv_flag]

            update_today.test_features = features2
            update_today.test_labels = label2
            update_today.test_metadata = meta_data2

        # update information from today
        del info_tomorrow
        info_today = update_today


def main():
    path_to_data_dir = 'results/time_domain/data/'
    features_method = 'Bazin'
    classifier = 'RandomForest'
    training = 'original'
    batch = 1
    strategy = 'UncSampling'
    days = [19, 182]
    output_diag_file = 'results/time_domain/diag_rand_original.dat'
    output_queried_file = 'results/time_domain/queried_rand_original.dat'
    output_plot = 'plots/time_domain_original.png'
    path_to_full_features = 'results/Bazin.dat'
    screen = True

    time_domain_loop(days=days, strategy=strategy,
                     path_to_data_dir=path_to_data_dir,
                     output_diag_file=output_diag_file,
                     output_queried_file=output_queried_file,
                     features_method=features_method, classifier=classifier,
                     training=training, batch=batch,
                     path_to_full_features=path_to_full_features, screen=screen)

    from actsnclass.plot_results import Canvas

    cv = Canvas()
    cv.load_diagnostics([output_diag_file], [strategy])
    cv.set_plot_dimensions()
    cv.plot_diagnostics(output_plot, [strategy])

    return None


if __name__ == '__main__':
    main()
