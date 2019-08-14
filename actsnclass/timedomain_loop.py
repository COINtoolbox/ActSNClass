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

__all__ = ['get_sample_tomorrow', 'timedomain_loop']

import numpy as np

from actsnclass import DataBase


def get_sample_tomorrow(info_today: DataBase, info_tomorrow: DataBase):
    """
    Update the property today's data set with the sample info for tomorrow.

    Parameters
    ----------
    info_today: DataBase
        DataBase with the status of observations today.
    info_tomorrow
        DataBase with the status of observations tomorrow.

    Returns
    -------
    info_today
        Updated DataBase where the 'sample' property indicates
        which objects can be queried tomorrow.
    """

    # isolate ids information
    today_ids = info_today.metadata['id'].values
    tomorow_ids = info_tomorrow.metadata['id'].values

    # for each element in tomorrow's sample
    for i in range(info_tomorrow.metadata.shape[0]):

        # get id
        snid = info_tomorrow.metadata['id'].values[i]

        # get sample
        sample_tomorrow = info_tomorrow.metadata.at[i, 'sample']

        # find this object in the data from today
        if snid in today_ids:
            loc_today = list(info_today.metadata['id'].values).index(snid)

            # update the sample with values from tomorrow
            info_today.metadata.at[loc_today, 'sample'] = sample_tomorrow

    for k in range(len(today_ids)):
        if np.isin(today_ids[k], tomorow_ids):
            info_today.metadata.at[k, 'sample'] = 'test'

        return info_today


def timedomain_loop(days: list, strategy: str, path_to_data_dir: str,
                    output_diag_file: str, output_queried_file: str,
                    features_method='Bazin', classifier='RandomForest',
                    training=5, batch=1):
    """Perform the active learning loop in time domain.

    Trains the machine learning model in day X and queries in day X+1.
    Features files for each individual day should follow the standards
    of the `actsnclass.build_time_domain` module.

    Parameters
    ----------
    days: list
        List of 2 elements, corresponding to first and last days since
        the beginning of the survey to simulate. [first_day, last_day]
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    path_to_data_dir: str
        Path to directory containing time domain data.
        The files for storing data from each day must have the format
        'day_X.dat' where X corresponds to days since the beginning of the survey.
    output_diag_file: str
        Full path to output file to store diagnostics of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    features_method: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    training: str or int (optional)
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    """

    # name of features file for today
    features_today = path_to_data_dir + 'day_' + str(days[0]) + '.dat'

    # load features
    info_today = DataBase()
    info_today.load_features(features_today, method=features_method)

    # separate training and test samples
    info_today.build_samples(initial_training=training)

    # cont loop
    loop = 0

    for epoch in range(days[0], days[1] - 1):
        print('Processing... ', epoch)

        # classify
        info_today.classify(method=classifier)

        # calculate metrics
        info_today.evaluate_classification()

        # name of features file for tomorrow
        features_tomorrow = path_to_data_dir + \
                            'day_' + str(epoch + 1) + '.dat'

        # read data for next day
        info_tomorrow = DataBase()
        info_tomorrow.load_features(features_tomorrow, method=features_method)

        # update 'sample' property with values for tomorrow
        update_today = get_sample_tomorrow(info_today, info_tomorrow)

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

        # update information from today
        del info_tomorrow
        info_today = update_today


def main():
    path_to_data_dir = 'results/time_domain/data/'
    features_method = 'Bazin'
    classifier = 'RandomForest'
    training = 5
    batch = 1
    strategy = 'RandomSampling'
    days = [19, 182]
    output_diag_file = 'results/time_domain/diag_rand.dat'
    output_queried_file = 'results/time_domain/queried_rand.dat'
    output_plot = 'plots/time_domain.png'

    timedomain_loop(days=days, strategy=strategy,
                    path_to_data_dir=path_to_data_dir,
                    output_diag_file=output_diag_file,
                    output_queried_file=output_queried_file,
                    features_method=features_method, classifier=classifier,
                    training=training, batch=batch)

    from actsnclass.plot_results import Canvas
    cv=Canvas()
    cv.load_diagnostics([output_diag_file], [strategy])
    cv.set_plot_dimensions()
    cv.plot_diagnostics(output_plot, [strategy])

    return None


if __name__ == '__main__':
    main()
