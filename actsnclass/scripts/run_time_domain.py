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

__all__ = ['main']

import argparse

from actsnclass.time_domain_loop import time_domain_loop


def main(user_choice):
    """Command line interface to the Time Domain Active Learning scenario.

    Parameters
    ----------
    -d: sequence
         List of 2 elements. First and last day of observations since the
         beginning of the survey.
    -m: str
        Full path to output file to store metrics of each loop.
    -q: str
        Full path to output file to store the queried sample.
    -f: str
        Complete path to directory holding features files for all days.
    -s: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    -b: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    -c: str (optional)
        Machine Learning algorithm.
        Currently 'RandomForest', 'GradientBoostedTrees'
        'K-NNclassifier' and 'MLPclassifier' are implemented.
    -fm: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    -sc: bool (optional)
        If True, display comment with size of samples on screen.
    -t: str or int
        Choice of initial training sample.
        If 'original': begin from the train sample flagged in the file
        If int: choose the required number of samples at random,
        ensuring that at least half are SN Ia
        Default is 'original'.
    -fl: file containing full light curve features to train on
         if -t original is chosen.

    Returns
    -------
    metric file: file
        File with metrics calculated in each iteration.
    queried file: file
        All objects queried during the search, in sequence.

    Examples
    --------
    Use directly from the command line.

    >>> run_time_domain.py -d <first day of survey> <last day of survey>
    >>>        -m <output metrics file> -q <output queried file> -f <features directory>
    >>>        -s <learning strategy> -fm <path to full light curve features >

    Be aware to check the default options as well!
    """

    # set parameters
    days = user_choice.days
    output_diag_file = user_choice.metrics
    output_query_file = user_choice.queried
    path_to_features_dir = user_choice.features_dir
    strategy = user_choice.strategy

    batch = user_choice.batch
    classifier = user_choice.classifier
    feature_method = user_choice.feat_method
    path_to_full_lc_features = user_choice.full_features
    screen = user_choice.screen
    training = user_choice.training

    # run time domain loop
    time_domain_loop(days=days, output_diag_file=output_diag_file,
                     output_queried_file=output_query_file,
                     path_to_features_dir=path_to_features_dir,
                     strategy=strategy, batch=batch, classifier=classifier,
                     features_method=feature_method,
                     path_to_full_lc_features=path_to_full_lc_features,
                     screen=screen, training=training)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='actsnclass - '
                                                 'Time Domain loop module')
    parser.add_argument('-d', '--days', dest='days', required=True,
                        help='First and last day of survey.',
                        nargs='+')
    parser.add_argument('-m', '--metrics', dest='metrics', required=True,
                        help='File to store metrics results.')
    parser.add_argument('-q', '--queried', dest='queried',
                        help='File to store queried sample.',
                        required=True)
    parser.add_argument('-f', '--feat-dir', dest='features_dir',
                        required=True, help='Path to directory '
                                            'of feature files for each day.')
    parser.add_argument('-s', '--strategy', dest='strategy',
                        required=True,
                        help='Active Learning strategies. Possibilities are'
                             '"RandomSampling" or UncSampling.')
    parser.add_argument('-b', '--batch', dest='batch', required=False,
                        help='Number of queries in each iteration.',
                        default=1)
    parser.add_argument('-c', '--classifier', dest='classifier',
                        required=False, default='RandomForest',
                        help='Classifier. Currently only accepts '
                             '"RandomForest", "GradientBoostedTrees",'
                             ' "K-NNclassifier" and "MLPclassifier".')
    parser.add_argument('-fm', '--feature-method', dest='feat_method',
                        required=False, default='Bazin',
                        help='Feature extraction method. Currently only accepts '
                             '"Bazin".')
    parser.add_argument('-fl', '--full-light-curve',
                        dest='full_features', required=False,
                        default=' ', help='Path to full light curve features.'
                                          'Only used if '
                                          '"training==original".')
    parser.add_argument('-sc', '--screen', dest='screen', required=False,
                        default=True, type=str2bool,
                        help='If True, display size info on training and '
                             'test sample on screen.')
    parser.add_argument('-t', '--training', dest='training', required=False,
                        default='original', help='Choice of initial training'
                                                 'sample. It can be "original"'
                                                 'or an integer.')

    from_user = parser.parse_args()

    main(from_user)
