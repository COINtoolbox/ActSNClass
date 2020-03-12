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

__all__ = ['learn_loop', 'main']

from actsnclass.learn_loop import learn_loop

import argparse


def main(args):
    """Command line interface to run the active learning loop.

    Parameters
    ----------
    -d: str
        Full path to output file to store diagnostics of each loop.
    -i: str
        Complete path to input features file.
    -n: int
        Number of active learning loops to run.
    -q: str
        Full path to output file to store the queried sample.
    -s: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    -b: int (optional)
       Size of batch to be queried in each loop. Default is 1.
    -c: str (optional)
        Machine Learning algorithm.
        Currently 'RandomForest','GradientBoostedTrees','K-NNclassifier' and 'MLPclassifier' are implemented.
    -m: str (optional)
        Feature extraction method. Currently only 'Bazin' is implemented.
    -t: str or int (optional)
       Choice of initial training sample.
       If 'original': begin from the train sample flagged in the file
       If int: choose the required number of samples at random,
       ensuring that at least half are SN Ia.
       Default is 'original'.

    Examples
    --------

    Run directly from the command line:

    >>> run_loop.py -i <input features file> -b <batch size> -n <number of loops>
    >>>             -d <output diagnostic file> -q <output queried sample file>
    >>>             -s <learning strategy> -t <choice of initial training>

    """

    # set training sample variable
    if args.training == 'original':
        train = 'original'
    elif isinstance(int(args.training), int):
        train = int(args.training)
    else:
        raise ValueError('-t or --training option must be '
                         '"original" or integer!')

    # run active learning loop
    learn_loop(nloops=args.nquery, features_method=args.method,
               classifier=args.classifier,
               strategy=args.strategy, path_to_features=args.input,
               output_metrics_file=args.metrics,
               output_queried_file=args.queried,
               training=train, batch=args.batch)


if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='actsnclass - '
                                                 'Learn loop module')
    parser.add_argument('-b', '--batch', dest='batch', required=True,
                        help='Number of samples to query in each loop.',
                        type=int)
    parser.add_argument('-c', '--classifier', dest='classifier',
                        help='Choice of machine learning classification.'
                             ' "RandomForest", "GradientBoosetedTrees","K-NNclassifier", "MLPclassifier"'
                             ' "SVMclassifier" and "NBclassifier" are implemented.'
                             'algorithm.', required=False, default='RandomForest',
                        type=str)
    parser.add_argument('-m', '--metrics', dest='metrics',
                        help='Path to output metrics file.', required=True,
                        type=str)
    parser.add_argument('-i', '--input', dest='input',
                        help='Path to features file.', required=True, type=str)
    parser.add_argument('-mt', '--method', dest='method',
                        help='Feature extraction method. '
                             'Only "Bazin" is implemented.', required=False,
                        default='Bazin', type=str)
    parser.add_argument('-n', '--nquery', dest='nquery', required=True,
                        help='Number of query loops to run.', type=int)
    parser.add_argument('-q', '--queried', dest='queried', type=str,
                        help='Path to output queried sample file.', required=True)
    parser.add_argument('-s', '--strategy', dest='strategy',
                        help='Active Learning strategy. Options are '
                             '"UncSampling" and "RanomSampling".',
                        required=True, type=str)
    parser.add_argument('-t', '--training', dest='training', required=True,
                        help='Initial training sample. Options are '
                             '"original" or integer.', type=str)

    from_user = parser.parse_args()

    main(from_user)
