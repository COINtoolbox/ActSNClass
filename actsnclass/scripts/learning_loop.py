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

__all__ = ['learn_loop']

from actsnclass import DataBase

import argparse


def learn_loop(nquery: int, strategy: str, path_to_features: str,
               output_diag_file: str, output_queried_file: str,
               features_method='Bazin', classifier='RandomForest',
               training='original', batch=1):
    """Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    nquery: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    path_to_features: str
        Complete path to input features file.
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

    # initiate object
    data = DataBase()

    # load Bazin features
    data.load_features(path_to_features, method=features_method)

    # separate training and test samples
    data.build_samples(initial_training=training)

    for loop in range(nquery):

        print('Processing... ', loop)

        # classify
        data.classify(method=classifier)

        # calculate metrics
        data.evaluate_classification()

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch)

        # update training and test samples
        data.update_samples(indx)

        # save diagnostics for current state
        data.save_metrics(loop=loop, output_metrics_file=output_diag_file)

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop, full_sample=False)


def main(args):
    path_to_data = args.input
    diag_file = args.diagnostics
    queried_sample_file = args.queried

    features = args.method
    classifier = args.classifier
    strategy = args.strategy
    nquery = args.nquery
    initial_training = args.training

    learn_loop(nquery=args.nquery, features_method=args.method,
               classifier=args.classifier,
               strategy=args.strategy, path_to_features=args.input,
               output_diag_file=args.diagnostics,
               output_queried_file=args.queried,
               training=args.training, batch=args.batch)


if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='actsnclass - '
                                                 'Learn loop module')
    parser.add_argument('-b', '--batch', dest='batch', required=True,
                        help='Number of samples to query in each loop.')
    parser.add_argument('-c', '--classifier', dest='classifier',
                        help='Choice of machine learning classification.'
                             'Only "RandomForest" is implemented.'
                             'algorithm.', required=False, default='RandomForest')
    parser.add_argument('-d', '--diagnostics', dest='diagnostics',
                        help='Path to output metrics file.', required=True)
    parser.add_argument('-i', '--input', dest='input',
                        help='Path to features file.', required=True)
    parser.add_argument('-m', '--method', dest='method',
                        help='Feature extraction method. '
                             'Only "Bazin" is implemented.', required=False,
                        default='Bazin')
    parser.add_argument('-n', '--nquery', dest='nquery', required=True,
                        help='Number of query loops to run.')
    parser.add_argument('-q', '--queried', dest='queried',
                        help='Path to output queried sample file.', required=True)
    parser.add_argument('-s', '--strategy', dest='strategy',
                        help='Active Learning strategy. Options are '
                             '"UncSampling" and "RanomSampling".',
                        required=True)
    parser.add_argument('-t', '--training', dest='training', require=True,
                        help='Initial training sample. Options are '
                             '"original" or integer.')

    args = parser.parse_args()

    main()
