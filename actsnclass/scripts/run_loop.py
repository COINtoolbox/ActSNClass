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

from actsnclass import learn_loop

import argparse


def main(args):
    """
    Command line interface to run the active learning loop.

    Parameters
    ----------
    args: argparse
        Parameters give by the user.
    """

    # run active learning loop
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
