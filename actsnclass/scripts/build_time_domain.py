# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 12 August 2019
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

import argparse

from actsnclass.time_domain import SNPCCPhotometry

__all__ = ['main']


def main(user_choice):
    """Generates features files for a list of days of the survey.

    Parameters
    ----------
    -d: sequence
        Sequence of days since the begin of the survey to be processed.
    -p: str
        Complete path to raw data directory.
    -o: str
        Complete path to output time domain directory.

    Examples
    -------
    Use it directly from the command line.

    >>> build_time_domain.py -d 20 21 22 23 -p <path to raw data dir> -o <path to output time domain dir>
    """
    path_to_data = user_choice.raw_data_dir
    output_dir = user_choice.output
    day = user_choice.day_of_survey

    for item in day:
        data = SNPCCPhotometry()
        data.create_daily_file(output_dir=output_dir, day=item)
        data.build_one_epoch(raw_data_dir=path_to_data, day_of_survey=int(item),
                             time_domain_dir=output_dir)


if __name__ == '__main__':
    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='actsnclass - '
                                                 'Prepare Time Domain module')

    parser.add_argument('-d', '--day', dest='day_of_survey', required=True,
                        nargs='+', help='Day after survey starts.')
    parser.add_argument('-p', '--path-to-data', required=True, type=str,
                        dest='raw_data_dir',
                        help='Complete path to raw data directory.')
    parser.add_argument('-o', '--output', dest='output', required=True,
                        type=str, help='Path to output time domain directory.')

    # get input directory and output file name from user
    from_user = parser.parse_args()

    main(from_user)
