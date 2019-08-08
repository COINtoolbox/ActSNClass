# Copyright 2019 actsnclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 8 August 2019
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

from actsnclass import fit_bazin_samples


def main(args):
    """Fit the entire sample with the Bazin function."""

    # raw data directory
    data_dir = args.input
    features_file = args.output

    # fit the entire sample
    fit_bazin_samples(path_to_data_dir=data_dir, features_file=features_file)

if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='actsnclass - Fit Light curves module')
    parser.add_argument('-dd', '--datadir', dest='input', help='Path to directory holding raw data.', required=True)
    parser.add_argument('-o', '--output', dest='output', help='Path to output file.', required=True)

    args = parser.parse_args()

    main(args)
