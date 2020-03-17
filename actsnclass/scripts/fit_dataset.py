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

from actsnclass.fit_lightcurves import fit_snpcc_bazin
from actsnclass.fit_lightcurves import fit_resspect_bazin

__all__ = ['main']


def main(user_choices):
    """Fit the entire sample with the Bazin function.

    All results are saved to file.

    Parameters
    ----------
    -o: str
        Path to output feature file.
    -s: str
        Simulation name. Options are 'SNPCC' or 'RESSPECT'.
    -dd: str (optional)
        Path to directory containing raw data.
        Only used for SNPCC simulations.
    -hd: str (optional)
        Path to header file. Only used for RESSPECT simulations.
    -p: str (optional)
        Path to photometry file. Only used for RESSPECT simulations.
    -sp: str or None (optional)
        Sample to be fitted. Options are 'train', 'test' or None.
        Default is None.

    Examples
    --------

    For SNPCC: 

    >>> fit_dataset.py -s SNPCC -dd <path_to_data_dir> -o <output_file>

    For RESSPECT:

    >>> fit_dataset.py -s RESSPECT -p <path_to_photo_file> 
             -hd <path_to_header_file> -o <output_file> 
    """

    # raw data directory
    data_dir = user_choices.input
    features_file = user_choices.output

    if user_choices.sim_name == 'SNPCC':
        # fit the entire sample
        fit_snpcc_bazin(path_to_data_dir=data_dir, features_file=features_file)
    
    elif user_choices.sim_name == 'RESSPECT':
        fit_resspect_bazin(path_photo_file=user_choices.photo_file,
                           path_header_file=user_choices.header_file,
                           output_file=features_file, sample=user_choices.sample)

    return None


if __name__ == '__main__':

    # get input directory and output file name from user
    parser = argparse.ArgumentParser(description='actsnclass - Fit Light curves module')
   
    parser.add_argument('-dd', '--datadir', dest='input',
                        help='Path to directory holding raw data. Only used for SNPCC',
                        required=False, default=' ')
    parser.add_argument('-hd', '--header', dest='header_file', 
                        help='Path to header file. Only used for RESSPECT.',
                        required=False, default=' ')
    parser.add_argument('-o', '--output', dest='output', help='Path to output file.', 
                        required=True)
    parser.add_argument('-p', '--photo', dest='photo_file',
                        help='Path to photometry file. Only used for RESSPECT.',
                        required=False, default=' ')
    parser.add_argument('-s', '--simulation', dest='sim_name', 
                        help='Name of simulation (data set). ' + \
                             'Options are "SNPCC" or "RESSPECT".',
                        required=True)
    parser.add_argument('-sp', '--sample', dest='sample',
                        help='Sample to be fitted. Options are "train", ' + \
                             ' "test" or None.',
                        required=False, default=None)

    user_input = parser.parse_args()

    main(user_input)
