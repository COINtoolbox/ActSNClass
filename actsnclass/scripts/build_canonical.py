# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 11 August 2019
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

from actsnclass.build_snpcc_canonical import *

__all__ = ['main']

def main(user_choices):
    """Build canonical sample for SNPCC data set fitted with Bazin features.

    Parameters
    ----------
    -c : bool
        If True, compute metadata on SNR and peak mag.
    -d : str (optional)
        Path to raw data directory. Needed only if "compute == False".
    -f : str
        Path to features file.
    -i: str (optional)
        Path to read metadata on SNR and peak mag from file.
        It is only used if "compute == False".
    -m : str (optional)
        Path to output file where to store metadata on SNR and peak mag.
        Only used if "save == True".
    -o : str
        Path to store full canonical sample.
    -p : str (optional)
        File to store comparison plot.
        If not provided plot is shown on screen.
    -s :  bool
        If True, save to file metadata on SNR and peakmag.

    Examples
    -------
    Use directly from the command line:

    >>> build_canonical.py -c <if True compute metadata>
    >>>       -d <path to raw data dir>
    >>>       -f <input features file> -m <output file for metadata>
    >>>       -o <output file for canonical sample> -p <comparison plot file>
    >>>       -s <if True save metadata to file>

    """

    print(user_choices.compute)

    sample = build_snpcc_canonical(path_to_raw_data=user_choices.raw_data_dir,
                                   path_to_features=user_choices.features,
                                   compute=user_choices.compute,
                                   save=user_choices.save,
                                   output_info_file=user_choices.output_meta,
                                   input_info_file=user_choices.input_meta,
                                   output_canonical_file=user_choices.output)

    plot_snpcc_train_canonical(sample, user_choices.output_plot_file)


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
                                                 'Build Canonical module')

    parser.add_argument('-c', '--compute', required=True, type=str2bool,
                        dest='compute', help='If True, compute metadata '
                                             'on SNR and peak mag.')
    parser.add_argument('-d', '--raw-data-dir', dest='raw_data_dir',
                        required=False, type=str, default=None,
                        help='Path to raw data directory. Needed only'
                             'if "compute == False".')
    parser.add_argument('-f', '--features', required=True, type=str,
                        dest='features', help='Path to features file.')
    parser.add_argument('-i', '--input', required=False,
                        type=str, default='', dest='input_meta',
                        help='Path to read metadata on SNR and peak mag'
                             'from file. It is only used if '
                             '"compute == False".')
    parser.add_argument('-m', '--output-meta-data-file', required=False,
                        dest='output_meta', type=str, default=None,
                        help='Path to output file where to store '
                             'metadata on SNR and peak mag. Only used'
                             'if "save == True".')
    parser.add_argument('-o', '--output', required=True, type=str,
                        dest='output', help='Path to store '
                                            'full canonical sample.')
    parser.add_argument('-p', '--output-plot-file', required=False, type=str,
                        default=None, help='File to store comparison plot.'
                                         'If not provided plot is '
                                         'shown on screen.',
                        dest='output_plot_file')
    parser.add_argument('-s', '--save', required=True, type=str2bool,
                        dest='save', help='If True, save to file metadata'
                                          'on SNR and peakmag.')

    # get input directory and output file name from user
    from_user = parser.parse_args()

    main(from_user)
