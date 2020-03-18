# Copyright 2020  RESSPECT software
# Author: Bruno Quint
#
# created on 17 March 2020
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
import numpy as np
import os
import pandas as pd
​
from actsnclass.snana_fits_to_pd import read_fits, save_fits


def  _parse_arguments():
    """
    Parses arguments from the command line.
    """
    parser = argparse.ArgumentParser(description='Sub-sample photometric curves.')
    
    parser.add_argument('frequency', type=int, 
                        help='sub-sample frequency in number of days')
​
    parser.add_argument('filename', type=str, 
                        help='input photometric curve data filename')
​
    return parser.parse_args()

​
def main():
    """
    Sub-sample PHOT data getting data on each X days.
    """
    
    args = _parse_arguments()
​
    print('\n Reading input file:\n  {:s}'.format(args.filename))

    header, photo = read_fits(args.filename)

    print(' Ok.')
    
    print(' Sub-sampling data frame')
    mjd = photo.MJD.unique()
    mjd = mjd[mjd > 0]
    mjd.sort()
    mjd = np.arange(mjd.min(), mjd.max(), args.frequency)
​
    sub_photo = photo[photo.MJD.isin(mjd)]
    print(sub_photo.columns)
​
    header_fname = args.filename.replace("_PHOT", "_HEAD_{:d}days".format(args.frequency)) 
    print(' Writing metadata to:\n  {}'.format(header_fname))
    save_fits(header, header_fname)
    
    photo_fname = args.filename.replace("_PHOT", "_PHOT_{:d}days".format(args.frequency))
    print(' Writing data frame to:\n  {}\n'.format(photo_fname))
    save_fits(sub_photo, photo_fname)
​
​
if __name__ == "__main__":
    main()
