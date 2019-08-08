# Copyright 2019 actsnclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 7 August 2019
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

from actsnclass.bazin import fit_scipy, bazin

import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd

__all__ = ['LightCurve', 'fit_bazin_samples']


class LightCurve(object):
    """ Light Curve object, holding meta and photometric data.

    Parameters
    ----------
    bazin_features: list
        List with the 5 best-fit Bazin parameters in filter.
        Concatenated from blue to red.
    dataset_name: str
        Name of the survey or data set being analyzed.
    fitlers: list
        List of broad band filters.
    id: int
        SN identification number
    photometry: pd.DataFrame
        Photometry information. Keys --> [mjd, band, flux, fluxerr, SNR]
    redshift: float
        Redshift
    sample: str
        Original sample to which this light curve is assigned
    sim_peakmag: np.array
        Simulated peak magnitude in each filter
    sncode: int
        Number identifying the SN model used in the simulation
    sntype: str
        General classification, possibilities are: Ia, II or Ibc

    Methods
    -------
    load_snpcc_lc(path_to_data: str)
        Reads header and photometric information for 1 light curve
    fit_bazin(band: str) -> list
        Calculates best-fit parameters from the Bazin function in a given filter
    fit_bazin_all()
        Calculates  best-fit parameters from the Bazin function for all filters
    plot_bazin_fit(save: bool, show: bool, plot_dir: str)
        Plot the light curve points and Bazin fitted function

    Examples
    --------
    >>> from actsnclass import LightCurve

    # path to LC data file
    >>> path_to_lc = 'data/SIMGEN_PUBLIC_DES/DES_SN321244.DAT'

    >>> lc = LightCurve()
    >>> lc.load_snpcc_lc(path_to_lc)         # read LC
    >>> lc.photometry
          mjd band     flux  fluxerr   SNR
    0   56171.051    g -36.7900   78.090 -0.47
    1   56172.039    r   4.4100   17.580  0.25
    ..        ...  ...      ...      ...   ...
    91  56305.055    i  15.5400    3.070  5.06
    92  56305.070    z  30.0200    3.711  8.09

    [93 rows x 5 columns]

     >>> lc.fit_bazin_all()                   # apply Bazin fit to all filters
     >>> lc.bazin_features
     [62.0677260096896, -7.959383808822104, 47.37511467606875, 37.4919069623379,
     ... ... ...
     -0.2598668353407374, 48.41581554398582, 206.65806244385922, -4.777010246622081]

     >>> lc.plot_bazin_fit(save=True, show=False, plot_dir='../plots/')
     # This will save a plot with the light curve data and Bazin fit in all fitters

     # In order to fit the entire data set do:
     >>> data_dir = '../../data/SIMGEN_DES_PUBLIC/'            # directory with raw data
     >>> output_file = '../../data/Bazin_fit.dat'              # name of output Bazin file
     >>> fit_bazin_samples(path_to_data_dir=data_dir, features_file=output_file)
    """

    def __init__(self):
        self.bazin_features = []
        self.dataset_name = ' '
        self.filters = []
        self.id = 0
        self.photometry = pd.DataFrame()
        self.redshift = 0
        self.sample = ' '
        self.sim_peakmag = []
        self.sncode = 0
        self.sntype = ' '

    def load_snpcc_lc(self, path_to_data: str):
        """Reads one LC from SNPCC data.

        Populates the properties: dataset_name, id, sample, redshift, sncode,
        sntype, photometry and sim_peakmag.

        Parameters
        ---------
        path_to_data: str
            Path to text file with data from a single SN.
        """

        # set the designation of the data set
        self.dataset_name = 'SNPCC'

        # set filters
        self.filters = ['g', 'r', 'i', 'z']

        # set SN types
        snii = ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25', '26',
                '27', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44']

        snibc = ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16', '18',
                 '22', '23', '29', '45', '28']

        # read light curve data
        op = open(path_to_data, 'r')
        lin = op.readlines()
        op.close()

        # separate elements
        data_all = np.array([elem.split() for elem in lin])

        # flag useful lines
        flag_lines = np.array([True if len(line) > 1 else False for line in data_all])

        # get only informative lines
        data = data_all[flag_lines]

        photometry_raw = []               # store photometry
        header = []                      # store parameter header

        # get header information
        for line in data:
            if line[0] == 'SNID:':
                self.id = int(line[1])
            elif line[0] == 'SNTYPE:':
                if line[1] == '-9':
                    self.sample = 'test'
                else:
                    self.sample = 'train'
            elif line[0] == 'SIM_REDSHIFT:':
                self.redshift = float(line[1])
            elif line[0] == 'SIM_NON1a:':
                self.sncode = line[1]
                if line[1] in snibc:
                    self.sntype = 'Ibc'
                elif line[1] in snii:
                    self.sntype = 'II'
                elif line[1] == '0':
                    self.sntype = 'Ia'
                else:
                    raise ValueError('Unknown supernova type!')
            elif line[0] == 'VARLIST:':
                header: list = line[1:]
            elif line[0] == 'OBS:':
                photometry_raw.append(np.array(line[1:]))
            elif line[0] == 'SIM_PEAKMAG:':
                self.sim_peakmag = np.array([float(item) for item in line[1:5]])

        # transform photometry into array
        photometry_raw = np.array(photometry_raw)

        # put photometry into data frame
        self.photometry['mjd'] = np.array([float(item) for item in photometry_raw[:, header.index('MJD')]])
        self.photometry['band'] = np.array(photometry_raw[:, header.index('FLT')])
        self.photometry['flux'] = np.array([float(item) for item in photometry_raw[:, header.index('FLUXCAL')]])
        self.photometry['fluxerr'] = np.array([float(item) for item in photometry_raw[:, header.index('FLUXCALERR')]])
        self.photometry['SNR'] = np.array([float(item) for item in photometry_raw[:, header.index('SNR')]])

    def fit_bazin(self, band: str):
        """Extract Bazin features for one filter.

        Parameters
        ----------
        band: str
            Choice of broad band filter

        Returns
        -------
        bazin_param: list
            Best fit parameters for the Bazin function: [a, b, t0, tfall, trise]
        """

        # build filter flag
        filter_flag = self.photometry['band'] == band

        # get info for this filter
        time = self.photometry['mjd'].values[filter_flag]
        flux = self.photometry['flux'].values[filter_flag]

        # fit Bazin function
        bazin_param = fit_scipy(time - time[0], flux)

        return bazin_param

    def fit_bazin_all(self):
        """Perform Bazin fit for all filters independently and concatenate results.

        Populates the property: bazin_features.
        """

        self.bazin_features = []

        for band in self.filters:
            # build filter flag
            filter_flag = self.photometry['band'] == band

            if sum(filter_flag) > 4:
                best_fit = self.fit_bazin(band)

                if sum([str(item) == 'nan' for item in best_fit]) == 0:
                    for fit in best_fit:
                        self.bazin_features.append(fit)
                else:
                    for i in range(5):
                        self.bazin_features.append('None')
            else:
                for i in range(5):
                    self.bazin_features.append('None')

    def plot_bazin_fit(self, save=True, show=False, plot_dir=' '):
        """Plot fitted light curve and data points

        Parameters
        ----------
        save: bool (optional)
            Save plot to file. Default is True
        show: bool (optional)
            Show plot on screen. Default is False
        plot_dir: str (optional)
            Directory where to store the plot files

        Returns
        -------
        lc: image, if show==True
            Plot of LC fit and data points
        """

        plt.figure()
        plt.suptitle('DES_SN' + str(self.id) + ', z = ' + str(self.redshift))
        for i in range(len(self.filters)):
            # filter flag for photometry data
            filter_flag = self.photometry['band'] == self.filters[i]

            if sum(filter_flag) > 1:
                x = self.photometry['mjd'][filter_flag].values
                y = self.photometry['flux'][filter_flag].values
                yerr = self.photometry['fluxerr'][filter_flag].values

                # shift time scale to avoid large numbers in x-axis
                time = self.photometry['mjd'].values - min(self.photometry['mjd'].values)
                time_axis = np.linspace(0, max(time), 500)[:, np.newaxis]

                fitted_flux = np.array([bazin(t, self.bazin_features[0], self.bazin_features[1], self.bazin_features[2],
                                        self.bazin_features[3], self.bazin_features[4]) for t in time_axis])

            plt.subplot(2, len(self.filters) / 2 + len(self.filters) % 2, i + 1)
            plt.title(self.filters[i])
            if len(x) > 1:
                plt.errorbar(x - min(x), y, yerr=yerr, fmt='o', color='blue')
                plt.plot(time_axis, fitted_flux, color='red', lw=1.5)
            plt.xlabel('MJD - ' + str(min(x)))
            plt.ylabel('Flux')
            plt.tight_layout()

        if save:
            plt.savefig(plot_dir + 'DES_SN' + str(self.id) + '.png')
        elif show:
            plt.show()



def fit_bazin_samples(path_to_data_dir: str, features_file: str):
    """Fit Bazin functions to all filters in training and test samples.

    Parameters
    ----------
    path_to_data_dir: str
        Path to directory containing the set of individual files, one for each light curve.
    features_file: str
        Path to output file where results should be stored.
    """

    # read file names
    file_list_all = os.listdir(path_to_data_dir)
    lc_list = [elem for elem in file_list_all if 'DES_SN' in elem]

    # count survivers
    count_surv = 0

    # add headers to files
    with open(features_file, 'w') as param_file:
        param_file.write('id redshift type code sample gA gB gt0 gtfall gtrise rA rB rt0 rtfall rtrise iA iB it0 ' +
                         'itfall itrise zA zB zt0 ztfall ztrise\n')

    for file in lc_list:

        # fit individual light curves
        lc = LightCurve()
        lc.load_snpcc_lc(path_to_data_dir + file)
        lc.fit_bazin_all()

        print(lc_list.index(file), ' - id:', lc.id)

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(features_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

    param_file.close()


def main():
    print(__doc__)


if __name__ == '__main__':
    main()
fit_scipy