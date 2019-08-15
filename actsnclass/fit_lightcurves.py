# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 9 August 2019
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

from actsnclass.bazin import bazin, fit_scipy

import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd

__all__ = ['LightCurve', 'fit_snpcc_bazin']


class LightCurve(object):
    """ Light Curve object, holding meta and photometric data.

    Attributes
    ----------
    bazin_features_names: list
        List of names of the Bazin function parameters.
    bazin_features: list
        List with the 5 best-fit Bazin parameters in all filters.
        Concatenated from blue to red.
    dataset_name: str
        Name of the survey or data set being analyzed.
    filters: list
        List of broad band filters.
    id: int
        SN identification number
    photometry: pd.DataFrame
        Photometry information. Keys --> [mjd, band, flux, fluxerr, SNR, MAG, MAGERR].
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
    check_queryable(mjd: float, r_lim: float)
        Check if this light can be queried in a given day.
    load_snpcc_lc(path_to_data: str)
        Reads header and photometric information for 1 light curve
    fit_bazin(band: str) -> list
        Calculates best-fit parameters from the Bazin function in 1 filter
    fit_bazin_all()
        Calculates  best-fit parameters from the Bazin func for all filters
    plot_bazin_fit(save: bool, show: bool, output_file: srt)
        Plot photometric points and Bazin fitted curve

    Examples
    --------
    >>> from actsnclass import LightCurve

    define path to light curve file

    >>> path_to_lc = 'data/SIMGEN_PUBLIC_DES/DES_SN431546.DAT'

    >>> lc = LightCurve()                        # create light curve instance
    >>> lc.load_snpcc_lc(path_to_lc)             # read data
    >>> lc.photometry                            # display photometry
              mjd band     flux  fluxerr   SNR
    0   56207.188    g   9.6560    4.369  2.21
    1   56207.195    r   6.3370    3.461  1.83
    ...        ...  ...      ...      ...   ...
    96  56336.043    r  14.4300    3.098  4.66
    97  56336.055    i  18.9500    5.029  3.77
    [98 rows x 5 columns]

    >>> lc.fit_bazin_all()                  # perform Bazin fit in all filters
    >>> lc.bazin_features                   # display Bazin parameters
    [62.0677260096896, -7.959383808822104, 47.37511467606875, 37.4919069623379,
    ... ... ...
    206.65806244385922, -4.777010246622081]

    plot light curve fit

    >>> lc.plot_bazin_fit(output_file=str(lc.id) + '.png')

    for fitting the entire sample...

    >>> path_to_data_dir = 'data/SIMGEN_PUBLIC_DES/'     # raw data directory
    >>> output_file = 'results/Bazin.dat'       # output file
    >>> fit_snpcc_bazin(path_to_data_dir=path_to_data_dir, features_file=output_file)

    a file with all Bazin fits for this data set was produced
    """

    def __init__(self):
        self.bazin_features = []
        self.bazin_features_names = ['a', 'b', 't0', 'tfall', 'trsise']
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

        Populates the attributes: dataset_name, id, sample, redshift, sncode,
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
        snii = ['2', '3', '4', '12', '15', '17', '19', '20', '21', '24', '25',
                '26', '27', '30', '31', '32', '33', '34', '35', '36', '37',
                '38', '39', '40', '41', '42', '43', '44']

        snibc = ['1', '5', '6', '7', '8', '9', '10', '11', '13', '14', '16',
                 '18', '22', '23', '29', '45', '28']

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
        self.photometry['MAG'] = np.array([float(item) for item in photometry_raw[:, header.index('MAG')]])
        self.photometry['MAGERR'] = np.array([float(item) for item in photometry_raw[:, header.index('MAGERR')]])

    def check_queryable(self, mjd: float, r_lim: float):
        """Check if this light can be queried in a given day.

        This checks only r-band mag limit in a given epoch.
        It there is no observation on that day, use the last available
        observation.

        Parameters
        ----------
        mjd: float
            MJD where the query will take place.
        r_lim: float
            r-band magnitude limit below which query is possible.

        Returns
        -------
        bool
            If true, sample is changed to `queryable`.
        """

        # create photo flag
        photo_flag = self.photometry['mjd'].values <= mjd
        rband_flag = self.photometry['band'].values == 'r'
        surv_flag = np.logical_and(photo_flag, rband_flag)

        # check surviving photometry
        surv_mag = self.photometry['MAG'].values[surv_flag]

        if len(surv_mag) > 0 and 0 < surv_mag[-1] <= r_lim:
            return True
        else:
            return False

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

        Populates the attributes: bazin_features.
        """

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

    def plot_bazin_fit(self, save=True, show=False, output_file=' '):
        """
        Plot data and Bazin fitted function.

        Parameters
        ----------
        save: bool (optional)
             Save figure to file. Default is True.
        show: bool (optinal)
             Display plot in windown. Default is False.
        output_file: str
            Name of file to store the plot.
        """

        plt.figure()

        for i in range(len(self.filters)):
            plt.subplot(2, len(self.filters) / 2 + len(self.filters) % 2, i + 1)
            plt.title('Filter: ' + self.filters[i])

            # filter flag
            filter_flag = self.photometry['band'] == self.filters[i]
            x = self.photometry['mjd'][filter_flag].values
            y = self.photometry['flux'][filter_flag].values
            yerr = self.photometry['fluxerr'][filter_flag].values

            # shift to avoid large numbers in x-axis
            time = x - min(x)
            xaxis = np.linspace(0, max(time), 500)[:, np.newaxis]
            # calculate fitted function
            fitted_flux = np.array([bazin(t, self.bazin_features[i * 5],
                                          self.bazin_features[i * 5 + 1],
                                          self.bazin_features[i * 5 + 2],
                                          self.bazin_features[i * 5 + 3],
                                          self.bazin_features[i * 5 + 4])
                                    for t in xaxis])

            plt.errorbar(time, y, yerr=yerr, color='blue', fmt='o')
            plt.plot(xaxis, fitted_flux, color='red', lw=1.5)
            plt.xlabel('MJD - ' + str(min(x)))
            plt.ylabel('FLUXCAL')
            plt.tight_layout()

        if save:
            plt.savefig(output_file)
        if show:
            plt.show()


def fit_snpcc_bazin(path_to_data_dir: str, features_file: str):
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
    return None


if __name__ == '__main__':
    main()
