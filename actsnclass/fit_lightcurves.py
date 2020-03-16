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

import io
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import tarfile

__all__ = ['LightCurve', 'fit_snpcc_bazin', 'fit_resspect_bazin', 'fit_plasticc_bazin']


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
    id_name:
        Column name of object identifier.
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
    evaluate_bazin(param: list, time: np.array) -> np.array
        Evaluate the Bazin function given parameter values.
    load_snpcc_lc(path_to_data: str)
        Reads header and photometric information for 1 light curve
    load_plasticc_lc(photo_file: str, snid: int)
	Load photometric information for 1 PLAsTiCC light curve
    load_resspect_lc(photo_file: str, snid: int)
	Load photometric information for 1 RESSPECT light curve
    fit_bazin(band: str) -> list
        Calculates best-fit parameters from the Bazin function in 1 filter
    fit_bazin_all()
        Calculates  best-fit parameters from the Bazin func for all filters
    plot_bazin_fit(save: bool, show: bool, output_file: srt)
        Plot photometric points and Bazin fitted curve

    Examples
    --------
    >>> from actsnclass import LightCurve

    ##### for SNPCC light curves
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

    a file with all Bazin fits for this data set was produced.

    ##### for RESSPECT light curves 
    >>> data_dir = '../data/light_curves/'           # path to data directory
    >>> header_file = 'RESSPECT_PERFECT_PHOTO_TRAIN.csv'
    >>> photo_file = 'RESSPECT_PERFECT_HEADER_TRAIN.csv'

    >>> header = pd.read_csv(data_dir + header_file)
    >>> snid = header['SNID'].values[0]

    >>> lc = LightCurve()                       
    >>> lc.load_resspect_lc(data_dir + photo_file, snid) 
    >>> lc.photometry
             mjd band       flux   fluxerr         SNR
    0    53214.0    u   0.165249  0.142422    1.160276
    1    53214.0    g  -0.041531  0.141841   -0.292803
    ..       ...  ...        ...       ...         ...
    472  53370.0    z  68.645930  0.297934  230.406460
    473  53370.0    Y  63.254270  0.288744  219.067050

    >>> lc.fit_bazin_all()               # perform Bazin fit in all filters
    >>> lc.bazin_features                # display Bazin parameters
    [198.63302952843623, -9.38297128588733, 43.99971014717201,
    ... ...
    -1.546372806815066]

    for fitting the entire sample ...

    >>> output_file = 'RESSPECT_PERFECT_TRAIN.DAT'
    >>> fit_resspect_bazin(photo_file, header_file, output_file, sample='train')
    """

    def __init__(self):
        self.bazin_features = []
        self.bazin_features_names = ['a', 'b', 't0', 'tfall', 'trsise']
        self.dataset_name = ' '
        self.filters = []
        self.id = 0
        self.id_name = None
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
                self.id_name = 'SNID'
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

    def load_resspect_lc(self, photo_file, snid):
        """
        Return 1 light curve from RESSPECT simulations.
    
        Parameters
        ----------
        photo_file: str
            Complete path to full sample file.
        snid: int
            Identification number for the desired light curve.
        """

        if '.tar.gz' in photo_file:
            tar = tarfile.open(photo_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            all_photo = pd.read_csv(io.BytesIO(content))
            tar.close()
        else:    
            all_photo = pd.read_csv(photo_file, index_col=False)

            if ' ' in all_photo.keys()[0]:
                all_photo = pd.read_csv(photo_file, sep=' ', index_col=False)

        if 'SNID' in all_photo.keys():
            flag = all_photo['SNID'] == snid
            self.id_name = 'SNID'
        elif 'snid' in all_photo.keys():
            flag = all_photo['snid'] == snid
            self.id_name = 'snid'
        elif 'objid' in all_photo.keys():
            flag = all_photo['objid'] == snid
            self.id_name = 'objid'
        elif 'id' in all_photo.keys():
            flag = all_photo['id'] == snid
            self.id_name = 'id'

        photo = all_photo[flag]
           
        self.dataset_name = 'RESSPECT'              # name of data set
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']       # list of filters
        self.id = snid 
        self.photometry = {}
        self.photometry['mjd'] = photo['MJD'].values
        self.photometry['band'] = photo['FLT'].values
        self.photometry['flux'] = photo['FLUXCAL'].values
        self.photometry['fluxerr'] = photo['FLUXCALERR'].values
        self.photometry['SNR'] = photo['SNR'].values
        self.photometry = pd.DataFrame(self.photometry)
        
    def load_plasticc_lc(self, photo_file: str, snid: int):
        """
        Return 1 light curve from PLAsTiCC simulations.
    
        Parameters
        ----------
        photo_file: str
            Complete path to full sample file.
        snid: int
            Identification number for the desired light curve.
        """

        if '.tar.gz' in photo_file:
            tar = tarfile.open(photo_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            all_photo = pd.read_csv(io.BytesIO(contente))
        else:
            all_photo = pd.read_csv(photo_file, index_col=False)

            if ' ' in all_photo.keys()[0]:
                all_photo = pd.read_csv(photo_file, sep=' ', index_col=False)

        if 'object_id' in all_photo.keys():
            flag = all_photo['object_id'] == snid
            self.id_name = 'object_id'
        elif 'SNID' in all_photo.keys():
            flag = all_photo['SNID'] == snid
            self.id_name = 'SNID'
        elif 'snid' in all_photo.keys():
            flag = all_photo['snid'] == snid
            self.id_name = 'snid'

        photo = all_photo[flag]

        filter_dict = {0:'u', 1:'g', 2:'r', 3:'i', 4:'z', 5:'Y'}
           
        self.dataset_name = 'PLAsTiCC'              # name of data set
        self.filters = ['u', 'g', 'r', 'i', 'z', 'Y']       # list of filters
        self.id = snid
        self.photometry = {}
        self.photometry['mjd'] = photo['mjd'].values
        self.photometry['band'] = [filter_dict[photo['passband'].values[k]] 
                                   for k in range(photo['passband'].shape[0])]
        self.photometry['flux'] = photo['flux'].values
        self.photometry['fluxerr'] = photo['flux_err'].values
        self.photometry['detected_bool'] = photo['detected_bool'].values
        self.photometry = pd.DataFrame(self.photometry)

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

        if 'MAG' in self.photometry.keys():
            # check surviving photometry
            surv_mag = self.photometry['MAG'].values[surv_flag]

        else:
            surv_flux = self.photometry['flux'].values[surv_flag]
            if surv_flux[-1] > 0:
                surv_mag = [2.5 * (11 - np.log10(surv_flux[-1]))]
            else:
                surv_mag = []

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

    def evaluate_bazin(self, param: list, time: np.array):
        """Evaluate the Bazin function given parameter values.

        Parameters
        ----------
        param: list
            List of Bazin parameters in order [a, b, t0, tfall, trise] 
            for all filters, concatenated from blue to red
        time: np.array or list
            Time since maximum where to evaluate the Bazin fit.

        Returns
        -------
        np.array
            Value of the Bazin function in each required time
        """
        # store flux values and starting points
        flux = []
        first_obs = []
        tmax_all = []

        for k in range(len(self.filters)):
            # find day of maximum
            x = range(400)
            y = [bazin(epoch, param[0 + k * 5], 
                      param[1 + k * 5], param[2 + k * 5], param[3 + k * 5], param[4 + k * 5])
                      for epoch in x]

            t_max = x[y.index(max(y))]
            tmax_all.append(t_max)
            
            for item in time:
                epoch = t_max + item
                flux.append(bazin(epoch, param[0 + k * 5], 
                      param[1 + k * 5], param[2 + k * 5], param[3 + k * 5], param[4 + k * 5]))

            first_obs.append(t_max + time[0])

        return np.array(flux), first_obs, tmax_all
        

    def fit_bazin_all(self):
        """Perform Bazin fit for all filters independently and concatenate results.

        Populates the attributes: bazin_features.
        """
        # remove previous fit attempts
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

    def plot_bazin_fit(self, save=True, show=False, output_file=' ', figscale=1):
        """
        Plot data and Bazin fitted function.

        Parameters
        ----------
        save: bool (optional)
             Save figure to file. Default is True.
        show: bool (optinal)
             Display plot in windown. Default is False.
        output_file: str (optional)
            Name of file to store the plot.
        figscale: float (optional)
            Allow to control the size of the figure.
        """

        # number of columns in the plot
        ncols = len(self.filters) / 2 + len(self.filters) % 2
        fsize = (figscale * 5 * ncols , figscale * 10)
        
        plt.figure(figsize=fsize)

        for i in range(len(self.filters)):
            plt.subplot(2, ncols, i + 1)
            plt.title('Filter: ' + self.filters[i])

            # filter flag
            filter_flag = self.photometry['band'] == self.filters[i]
            x = self.photometry['mjd'][filter_flag].values
            y = self.photometry['flux'][filter_flag].values
            yerr = self.photometry['fluxerr'][filter_flag].values

            if len(x) > 4:
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

            else:
                plt.xlabel('MJD')
            plt.ylabel('FLUXCAL')
            plt.tight_layout()

        if save:
            plt.savefig(output_file)
            plt.show('all')
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

def fit_resspect_bazin(path_photo_file: str, path_header_file:str,
                       output_file: str, sample=None):
    """Fit Bazin functions to all filters.

    Parameters
    ----------
    path_photo_file: str
        Complete path to light curve file.
    path_header_file: str
        Complete path to header file.
    output_file: str
        Output file where the features will be stored.
    sample: str
	'train' or 'test'. Default is None.
    """
    # count survivers
    count_surv = 0

    # read header information
    if '.tar.gz' in path_header_file:
        tar = tarfile.open(path_header_file, 'r:gz')
        fname = tar.getmembers()[0]
        content = tar.extractfile(fname).read()
        header = pd.read_csv(io.BytesIO(content))
        tar.close()
        
    else:    
        header = pd.read_csv(path_header_file, index_col=False)
        if ' ' in header.keys()[0]:
            header = pd.read_csv(path_header_file, sep=' ', index_col=False)
    
    # add headers to files
    with open(output_file, 'w') as param_file:
        param_file.write('id redshift type code sample uA uB ut0 utfall ' +
                         'utrise gA gB gt0 gtfall gtrise rA rB rt0 rtfall ' +
                         'rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ' + 
                         'ztrise YA YB Yt0 Ytfall Ytrise\n')

    # check id flag
    if 'SNID' in header.keys():
        id_name = 'SNID'
    elif 'snid' in header.keys():
        id_name = 'snid'
    elif 'objid' in header.keys():
        id_name = 'objid'

    # check redshift flag
    if 'redshift' in header.keys():
        z_name = 'redshift'
    elif 'REDSHIFT_FINAL' in header.keys():
        z_name = 'REDSHIFT_FINAL'

    # check type flag
    if 'type' in header.keys():
        type_name = 'type'
    elif 'SIM_TYPE_NAME' in header.keys():
        type_name = 'SIM_TYPE_NAME'

    # check subtype flag
    if 'code' in header.keys():
        subtype_name = 'code'
    elif 'SIM_TYPE_INDEX' in header.keys():
        subtype_name = 'SIM_TYPE_NAME'

    for snid in header[id_name].values:      

        # load individual light curves
        lc = LightCurve()                       
        lc.load_resspect_lc(path_photo_file, snid)

        # check filter name
        if 'b' in lc.photometry['band'].iloc[0]:
            for i in range(lc.photometry['band'].values.shape[0]):
                for f in lc.filters:
                    if "b'" + f + " '" == str(lc.photometry['band'].values[i]):
                        lc.photometry['band'].iloc[i] = f
                
        lc.fit_bazin_all()

        # get model name 
        lc.redshift = header[z_name][header[lc.id_name] == snid].values[0]
        lc.sntype = header[type_name][header[lc.id_name] == snid].values[0]
        lc.sncode = header[subtype_name][header[lc.id_name] == snid].values[0]
        lc.sample = sample

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(output_file, 'a') as param_file:
                param_file.write(str(lc.id) + ' ' + str(lc.redshift) + ' ' + str(lc.sntype) + ' ')
                param_file.write(str(lc.sncode) + ' ' + str(lc.sample) + ' ')
                for item in lc.bazin_features:
                    param_file.write(str(item) + ' ')
                param_file.write('\n')

    param_file.close()


def fit_plasticc_bazin(path_photo_file: str, path_header_file:str,
                       output_file: str, sample=None):
    """Fit Bazin functions to all filters.

    Parameters
    ----------
    path_photo_file: str
        Complete path to light curve file.
    path_header_file: str
        Complete path to header file.
    output_file: str
        Output file where the features will be stored.
    sample: str
	'train' or 'test'. Default is None.
    """
    types = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
             95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
             16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
             993:'CART', 994:'PISN',995:'MLString'}

    # count survivers
    count_surv = 0

    # read header information
    if '.tar.gz' in path_header_file:
        tar = tarfile.open(path_header_file, 'r:gz')
        fname = tar.getmembers()[0]
        content = tar.extracfile(fname).read()
        header = pd.read_csv(io.BytesIO(content))
        tar.close()
    else:
        header = pd.read_csv(path_header_file, index_col=False)

        if ' ' in header.keys()[0]:
            header = pd.read_csv(path_header_file, sep=' ', index_col=False)

    # add headers to files
    with open(output_file, 'w') as param_file:
        param_file.write('id redshift type code sample uA uB ut0 utfall ' +
                         'utrise gA gB gt0 gtfall gtrise rA rB rt0 rtfall ' +
                         'rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ' + 
                         'ztrise YA YB Yt0 Ytfall Ytrise\n')

    # check id flag
    if 'SNID' in header.keys():
        id_name = 'SNID'
    elif 'snid' in header.keys():
        id_name = 'snid'
    elif 'objid' in header.keys():
        id_name = 'objid'

    for snid in header[id_name].values:      

        # load individual light curves
        lc = LightCurve()                       
        lc.load_plasticc_lc(path_photo_file, snid) 
        lc.fit_bazin_all()

        # get model name 
        lc.redshift = header['true_z'][header[lc.id_name] == snid].values[0]
        lc.sntype = types[header['true_target'][header[lc.id_name] == snid].values[0]]            
        lc.sncode = header['true_target'][header[lc.id_name] == snid].values[0]
        lc.sample = sample

        # append results to the correct matrix
        if 'None' not in lc.bazin_features:
            count_surv = count_surv + 1
            print('Survived: ', count_surv)

            # save features to file
            with open(output_file, 'a') as param_file:
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
