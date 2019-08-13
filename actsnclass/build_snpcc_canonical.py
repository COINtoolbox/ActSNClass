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

import numpy as np
import os
import pandas as pd
import matplotlib.pylab as plt

from sklearn.neighbors import NearestNeighbors, KernelDensity

from actsnclass.fit_lightcurves import LightCurve
from actsnclass.database import DataBase

__all__ = ['Canonical', 'build_snpcc_canonical', 'plot_snpcc_train_canonical']


class Canonical(object):
    """Canonical sample object.

    Attributes
    ----------
    canonical_ids: list
        List of ids for objects in the canonical sample.
    canonical_sample: list
        Complete data matrix for the canonical sample.
    meta_data: pd.DataFrame
        Metadata on sim peakmag and SNR for all objects in the original data set.
    test_ia_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ias in the test sample.
    test_ia_id: np.array
        Set of ids for all SN Ia in the test sample.
    test_ibc_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ibcs in the test sample.
    test_ibc_id: np.array
        Set of ids for all SN Ibc in the test sample.
    test_ii_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN IIs in the test sample.
    test_ii_id: np.array
        Set of ids for all SN II in the test sample.
    train_ia_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ias in the train sample.
    train_ia_id: np.array
        Set of ids for all SN Ia in the train sample.
    train_ibc_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN Ibcs in the train sample.
    train_ibc_id: np.array
        Set of ids for all SN Ibc in the train sample.
    train_ii_data: pd.DataFrame
        Metadata on sim peakmag and SNR for SN IIs in the train sample.
    train_ii_id: np.array
        Set of ids for all SN II in the train sample.

    Methods
    -------
    snpcc_get_canonical_info(path_to_rawdata_dir: str, canonical_output_file: st, compute: bool, save: bool, canonical_input_file: str)
        Load SNPCC metada data required to characterize objects.
    snpcc_identify_samples()
        Identify training and test sample.
    find_neighbors()
        Identify 1 nearest neighbor for each object in training.
    """

    def __init__(self):
        self.canonical_ids = []
        self.canonical_sample = []
        self.meta_data = pd.DataFrame()
        self.test_ia_data = pd.DataFrame()
        self.test_ia_id = np.array([])
        self.test_ibc_data = pd.DataFrame()
        self.test_ibc_id = np.array([])
        self.test_ii_data = pd.DataFrame()
        self.test_ii_id = np.array([])
        self.train_ia_data = pd.DataFrame()
        self.train_ia_id = np.array([])
        self.train_ibc_data = pd.DataFrame()
        self.train_ibc_id = np.array([])
        self.train_ii_data = pd.DataFrame()
        self.train_ii_id = np.array([])

    def snpcc_get_canonical_info(self, path_to_rawdata_dir: str,
                                 canonical_output_file: str,
                                 compute=True, save=True,
                                 canonical_input_file=''):
        """
        Load SNPCC metada data required to characterize objects.

        Populates attribute: data.

        Parameters
        ----------
        path_to_rawdata_dir: str
            Complete path to directory holding raw data files.
        canonical_output_file: str
            Complete path to output canonical sample file.
        compute: bool (optional)
            Compute required metada from raw data files.
            Default is True.
        save: bool (optional)
            Save metadata to file. Default is True.
        canonical_input_file: str (optional)
            Path to input file if required metadata was previously calculated.
            If name is give, 'compute' must be False.
        """

        if compute:
            # read file names
            file_list_all = os.listdir(path_to_rawdata_dir)
            lc_list = [elem for elem in file_list_all if 'DES_SN' in elem]

            sim_info_matrix = []

            for fname in lc_list:

                print('Processed for canonical: ', str(lc_list.index(fname)))

                # fit individual light curves
                lc = LightCurve()
                lc.load_snpcc_lc(path_to_rawdata_dir + fname)

                # store information for id, redshift and sim peak magnitude
                line = [lc.id, lc.sample, lc.sntype, lc.redshift]
                for val in lc.sim_peakmag:
                    line.append(val)

                # calculate mean SNR for each filter
                for f in lc.filters:
                    filter_flag = lc.photometry['band'].values == f
                    if sum(filter_flag) > 0:
                        snr = lc.photometry['SNR'].values[filter_flag]
                        mean_snr = np.mean(snr)
                        line.append(mean_snr)
                    else:
                        line.append(0)

                sim_info_matrix.append(line)

            metadata_header = ['snid', 'sample', 'sntype', 'z', 'g_pkmag',
                               'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR',
                               'r_SNR', 'i_SNR', 'z_SNR']
            self.meta_data = pd.DataFrame(sim_info_matrix,
                                          columns=metadata_header)

        else:
            if not canonical_input_file:
                raise ValueError('File not found! Set "calculate = True" '
                                 'to build canonical info file.')

            else:
                self.meta_data = pd.read_csv(canonical_input_file, sep=' ',
                                             index_col=False)

        if save:
            self.meta_data.to_csv(canonical_output_file, sep=' ', index=False)

    def snpcc_identify_samples(self):
        """Identify training and test sample.

        Populates attributes: train_ia_data, train_ia_id, train_ibc_data,
        train_ibc_id, train_ii_data, train_ibc_id, test_ia_data, test_ia_id,
        test_ibc_data, test_ibc_id, test_ii_data and test_ii_id.
        """

        # define parameters
        features_list = ['z', 'g_pkmag', 'r_pkmag', 'i_pkmag',
                         'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR', 'z_SNR']
        train_flag = self.meta_data['sample'].values == 'train'
        ia_flag = self.meta_data['sntype'].values == 'Ia'
        ibc_flag = self.meta_data['sntype'].values == 'Ibc'
        ii_flag = self.meta_data['sntype'].values == 'II'

        # get training sample
        self.train_ia_data = \
            self.meta_data[features_list][np.logical_and(train_flag, ia_flag)]
        self.train_ia_id = \
            self.meta_data['snid'].values[np.logical_and(train_flag, ia_flag)]
        self.train_ibc_data = \
            self.meta_data[features_list][np.logical_and(train_flag, ibc_flag)]
        self.train_ibc_id = \
            self.meta_data['snid'].values[np.logical_and(train_flag, ibc_flag)]
        self.train_ii_data = \
            self.meta_data[features_list][np.logical_and(train_flag, ii_flag)]
        self.train_ii_id = \
            self.meta_data['snid'].values[np.logical_and(train_flag, ii_flag)]

        # get test sample
        self.test_ia_data = \
            self.meta_data[features_list][np.logical_and(~train_flag, ia_flag)]
        self.test_ia_id = \
            self.meta_data['snid'].values[np.logical_and(~train_flag, ia_flag)]
        self.test_ibc_data = \
            self.meta_data[features_list][np.logical_and(~train_flag, ibc_flag)]
        self.test_ibc_id = \
            self.meta_data['snid'].values[np.logical_and(~train_flag, ibc_flag)]
        self.test_ii_data = \
            self.meta_data[features_list][np.logical_and(~train_flag, ii_flag)]
        self.test_ii_id = \
            self.meta_data['snid'].values[np.logical_and(~train_flag, ii_flag)]

    def find_neighbors(self):
        """Identify 1 nearest neighbor for each object in training.

        Populates attribute: canonical_ids.
        """

        # gather samples by type
        train_samples = [self.train_ia_data, self.train_ibc_data,
                         self.train_ii_data]
        test_samples = [self.test_ia_data, self.test_ibc_data,
                        self.test_ii_data]
        test_ids = [self.test_ia_id, self.test_ibc_id, self.test_ii_id]
        types = ['Ia', 'Ibc', 'II']

        # store the objects already chosen
        vault = []

        # find nearest neighbor
        for i in range(len(test_samples)):
            print('Scanning ', types[i], ' . . . ')

            # find 10 neighbors in case there are repetitions
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto')
            nbrs.fit(test_samples[i])
            for j in range(train_samples[i].shape[0]):

                # search samples by type
                stype = np.array(train_samples[i].values[j]).reshape(1, -1)
                indices = nbrs.kneighbors(stype)

                # only add elements which were not added in a previous loop
                done = False
                count = 0
                while not done:
                    if indices[1][0][count] not in vault:
                        element = test_ids[i][indices[1][0][count]]
                        self.canonical_ids.append(element)
                        vault.append(indices[1][0][count])
                        done = True
                    else:
                        count = count + 1

            print('Processed: ', len(vault))


def build_snpcc_canonical(path_to_raw_data: str, path_to_features: str,
                          output_canonical_file: str, output_info_file='',
                          compute=True, save=True, input_info_file='',
                          features_method='Bazin'):
    """Build canonical sample for SNPCC data.

    Parameters
    ----------
    path_to_raw_data: str
        Complete path to raw data directory.
    path_to_features: str
        Complete path to Bazin features files.
    output_canonical_file: str
        Complete path to output canonical sample file.
    output_info_file: str
        Complete path to output metadata file for canonical sample.
        This includes SNR and simulated peak magnitude for each filter.
    compute: bool (optional)
        If True, compute metadata information on SNR and sim peak mag.
        If False, read info from file.
        Default is True.
    save: bool (optional)
        Save simulation metadata information to file.
        Default is True.
    input_info_file: str (optional)
        Complete path to sim metadata file.
        This must be provided if save == False.
    features_method: str (optional)
        Method for feature extraction. Only 'Bazin' is implemented.

    Returns
    -------
    actsnclass.Canonical: obj
        Updated canonical object with the attribute 'canonical_sample'.
    """

    # initiate canonical object
    sample = Canonical()

    # get necessary info
    sample.snpcc_get_canonical_info(path_to_rawdata_dir=path_to_raw_data,
                                    canonical_output_file=output_info_file,
                                    canonical_input_file=input_info_file,
                                    compute=compute, save=save)

    sample.snpcc_identify_samples()                 # identify samples
    sample.find_neighbors()                         # find neighbors

    # get metadata from features file
    data = DataBase()
    data.load_features(path_to_features, method=features_method)
    sample.header = data.metadata

    # identify new samples
    for i in range(data.data.shape[0]):
        if data.data.iloc[i]['id'] in sample.canonical_ids:
            data.data.at[i, 'sample'] = 'queryable'

    # save to file
    data.data.to_csv(output_canonical_file, sep=' ', index=False)

    # update Canonical object
    sample.canonical_sample = data.data

    print('Built canonical sample!')

    return sample


def plot_snpcc_train_canonical(sample: Canonical, output_plot_file=False):
    """Plot comparison between training and canonical samples.

    Parameters
    ----------
    sample: actsnclass.Canonical
        Canonical object holding infor for canonical sample
    output_plot_file: str (optional)
        Complete path to output plot.
        If not provided, plot is displayed on screen only.
    """

    # prepare data
    ztrain = sample.meta_data['z'][sample.meta_data['sample'].values == 'train']
    ztrain_axis = np.linspace(min(ztrain) - 0.25, max(ztrain) + 0.25, 1000)[:, np.newaxis]
    kde_ztrain = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.array(ztrain).reshape(-1, 1))
    log_dens_ztrain = kde_ztrain.score_samples(ztrain_axis)

    canonical_flag = sample.canonical_sample['sample'] == 'queryable'
    zcanonical = sample.canonical_sample[canonical_flag]['redshift'].values
    zcanonical_axis = np.linspace(min(zcanonical) - 0.25, max(zcanonical) + 0.25, 1000)[:, np.newaxis]
    kde_zcanonical = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.array(zcanonical).reshape(-1, 1))
    log_dens_zcanonical = kde_zcanonical.score_samples(zcanonical_axis)

    gtrain = sample.meta_data['g_pkmag'][sample.meta_data['sample'] == 'train'].values
    gtrain_axis = np.linspace(min(gtrain) - 0.25, max(gtrain) + 0.25, 1000)[:, np.newaxis]
    kde_gtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(gtrain).reshape(-1, 1))
    log_dens_gtrain = kde_gtrain.score_samples(gtrain_axis)

    gcanonical = [sample.meta_data['g_pkmag'].values[i] for i in range(sample.meta_data.shape[0])
                  if sample.meta_data['snid'][i] in sample.canonical_ids]
    gcanonical_axis = np.linspace(min(gcanonical) - 0.25, max(gcanonical) + 0.25, 1000)[:, np.newaxis]
    kde_gcanonical = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(gcanonical).reshape(-1, 1))
    log_dens_gcanonical = kde_gcanonical.score_samples(gcanonical_axis)

    rtrain = sample.meta_data['r_pkmag'][sample.meta_data['sample'] == 'train'].values
    rtrain_axis = np.linspace(min(rtrain) - 0.25, max(rtrain) + 0.25, 1000)[:, np.newaxis]
    kde_rtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(rtrain).reshape(-1, 1))
    log_dens_rtrain = kde_rtrain.score_samples(rtrain_axis)

    rcanonical = [sample.meta_data['r_pkmag'].values[i] for i in range(sample.meta_data.shape[0])
                  if sample.meta_data['snid'][i] in sample.canonical_ids]
    rcanonical_axis = np.linspace(min(rcanonical) - 0.25, max(rcanonical) + 0.25, 1000)[:, np.newaxis]
    kde_rcanonical = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(rcanonical).reshape(-1, 1))
    log_dens_rcanonical = kde_rcanonical.score_samples(rcanonical_axis)

    plt.figure(figsize=(20, 7))
    plt.subplot(1, 3, 1)
    plt.plot(ztrain_axis, np.exp(log_dens_ztrain), label='original train')
    plt.plot(zcanonical_axis, np.exp(log_dens_zcanonical), label='canonical')
    plt.xlabel('redshift', fontsize=20)
    plt.ylabel('PDF', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)

    plt.subplot(1, 3, 2)
    plt.plot(gtrain_axis, np.exp(log_dens_gtrain), label='original train')
    plt.plot(gcanonical_axis, np.exp(log_dens_gcanonical), label='canonical')
    plt.xlabel('g_peakmag', fontsize=20)
    plt.ylabel('PDF', fontsize=20)
    plt.xlim(15, 35)
    plt.legend(loc='upper right', fontsize=15)

    plt.subplot(1, 3, 3)
    plt.plot(rtrain_axis, np.exp(log_dens_rtrain), label='original train')
    plt.plot(rcanonical_axis, np.exp(log_dens_rcanonical), label='canonical')
    plt.xlabel('r_peakmag', fontsize=20)
    plt.ylabel('PDF', fontsize=20)
    plt.xlim(15, 35)
    plt.legend(loc='upper right', fontsize=15)

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.99, wspace=0.2)

    if output_plot_file:
        plt.savefig(output_plot_file)
        plt.close('all')
    else:
        plt.show()


def main():
    return None


if __name__ == '__main__':
    main()
