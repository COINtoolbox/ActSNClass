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

import io
import numpy as np
import os
import pandas as pd
import tarfile

from actsnclass.classifiers import *

from actsnclass.query_strategies import *
from actsnclass.metrics import *


__all__ = ['DataBase']


class DataBase:
    """DataBase object, upon which the active learning loop is performed.

    Attributes
    ----------
    classprob: np.array
        Classification probability for all objects, [pIa, pnon-Ia].
    data: pd.DataFrame
        Complete information read from features files.
    features: pd.DataFrame
        Feature matrix to be used in classification (no metadata).
    features_names: list
        Header for attribute `features`.
    metadata: pd.DataFrame
        Features matrix which will not be used in classification.
    metadata_names: list
        Header for metadata.
    metrics_list_names: list
        Values for metric elements.
    plasticc_mjd_lim: list
        [min, max] mjds for plasticc data
    predicted_class: np.array
        Predicted classes - results from ML classifier.
    queried_sample: list
        Complete information of queried objects.
    queryable_ids: np.array()
        Flag for objects available to be queried.
    test_features: np.array()
        Features matrix for the test sample.
    test_metadata: pd.DataFrame
        Metadata for the test sample
    test_labels: np.array()
        True classification for the test sample.
    train_features: np.array()
        Features matrix for the train sample.
    train_metadata: pd.DataFrame
        Metadata for the training sample.
    train_labels: np.array
        Classes for the training sample.

    Methods
    -------
    build_samples(initial_training: str or int, nclass: int)
        Separate train and test samples.
    classify(method: str)
        Apply a machine learning classifier.
    evaluate_classification(metric_label: str)
        Evaluate results from classification.
    identify_keywords()
        Break degenerescency between keywords with equal meaning.
    load_bazin_features(path_to_bazin_file: str)
        Load Bazin features from file
    load_photometry_features(path_to_photometry_file:str)
        Load photometric light curves from file
    load_plasticc_mjd(path_to_data_dir: str)
        Get min and max mjds for PLAsTiCC data
    load_features(path_to_file: str, method: str)
        Load features according to the chosen feature extraction method.
    make_query(strategy: str, batch: int) -> list
        Identify new object to be added to the training sample.
    save_metrics(loop: int, output_metrics_file: str)
        Save current metrics to file.
    save_queried_sample(queried_sample_file: str, loop: int, full_sample: str)
        Save queried sample to file.
    update_samples(query_indx: list)
        Add the queried obj(s) to training and remove them from test.

    Examples
    --------
    >>> from actsnclass import DataBase

    Define the necessary paths

    >>> path_to_bazin_file = 'results/Bazin.dat'
    >>> metrics_file = 'results/metrics.dat'
    >>> query_file = 'results/query_file.dat'

    Initiate the DataBase object and load the data.
    >>> data = DataBase()
    >>> data.load_features(path_to_bazin_file, method='Bazin')

    Separate training and test samples and classify

    >>> data.build_samples(initial_training='original', nclass=2)
    >>> data.classify(method='RandomForest')
    >>> print(data.classprob)          # check predicted probabilities
    [[0.461 0.539]
    [0.346print(data.metrics_list_names)           # check metric header
    ['acc', 'eff', 'pur', 'fom']

    >>> print(data.metrics_list_values)          # check metric values
    [0.5975434599574068, 0.9024767801857585,
    0.34684684684684686, 0.13572404702012383] 0.654]
    ...
    [0.398 0.602]
    [0.396 0.604]]

    Calculate classification metrics

    >>> data.evaluate_classification(metric_label='snpcc')
    >>>

    Make query, choose object and update samples

    >>> indx = data.make_query(strategy='UncSampling', batch=1)
    >>> data.update_samples(indx)

    Save results to file

    >>> data.save_metrics(loop=0, output_metrics_file=metrics_file)
    >>> data.save_queried_sample(loop=0, queried_sample_file=query_file,
    >>>                          full_sample=False)
    """

    def __init__(self):
        self.classprob = np.array([])
        self.data = pd.DataFrame()
        self.features = pd.DataFrame([])
        self.features_names = []
        self.metadata = pd.DataFrame()
        self.metadata_names = []
        self.metrics_list_names = []
        self.metrics_list_values = []
        self.predicted_class = np.array([])
        self.queried_sample = []
        self.queryable_ids = np.array([])
        self.test_features = np.array([])
        self.test_metadata = pd.DataFrame()
        self.test_labels = np.array([])
        self.train_features = np.array([])
        self.train_metadata = pd.DataFrame()
        self.train_labels = np.array([])
        self.ensemble_probs = None

    def load_bazin_features(self, path_to_bazin_file: str, screen=False,
                            survey='DES', sample=None):
        """Load Bazin features from file.

        Populate properties: features, feature_names, metadata
        and metadata_names.

        Parameters
        ----------
        path_to_bazin_file: str
            Complete path to Bazin features file.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        survey: str (optional)
            Name of survey. Used to infer the filter set.
	    Options are DES or LSST. Default is DES.
        sample: str (optional)
            If None, sample is given by a column within the given file.
            else, read independent files for 'train' and 'test'.
            Default is None.
        """

        # read matrix with Bazin features
        if '.tar.gz' in path_to_bazin_file:
            tar = tarfile.open(path_to_bazin_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            data = pd.read_csv(io.BytesIO(content))
            tar.close()

        else:
            data = pd.read_csv(path_to_bazin_file, index_col=False)
            if ' ' in data.keys()[0]:
                data = pd.read_csv(path_to_bazin_file, sep=' ', index_col=False)

        # check if queryable is there
        if 'queryable' not in data.keys():
            data['queryable'] = [True for i in range(data.shape[0])]

        # list of features to use
        if survey == 'DES':
            self.features_names = ['gA', 'gB', 'gt0', 'gtfall', 'gtrise', 'rA',
                                   'rB', 'rt0', 'rtfall', 'rtrise', 'iA', 'iB',
                                   'it0', 'itfall', 'itrise', 'zA', 'zB', 'zt0',
                                   'ztfall', 'ztrise']

            self.metadata_names = ['id', 'redshift', 'type', 'code',
                                   'orig_sample', 'queryable']

        elif survey == 'LSST':
            self.features_names = ['uA', 'uB', 'ut0', 'utfall', 'utrise',
                                   'gA', 'gB', 'gt0', 'gtfall', 'gtrise',
                                   'rA', 'rB', 'rt0', 'rtfall', 'rtrise',
                                   'iA', 'iB', 'it0', 'itfall', 'itrise',
                                   'zA', 'zB', 'zt0', 'ztfall', 'ztrise',
                                   'YA', 'YB', 'Yt0', 'Ytfall', 'Ytrise']

            self.metadata_names = ['objid', 'redshift', 'type', 'code',
                                   'orig_sample', 'queryable']

        else:
            raise ValueError('Only "DES" and "LSST" filters are implemented at this point!')

        if sample == None:
            self.features = data[self.features_names]
            self.metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.metadata.shape[0], ' samples!')

                ntrain = sum(self.metadata['orig_sample'] == 'train')
                ntest = sum(self.metadata['orig_sample'] == 'test')
                nquery = sum(self.metadata['queryable'])

                print('   ... of which')
                print('       original train: ', ntrain)
                print('       original test: ', ntest)
                print('       query: ', nquery)

        elif sample == 'train':
            self.train_features = data[self.features_names].values
            self.train_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.train_metadata.shape[0], ' ' +  sample + ' samples!')

        elif sample == 'test':
            self.test_features = data[self.features_names].values
            self.test_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.test_metadata.shape[0], ' ' + sample +  ' samples!')


    def load_photometry_features(self, path_to_photometry_file: str,
                                 screen=False, sample=None):
        """Load photometry features from file.

        Gets as input file containing fitted flux in homogeneized cadences.
        Each line of the file is 1 object with concatenated fitted flux values.
        Populate properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_photometry_file: str
            Complete path to photometry features file.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        sample: str (optional)
            If None, sample is given by a column within the given file.
            else, read independent files for 'train' and 'test'.
            Default is None.
        """

        # read matrix with full photometry
        if '.tar.gz' in path_to_photometry_file:
            tar = tarfile.open(path_to_photometry_file, 'r:gz')
            fname = tar.getmembers()[0]
            content = tar.extractfile(fname).read()
            data = pd.read_csv(io.BytesIO(content))
            tar.close()
        else:
            data = pd.read_csv(path_to_photometry_file, index_col=False)
            if ' ' in data.keys()[0]:
                data = pd.read_csv(path_to_photometry_file, sep=' ', index_col=False)

        # list of features to use
        self.features_names = data.keys()[5:]

        if 'objid' in data.keys():
            id_name = 'objid'
        elif 'id' in data.keys():
            id_name = 'id'

        self.metadata_names = [id_name, 'redshift', 'type', 'code', 'orig_sample']

        if sample == None:
            self.features = data[self.features_names]
            self.metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.metadata.shape[0], ' samples!')

        elif sample == 'train':
            self.train_features = data[self.features_names].values
            self.train_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.train_metadata.shape[0], ' samples!')

        elif sample == 'test':
            self.test_features = data[self.features_names].values
            self.test_metadata = data[self.metadata_names]

            if screen:
                print('Loaded ', self.test_metadata.shape[0], ' samples!')


    def load_features(self, path_to_file: str, method='Bazin', screen=False,
                      survey='DES', sample=None ):
        """Load features according to the chosen feature extraction method.

        Populates properties: data, features, feature_list, header
        and header_list.

        Parameters
        ----------
        path_to_file: str
            Complete path to features file.
        method: str (optional)
            Feature extraction method. The current implementation only
            accepts method=='Bazin' or 'photometry'.
            Default is 'Bazin'.
        screen: bool (optional)
            If True, print on screen number of light curves processed.
            Default is False.
        survey: str (optional)
            Survey used to obtain the data. The current implementation
            only accepts survey='DES' or 'LSST'.
            Default is 'DES'.
        sample: str (optional)
            If None, sample is given by a column within the given file.
            else, read independent files for 'train' and 'test'.
            Default is None.
        """

        if method == 'Bazin':
            self.load_bazin_features(path_to_file, screen=screen,
                                     survey=survey, sample=sample)

        elif method == 'photometry':
            self.load_photometry_features(path_to_file, screen=screen,
                                          survey=survey, sample=sample)
        else:
            raise ValueError('Only Bazin and photometry features are implemented!'
                             '\n Feel free to add other options.')

    def load_plasticc_mjd(self, path_to_data_dir):
        """Return all MJDs from 1 file from PLAsTiCC simulations.

        Parameters
        ----------
        path_to_data_dir: str
            Complete path to PLAsTiCC data directory.
        """

        # list of PLAsTiCC photometric files
        flist = ['plasticc_test_lightcurves_' + str(x).zfill(2) + '.csv.gz'
                  for x in range(1, 12)]

        # add training file
        flist = flist + ['plasticc_train_lightcurves.csv.gz']

        # store max and min mjds
        min_mjd = []
        max_mjd = []

        for fname in flist:
        # read photometric points
            if '.tar.gz' in fname:
                tar = tarfile.open(path_to_data_dir + fname, 'r:gz')
                name = tar.getmembers()[0]
                content = tar.extractfile(name).read()
                all_photo = pd.read_csv(io.BytesIO(content))
            else:
                all_photo = pd.read_csv(path_to_data_dir + fname, index_col=False)

                if ' ' in all_photo.keys()[0]:
                    all_photo = pd.read_csv(path_to_data_dir + fname, sep=' ',
                                            index_col=False)

            # get limit mjds
            min_mjd.append(min(all_photo['mjd']))
            max_mjd.append(max(all_photo['mjd']))


        self.plasticc_mjd_lim = [min(min_mjd), max(max_mjd)]

    def identify_keywords(self):
        """Break degenerescency between keywords with equal meaning.

        Returns
        -------

        id_name: str
            String of object identification.
        """

        if 'id' in self.metadata_names:
            id_name = 'id'
        elif 'objid' in self.metadata_names:
            id_name = 'objid'

        return id_name

    def build_orig_samples(self, nclass=2, screen=False, queryable=False,
                           sep_files=False):
        """Construct train and test samples as given in the original data set.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        queryable: bool (optional)
            If True build also queryable sample for time domain analysis.
            Default is False.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        sep_files: bool (optional)
            If True, consider train and test samples separately read
            from independent files.
        """

        # object if keyword
        id_name = self.identify_keywords()

        if sep_files:
            # get samples labels in a separate object
            train_labels = self.train_metadata['type'].values == 'Ia'
            self.train_labels = train_labels.astype(int)

            test_labels = self.test_metadata['type'].values == 'Ia'
            self.test_labels = test_labels.astype(int)

            # identify queryable objects
            if 'queryable' in self.test_metadata['orig_sample'].values:
                queryable_flag = self.test_metadata['orig_sample'] == 'queryable'
                self.queryable_ids = self.test_metadata[queryable_flag][id_name].values

            else:
                self.queryable_ids = self.test_metadata[id_name].values

            # build complete metadata object
            self.metadata = pd.concat([self.train_metadata, self.test_metadata])

        else:
            train_flag = self.metadata['orig_sample'] == 'train'
            train_data = self.features[train_flag]
            self.train_features = train_data.values
            self.train_metadata = self.metadata[train_flag]

            test_flag = self.metadata['orig_sample'] == 'test'

            test_data = self.features[test_flag]
            self.test_features = test_data.values
            self.test_metadata = self.metadata[test_flag]

            if queryable:
                queryable_flag = self.test_metadata['queryable'].values
                self.queryable_ids = self.test_metadata[queryable_flag][id_name].values
            else:
                self.queryable_ids = self.test_metadata[id_name].values

            if nclass == 2:
                train_ia_flag = self.train_metadata['type'] == 'Ia'
                self.train_labels = np.array([int(item) for item in train_ia_flag])

                test_ia_flag = self.test_metadata['type'] == 'Ia'
                self.test_labels = np.array([int(item) for item in test_ia_flag])
            else:
                raise ValueError("Only 'Ia x non-Ia' are implemented! "
                                 "\n Feel free to add other options.")

    def build_random_training(self, initial_training: int, nclass=2, screen=False,
                              Ia_frac=0.5, queryable=True, sep_files=False):
        """Construct initial random training and corresponding test sample.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        initial_training : int
            Required number of samples at random
            Default is 10.
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        queryable: bool (optional)
            If True build also queryable sample for time domain analysis.
            Default is False.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        Ia_frac: float in [0,1] (optional)
            Fraction of Ia required in initial training sample.
            Default is 0.5.
        sep_files: bool (optional)
            If True, consider train and test samples separately read
            from independent files.
        """

        # object if keyword
        id_name = self.identify_keywords()

        if sep_files:
            # build complete metadata object
            self.metadata = pd.concat([self.train_metadata, self.test_metadata])
            self.features = np.concatenate((self.train_features, self.test_features))

        # identify Ia
        data_copy = self.metadata.copy()
        ia_flag = data_copy['type'] == 'Ia'

        # separate per class
        Ia_data = data_copy[ia_flag]
        nonIa_data = data_copy[~ia_flag]

        # get subsamples for training
        temp_train_ia = Ia_data.sample(n=int(Ia_frac * initial_training))
        temp_train_nonia = nonIa_data.sample(n=int((1-Ia_frac)*initial_training))

        # join classes
        frames_train = [temp_train_ia, temp_train_nonia]
        temp_train = pd.concat(frames_train, ignore_index=True)
        train_flag = np.array([data_copy[id_name].values[i] in temp_train[id_name].values
                               for i in range(data_copy.shape[0])])

        self.train_metadata = data_copy[train_flag]
        self.train_features = self.features[train_flag].values

        # get test sample
        self.test_metadata = data_copy[~train_flag]
        self.test_features = self.features[~train_flag].values

        if nclass == 2:
            self.train_labels = data_copy['type'][train_flag].values == 'Ia'
            self.test_labels = data_copy['type'][~train_flag].values == 'Ia'
        else:
            raise ValueError("Only 'Ia x non-Ia' are implemented! "
                             "\n Feel free to add other options.")

        if queryable:
            queryable_flag = data_copy['queryable'].values
            combined_flag = np.logical_and(~train_flag, queryable_flag)
            self.queryable_ids = data_copy[combined_flag][id_name].values
        else:
            self.queryable_ids = self.test_metadata[id_name].values

        # check if there are repeated ids
        train_test_flag = np.array([True if item in self.test_metadata[id_name].values
                                    else False for item in
                                    self.train_metadata[id_name].values])

        if sum(train_test_flag) > 0:
            raise ValueError('There are repeated ids!!')

        test_train_flag = np.array([True if item in self.train_metadata[id_name].values
                                    else False for item in
                                    self.test_metadata[id_name].values])

        if sum(test_train_flag) > 0:
            raise ValueError('There are repeated ids!!')

    def build_previous_runs(self, path_to_train: str,
                            path_to_queried: str, nclass=2,
                            sep_files=False, queryable=False):
        """Build train, test and queryable samples from previous runs.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        path_to_train: str
            Full path to initial training sample file from a previous run.
        path_to_queried: str
            Full path to queried sample file from a previous run.
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        queryable: bool (optional)
            If True build also queryable sample for time domain analysis.
            Default is False.
        sep_files: bool (optional)
            If True, consider train and test samples separately read
            from independent files. Default is False.
        """

        # read initial training data from a previous run
        train_previous = pd.read_csv(path_to_train, index_col=False, sep=' ')

        # read all queried objects in previous loops
        queried_previous = pd.read_csv(path_to_queried, index_col=False,
                                       sep=' ')

        for i in range(queried_previous.shape[0]):
            self.queried_sample.append(queried_previous.iloc[i].values[:-1])

        # object if keyword
        id_name = self.identify_keywords()

        # join train and queried data
        train_ids = list(train_previous[id_name].values) + \
                    list(queried_previous[id_name].values)

        if sep_files:
            # build complete metadata object
            self.metadata = pd.concat([self.train_metadata, self.test_metadata])
            self.features = np.concatenate((self.train_features, self.test_features))

        # make a copy of the metadata to avoid warnings
        data_copy = self.metadata.copy()

        # identify training sample
        train_flag = np.array([data_copy[id_name].values[i] in train_ids
                               for i in range(data_copy.shape[0])])

        # populate sample properties
        self.train_metadata = data_copy[train_flag]
        self.train_features = self.features[train_flag].values
        self.test_metadata = data_copy[~train_flag]
        self.test_features = self.features[~train_flag].values

        if queryable:
            queryable_flag = self.test_metadata['queryable'].values
            self.queryable_ids = self.test_metadata[queryable_flag][id_name].values
        else:
            self.queryable_ids = self.test_metadata[id_name].values

        if nclass == 2:
            train_ia_flag = self.train_metadata['type'] == 'Ia'
            self.train_labels = np.array([int(item) for item in train_ia_flag])

            test_ia_flag = self.test_metadata['type'] == 'Ia'
            self.test_labels = np.array([int(item) for item in test_ia_flag])
        else:
            raise ValueError("Only 'Ia x non-Ia' are implemented! "
                             "\n Feel free to add other options.")

    def build_samples(self, initial_training='original', nclass=2,
                      screen=False, Ia_frac=0.5,
                      queryable=False, save_samples=False, sep_files=False,
                      survey='DES', output_fname=' ', path_to_train=' ',
                      path_to_queried=' ', method='Bazin'):
        """Separate train and test samples.

        Populate properties: train_features, train_header, test_features,
        test_header, queryable_ids (if flag available), train_labels and
        test_labels.

        Parameters
        ----------
        initial_training : str or int
            Choice of initial training sample.
            If 'original': begin from the train sample flagged in original file
            elif 'previous': continue from a previously run loop
            elif int: choose the required number of samples at random,
            ensuring that at least half are SN Ia.
        Ia_frac: float in [0,1] (optional)
            Fraction of Ia required in initial training sample.
            Default is 0.5.
        method: str (optional)
            Feature extraction method. The current implementation only
            accepts method=='Bazin' or 'photometry'.
            Default is 'Bazin'. Only used if initial_training == 'previous'.
        nclass: int (optional)
            Number of classes to consider in the classification
            Currently only nclass == 2 is implemented.
        path_to_train: str (optional)
            Path to initial training file from previous run.
            Only used if initial_training == 'previous'.
        path_to_queried: str(optional)
            Path to queried sample from previous run.
            Only used if initial_training == 'previous'.
        queryable: bool (optional)
            If True build also queryable sample for time domain analysis.
            Default is False.
        screen: bool (optional)
            If True display the dimensions of training and test samples.
        save_samples: bool (optional)
            If True, save training and test samples to file.
            Default is False.
        survey: str (optional)
            Survey used to obtain the data. The current implementation
            only accepts survey='DES' or 'LSST'.
            Default is 'DES'.
        sep_files: bool (optional)
            If True, consider train and test samples separately read
            from independent files. Default is False.
        output_fname: str (optional)
            Complete path to output file where initial training will be stored.
            Only used if save_samples == True.
        """

        if initial_training == 'original':
            self.build_orig_samples(nclass=nclass, screen=screen,
                                    queryable=queryable, sep_files=sep_files)

        elif initial_training == 'previous':
            self.build_previous_runs(path_to_train=path_to_train,
                                    path_to_queried=path_to_queried,
                                    sep_files=sep_files, nclass=nclass)

        elif isinstance(initial_training, int):
            self.build_random_training(initial_training=initial_training,
                                       nclass=nclass, screen=screen,
                                       Ia_frac=Ia_frac, queryable=queryable,
                                       sep_files=sep_files)

        if screen:
            print('Training set size: ', self.train_metadata.shape[0])
            print('Test set size: ', self.test_metadata.shape[0])
            print('Queryable set size: ', sum(self.metadata['queryable']))

        if save_samples:

            full_header = self.metadata_names + self.features_names
            wsample = open(output_fname, 'w')
            for item in full_header:
                wsample.write(item + ' ')
            wsample.write('\n')

            for j in range(self.train_metadata.shape[0]):
                for name in self.metadata_names:
                    wsample.write(str(self.train_metadata[name].iloc[j]) + ' ')
                for k in range(self.train_features.shape[1] - 1):
                    wsample.write(str(self.train_features[j][k]) + ' ')
                wsample.write(str(self.train_features[j][-1]) + '\n')
            wsample.close()

    def classify(self, method: str, **kwargs):
        """Apply a machine learning classifier.

        Populate properties: predicted_class and class_prob

        Parameters
        ----------
        method: str
            Chosen classifier.
            The current implementation accepts `RandomForest`,
            'GradientBoostedTrees', 'KNN', 'MLP' and 'NB'.
        kwargs: extra parameters
            Parameters required by the chosen classifier.
        """

        if method == 'RandomForest':
            self.predicted_class,  self.classprob = \
                random_forest(self.train_features, self.train_labels,
                              self.test_features, **kwargs)
        elif method == 'GradientBoostedTrees':
            self.predicted_class,  self.classprob = \
                gradient_boosted_trees(self.train_features, self.train_labels,
                                       self.test_features, **kwargs)
        elif method == 'KNN':
            self.predicted_class,  self.classprob = \
                knn(self.train_features, self.train_labels,
                               self.test_features, **kwargs)
        elif method == 'MLP':
            self.predicted_class,  self.classprob = \
                mlp(self.train_features, self.train_labels,
                               self.test_features, **kwargs)
        elif method == 'SVM':
            self.predicted_class, self.classprob = \
                svm(self.train_features, self.train_labels,
                               self.test_features, **kwargs)
        elif method == 'NB':
            self.predicted_class, self.classprob = \
                nbg(self.train_features, self.train_labels,
                          self.test_features, **kwargs)


        else:
            raise ValueError("The only classifiers implemented are" +
                              "'RandomForest', 'GradientBoostedTrees'," +
                              "'KNN', 'MLP' and NB'." +
                             "\n Feel free to add other options.")

    def classify_bootstrap(self, method: str, **kwargs):
        """Apply a machine learning classifier bootstrapping the classifier.

        Populate properties: predicted_class and class_prob

        Parameters
        ----------
        method: str
            Chosen classifier.
            The current implementation accepts `RandomForest`,
            'GradientBoostedTrees', 'KNN', 'MLP' and 'NB'.
        kwargs: extra parameters
            Parameters required by the chosen classifier.
        """
        n_ensembles = 10
        if method == 'RandomForest':
            self.predicted_class, self.class_prob, self.ensemble_probs = \
            bootstrap_clf(random_forest, n_ensembles,
                          self.train_features, self.train_labels,
                          self.test_features, **kwargs)

        elif method == 'GradientBoostedTrees':
            self.predicted_class, self.class_prob, self.ensemble_probs = \
            bootstrap_clf(gradient_boosted_trees, n_ensembles,
                          self.train_features, self.train_labels,
                          self.test_features, **kwargs)
        elif method == 'KNN':
            self.predicted_class, self.class_prob, self.ensemble_probs = \
            bootstrap_clf(knn, n_ensembles,
                          self.train_features, self.train_labels,
                          self.test_features, **kwargs)
        elif method == 'MLP':
            self.predicted_class, self.class_prob, self.ensemble_probs = \
            bootstrap_clf(mlp, n_ensembles,
                          self.train_features, self.train_labels,
                          self.test_features, **kwargs)
        elif method == 'SVM':
            self.predicted_class, self.class_prob, self.ensemble_probs = \
            bootstrap_clf(svm, n_ensembles,
                          self.train_features, self.train_labels,
                          self.test_features, **kwargs)
        elif method == 'NB':
            self.predicted_class, self.class_prob, self.ensemble_probs = \
            bootstrap_clf(nbg, n_ensembles,
                          self.train_features, self.train_labels,
                          self.test_features, **kwargs)


        else:
            raise ValueError("The only classifiers implemented are" +
                              "'RandomForest', 'GradientBoostedTrees'," +
                              "'KNN', 'MLP' and NB'." +
                             "\n Feel free to add other options.")

    def evaluate_classification(self, metric_label='snpcc'):
        """Evaluate results from classification.

        Populate properties: metric_list_names and metrics_list_values.

        Parameters
        ----------
        metric_label: str
            Choice of metric. Currenlty only `snpcc` is accepted.
        """

        if metric_label == 'snpcc':
            self.metrics_list_names, self.metrics_list_values = \
                get_snpcc_metric(list(self.predicted_class),
                                 list(self.test_labels))
        else:
            raise ValueError('Only snpcc metric is implemented!'
                             '\n Feel free to add other options.')

    def make_query(self, strategy='UncSampling', batch=1,
                   screen=False, queryable=False, query_thre=1.0) -> list:
        """Identify new object to be added to the training sample.

        Parameters
        ----------
        strategy: str (optional)
            Strategy used to choose the most informative object.
            Current implementation accepts 'UncSampling' and
            'RandomSampling'. Default is `UncSampling`.

        batch: int (optional)
            Number of objects to be chosen in each batch query.
            Default is 1.
        queryable: bool (optional)
            If True, consider only queryable objects.
            Default is False.
        query_thre: float (optional)
            Percentile threshold where a query is considered worth it.
            Default is 1 (no limit).
        screen: bool (optional)
            If true, display on screen information about the
            displacement in order and classificaion probability due to
            constraints on queryable sample.

        Returns
        -------
        query_indx: list
            List of indexes identifying the objects to be queried in decreasing
            order of importance.
            If strategy=='RandomSampling' the order is irrelevant.
        """

        if 'objid' in self.test_metadata.keys():
            id_name = 'objid'
        elif 'id' in self.test_metadata.keys():
            id_name = 'id'

        if strategy == 'UncSampling':
            query_indx = uncertainty_sampling(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)
            return query_indx
        elif strategy == 'UncSamplingEntropy':
            query_indx = uncertainty_sampling_entropy(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)
            return query_indx
        elif strategy == 'UncSamplingLeastConfident':
            query_indx = uncertainty_sampling_least_confident(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)
            return query_indx
        elif strategy == 'UncSamplingMargin':
            query_indx = uncertainty_sampling_margin(class_prob=self.classprob,
                                              queryable_ids=self.queryable_ids,
                                              test_ids=self.test_metadata[id_name].values,
                                              batch=batch, screen=screen,
                                              query_thre=query_thre)
            return query_indx
        elif strategy == 'QBDMI':
            query_indx = qbd_mi(ensemble_probs=self.ensemble_probs,
                                queryable_ids=self.queryable_ids,
                                test_ids=self.test_metadata[id_name].values,
                                batch=batch, screen=screen,
                                query_thre=query_thre)
            return query_indx
        elif strategy =='QBDEntropy':
            query_indx = qbd_entropy(ensemble_probs=self.ensemble_probs,
                                    queryable_ids=self.queryable_ids,
                                    test_ids=self.test_metadata[id_name].values,
                                    batch=batch, screen=screen,
                                    query_thre=query_thre)
            return query_indx
        elif strategy == 'RandomSampling':
            query_indx = random_sampling(queryable_ids=self.queryable_ids,
                                         test_ids=self.test_metadata[id_name].values,
                                         queryable=queryable, batch=batch,
                                         query_thre=query_thre)

            for n in query_indx:
                if self.test_metadata[id_name].values[n] not in self.queryable_ids:
                    raise ValueError('Chosen object is not available for query!')

            return query_indx

        else:
            raise ValueError('Invalid strategy. Only "UncSampling" and '
                             '"RandomSampling are implemented! \n '
                             'Feel free to add other options. ')

    def update_samples(self, query_indx: list, loop: int, epoch=0):
        """Add the queried obj(s) to training and remove them from test.

        Update properties: train_headers, train_features, train_labels,
        test_labels, test_headers and test_features.

        Parameters
        ----------
        query_indx: list
            List of indexes identifying objects to be moved.
        loop: int
            Store number of loop when this query was made.
        """

        if 'id' in self.train_metadata.keys():
            id_name = 'id'
        elif 'objid' in self.train_metadata.keys():
            id_name = 'objid'

        all_queries = []

        while len(query_indx) > 0 and self.test_metadata.shape[0] > 0:

            # identify queried object index
            obj = query_indx[0]

            # add object to the query sample
            query_header = self.test_metadata.values[obj]
            query_features = self.test_features[obj]
            line = [epoch]
            for item in query_header:
                line.append(item)
            for item1 in query_features:
                line.append(item1)

            # add object to the training sample
            new_header = pd.DataFrame([query_header], columns=self.metadata_names)
            self.train_metadata = pd.concat([self.train_metadata, new_header], axis=0,
                                            ignore_index=True)
            self.train_features = np.append(self.train_features,
                                            np.array([self.test_features[obj]]),
                                            axis=0)
            self.train_labels = np.append(self.train_labels,
                                          np.array([self.test_labels[obj]]),
                                          axis=0)

            # remove queried object from test sample
            query_flag = self.test_metadata[id_name].values == self.test_metadata[id_name].iloc[obj]
            test_metadata_temp = self.test_metadata.copy()
            self.test_metadata = test_metadata_temp[~query_flag]
            self.test_labels = np.delete(self.test_labels, obj, axis=0)
            self.test_features = np.delete(self.test_features, obj, axis=0)
            all_queries.append(line)

            # update ids order
            query_indx.remove(obj)

            new_query_indx = []

            for item in query_indx:
                if item < obj:
                    new_query_indx.append(item)
                else:
                    new_query_indx.append(item - 1)

            query_indx = new_query_indx

        # update queried samples
        self.queried_sample.append(all_queries)

    def save_metrics(self, loop: int, output_metrics_file: str, epoch: int, batch=1):
        """Save current metrics to file.

        If loop == 0 the 'output_metrics_file' will be created or overwritten.
        Otherwise results will be added to an existing 'output_metrics file'.

        Parameters
        ----------
        loop: int
            Number of learning loops finished at this stage.
        output_metrics_file: str
            Full path to file to store metrics results.
        batch: int
            Number of queries in each loop.
        epoch: int
            Days since the beginning of the survey.
        """

        # add header to metrics file
        if not os.path.exists(output_metrics_file) or loop == 0:
            with open(output_metrics_file, 'w') as metrics:
                metrics.write('loop ')
                for name in self.metrics_list_names:
                    metrics.write(name + ' ')
                for j in range(batch):
                    metrics.write('query_id' + str(j + 1) + ' ')
                metrics.write('\n')

        # write to file
        if len(self.queried_sample[loop]) > 0:
            with open(output_metrics_file, 'a') as metrics:
                metrics.write(str(epoch) + ' ')
                for value in self.metrics_list_values:
                    metrics.write(str(value) + ' ')
                for j in range(len(self.queried_sample[loop])):
                    metrics.write(str(self.queried_sample[loop][j][1]) + ' ')
                metrics.write('\n')

    def save_queried_sample(self, queried_sample_file: str, loop: int,
                            full_sample=False, batch=1):
        """Save queried sample to file.

        Parameters
        ----------
        queried_sample_file: str
            Complete path to output file.
        loop: int
            Number of learning loops finished at this stage.
        full_sample: bool (optional)
            If true, write down a complete queried sample stored in
            property 'queried_sample'. Otherwise append 1 line per loop to
            'queried_sample_file'. Default is False.
        """

        if full_sample and len(self.queried_sample[loop]) > 0:
            full_header = self.metadata_names + self.features_names
            query_sample = pd.DataFrame(self.queried_sample, columns=full_header)
            query_sample.to_csv(queried_sample_file, sep=' ', index=False)

        elif isinstance(loop, int) and len(self.queried_sample[loop]) > 0:
            if not os.path.exists(queried_sample_file) or loop == 0:
                # add header to query sample file
                full_header = self.metadata_names + self.features_names
                with open(queried_sample_file, 'w') as query:
                    query.write('day ')
                    for item in full_header:
                        query.write(item + ' ')
                    query.write('\n')

            # save query sample to file
            with open(queried_sample_file, 'a') as query:
                for batch in range(batch):
                    for elem in self.queried_sample[loop][batch]:
                        query.write(str(elem) + ' ')
                    query.write('\n')


def main():
    return None


if __name__ == '__main__':
    main()
