# Copyright 2020 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 26 February 2020
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

import os

from copy import deepcopy
import numpy as np
import pandas as pd

from actsnclass import LightCurve


class PLAsTiCCPhotometry(object):
    """Handles photometric information for the PLAsTiCC data.

    Attributes
    ----------
    bazin_header: str
        Header to be added to features files for each day.
    class_code: dict
        Keywords are model numbers, values are model names. 
    fdic: dict
        Keywords: ['test', 'train'], values are names of photo files.
    include_classes: list
        List of classes to be considered.
    max_epoch: float
        Maximum MJD for the entire data set.
    metadata: dict
        Keywords: ['test', 'train'], values are metadata.
    min_epoch: float
        Minimum MJD for the entire data set.
    rmag_lim: float
        Maximum r-band magnitude allowing a query. 
    
    Methods
    -------
    build_one_epoch(raw_data_dir: str, day_of_survey: int,
                    time_domain_dir: str, feature_method: str)
        Construct 1 entire day of observation.
    create_all_daily_files(raw_data_dir: str)
        Create 1 file per day for all days of the survey. 
    create_daily_file(output_dir: str, day: int, vol: int, header: str)
        Create one file for a given day of the survey. Contains only header.    
    fit_one_lc(raw_data_dir: str, snid: int, sample: str)
        Fit one light curve throughout the entire survey.
    read_metadata(path_to_data_dir: str, classes: list)
        Read metadata and filter only required classes.
    write_bazin_to_file(lightcurve: LightCurve, features_file: str)
        Write Bazin parameters and metadata to file.
    """

    def __init__(self):
        self.bazin_header = 'id redshift type code sample queryable uA uB ut0 ' + \
                            'utfall utrise gA gB gt0 ' + \
                            'gtfall gtrise rA rB rt0 rtfall rtrise iA ' + \
                            'iB it0 itfall itrise zA zB zt0 ztfall ztrise' + \
                            'YA YB Yt0 Ytfall Ytrise\n'
        self.class_code = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
             95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
             16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
             993:'CART', 994:'PISN',995:'MLString'}
        self.fdic = {}
        self.fdic['test'] = ['plasticc_test_lightcurves_' + str(x).zfill(2) + '.csv.gz' 
                              for x in range(1, 12)]
        self.fdic['train'] = ['plasticc_train_lightcurves.csv.gz']
        self.include_classes = [] 
        self.max_epoch = 60675
        self.metadata = {}
        self.min_epoch = 59580
        self.rmag_lim = 24
        
    def create_daily_file(self, output_dir: str,
                          day: int, vol: int, header='Bazin'):
        """Create one file for a given day of the survey.

        The file contains only header for the features file.

        Parameters
        ----------
        output_dir: str
            Complete path to raw data directory.
        day: int
            Day passed since the beginning of the survey.
        vol: int
            Index to identify when there is more than 1 file per day.
        header: str (optional)
            List of elements to be added to the header.
            Separate by 1 space.
            Default option uses header for Bazin features file.
        """

        features_file = output_dir + 'day_' + str(day) + '_v' + str(vol) +'.dat'

        if header == 'Bazin':
            # add headers to files
            with open(features_file, 'w') as param_file:
                param_file.write(self.bazin_header)

        else:
            with open(features_file, 'w') as param_file:
                param_file.write(self.header)


    def read_metadata(self, path_to_data_dir: str, classes: list):
        """ Read metadata and filter only required classes.

        Populate the metadata attribute. 

        Parameters
        ----------
        path_to_data_dir: str
            Directory containing all PlAsTiCC files. 
        classes: list of int
            Codes for classes we wish to keep. 

        """
        flist = ['plasticc_train_metadata.csv.gz', 
                  'plasticc_test_metadata.csv.gz']

        # store classes information
        self.include_classes = classes

        # read header information
        for i in range(2):
            if '.tar.gz' in flist[i]:
                tar = tarfile.open(path_to_data_dir + flist[i], 'r:gz')
                name = tar.getmembers()[0]
                content = tar.extracfile(name).read()
                header = pd.read_csv(io.BytesIO(content))
                tar.close()
            else:
                header = pd.read_csv(path_to_data_dir + flist[i], 
                                     index_col=False)
                if ' ' in header.keys()[0]:
                    header = pd.read_csv(path_to_data_dir + flist[i], 
                                         sep=' ', index_col=False)
            
            # remove useless columns
            header = header[['object_id', 'true_target', 'true_z',
                             'true_peakmjd', 'true_distmod']]

            class_flag = np.array([header['true_target'].iloc[j] in classes
                                   for j in range(header.shape[0])])

            if i == 0:
                self.metadata['train'] = header[class_flag]
            else:
                self.metadata['test'] = header[class_flag]

    def create_all_daily_files(self, raw_data_dir: str, output_dir:str):
        """Create 1 file per day for all days of the survey. 
        
        Each file contains only the header. 

        Parameters
        ----------
        raw_data_dir: str
            Path to directory containing all PLAsTiCC zenodo data.
        output_dir: str
            Output directory.
        """
        
        ## create daily files
        # run through training and test
        for key in ['train', 'test']:

            # run through multiple photometric files
            for j in range(len(self.fdic[key])):

                # identify number of photometric test files
                if key == 'train':
                    vol = j
                else:
                    vol = j + 1

                # run through all days of the survey
                for day_of_survey in range(1, self.max_epoch - self.min_epoch):
                    self.create_daily_file(output_dir=output_dir,
                                           day=day_of_survey, vol=vol)

    def write_bazin_to_file(self, lc: LightCurve, 
                            features_file: str, queryable: bool):
        """Write Bazin parameters and metadata to file.

        Use output filename defined in the features_file attribute.

        Parameters
        ----------
        lc: LightCurve
            ActSNClass light curve object.
        features_file: str
            Output file to store Bazin features.
        queryable: bool
            Rather this object is available for querying.
        
        Returns
        -------
        line: str
            A line concatenating metadata and Bazin fits for 1 obj.
        """

        # build an entire line with bazin features
        line = str(lc.id) + ' ' + str(lc.redshift) + ' ' + \
               str(lc.sntype) + ' ' + str(lc.sncode) + ' ' + \
               str(lc.sample) + ' ' + str(queryable) + ' '

        for item in lc.bazin_features[:-1]:
            line = line + str(item) + ' '

        line = line + str(lc.bazin_features[-1]) + '\n'
        
        # save features to file
        with open(features_file, 'a') as param_file:
            param_file.write(line)

        return line
            
    def fit_one_lc(self, raw_data_dir: str, snid: int, sample: str,
                   output_dir: str, vol: int, day=None, screen=False):
        """Fit one light curve throughout the entire survey.

        Save results to appropriate file, considering 1 day survey 
        evolution. 

        Parameters
        ----------
        raw_data_dir: str
            Complete path to all PLAsTiCC zenodo files.
        snid: int
            Object id for the transient to be fitted.
        sample: str
            Original sample this object belonged to.
            Possibilities are 'train' or 'test'.
        output_dir:
            Directory to store output time domain files.
        vol: int or None
            Index of the original PLAsTiCC zenodo light curve
            files where the photometry for this object is stored.
            If None, search for id in all light curve files.
            If sample == 'train' it is automatically set to 0.
        day: int or None (optional)
            Day since beginning of survey to be considered.
            If None, fit all days. Default is None.
        screen: bool (optional)
            Print steps evolution on screen.
            Default is False.
        """

        # store number of points per day
        npoints = {}
        npoints[0] = 0

        # create light curve instance
        orig_lc = LightCurve()          

        if sample == 'train':
            vol = 0
            # load light curve
            if screen:
                print('vol: ', vol)

            orig_lc.load_plasticc_lc(raw_data_dir + self.fdic[sample][vol], snid)

        elif vol ==  None:
            # search within test light curve files
            while orig_lc.photometry.shape[0] == 0 and vol < 11:
                vol = vol + 1
                if screen:
                    print('vol: ', vol)
                    
                orig_lc.load_plasticc_lc(raw_data_dir + self.fdic[sample][vol - 1], snid)

        elif isinstance(vol, int):
            # load light curve
            orig_lc.load_plasticc_lc(raw_data_dir + self.fdic[sample][vol - 1], snid)

        # get number of points in all days of the survey for this light curve
        for days in range(1, self.max_epoch - self.min_epoch):

            # see which epochs are observed until this day
            now = days + self.min_epoch
            photo_flag_now = orig_lc.photometry['mjd'].values <= now

            # number of points today
            npoints[days] = sum(photo_flag_now)
            
        # create instance to store Bazin fit parameters    
        line = False

        if day == None:
            # for every day of survey
            fit_days = range(42, self.max_epoch - self.min_epoch)
            
        elif isinstance(day, int):
            # for a specific day of the survey
            fit_days = [day]
            
        for day_of_survey in fit_days:

            lc = deepcopy(orig_lc)

            # see which epochs are observed until this day
            today = day_of_survey + self.min_epoch
            photo_flag = lc.photometry['mjd'].values <= today

            # number of points today
            npoints[day_of_survey] = sum(photo_flag)

            # only the allowed photometry
            lc.photometry = lc.photometry[photo_flag]                                  

            # check if any point survived, other checks are made
            # inside lc object
            if npoints[day_of_survey] > 4 and \
               npoints[day_of_survey] - npoints[day_of_survey - 1] > 0:

                # perform feature extraction
                lc.fit_bazin_all()
   
                # only save to file if all filters were fitted
                if len(lc.bazin_features) > 0 and \
                        'None' not in lc.bazin_features:

                    # see if query is possible
                    queryable = \
                        lc.check_queryable(mjd=self.min_epoch + day_of_survey,
                                           r_lim=self.rmag_lim)              

                    # mask only metadata for this object
                    mask = self.metadata[sample]['object_id'].values == snid
                    
                    # set redshift
                    lc.redshift = self.metadata[sample]['true_z'].values[mask][0]

                    # set light curve type
                    lc.sncode  = self.metadata[sample]['true_target'].values[mask][0]
                    lc.sntype = self.class_code[lc.sncode]

                    # set id
                    lc.id = snid

                    # set sample
                    lc.sample = sample

                    # set filename
                    features_file = output_dir + 'day_' + \
                                         str(day_of_survey) + '_v' + str(vol) +'.dat'

                    # write to file
                    line = self.write_bazin_to_file(lc, features_file, queryable)

                    if screen:
                        print('   *** Wrote to file ***   ')
                    
            # write previous result to file if there is no change or if it is the first day            
            elif (npoints[day_of_survey] == npoints[day_of_survey - 1] and isinstance(line, str)):

                # set filename
                features_file = output_dir + 'day_' + \
                                str(day_of_survey) + '_v' + str(vol) +'.dat'

                # save results
                with open(features_file, 'a') as param_file:
                    param_file.write(line)

                if screen:
                    print('   *** wrote to file ***   ')
                
    def build_one_epoch(self, raw_data_dir: str, day_of_survey: int,
                        time_domain_dir: str, feature_method='Bazin'):
        """Construct 1 entire day of observation. 

        Save results to file.

        Parameters
        ----------
        raw_data_dir: str
            Directory containing all the PLAsTiCC zenodo data.
        day_of_survey: int
            Day since the beginning of the survey to be constructed.
        time_domain_dir: str
            Output directory
        feature_method: str
            Feature extraction method. 
            Only possibility now is 'Bazin'.
        """
        
        # count survivers
        count_surv = 0

        # run through training and test
        for key in ['train', 'test']:

            # run through multiple photometric files
            for j in range(len(self.fdic[key])):

                # create features file
                if key == 'train':
                    vol = j
                else:
                    vol = j + 1
                    
                self.create_daily_file(output_dir=time_domain_dir,
                                       day=day_of_survey, vol=vol)

                features_file = time_domain_dir + 'day_' + str(day_of_survey) + \
                                '_v' + str(vol) +'.dat'
                
                for i in range(self.metadata[key].shape[0]):

                    # choose 1 snid
                    snid =  self.metadata[key]['object_id'].iloc[i]
 
                    lc = LightCurve()  # create light curve instance

                    # define original sample
                    lc.sample = key
  
                    if dataset == 'PLAsTiCC':
                        lc.load_plasticc_lc(raw_data_dir + self.fdic[key][j], snid)
                    else:
                        raise ValueError('Only PLAsTiCC data set is ' + 
                                         'implemented in this module!')

                    # see which epochs are observed until this day
                    today = day_of_survey + self.min_epoch
                    photo_flag = lc.photometry['mjd'].values <= today

                    # check if any point survived, other checks are made
                    # inside lc object
                    if sum(photo_flag) > 4:
                        # only the allowed photometry
                        lc.photometry = lc.photometry[photo_flag]

                        # perform feature extraction
                        if feature_method == 'Bazin':
                            lc.fit_bazin_all()
                        else:
                            raise ValueError('Only Bazin features are implemented!')
   
                        # only save to file if all filters were fitted
                        if len(lc.bazin_features) > 0 and \
                                'None' not in lc.bazin_features:

                            # see if query is possible
                            queryable = \
                                lc.check_queryable(mjd=self.min_epoch + day_of_survey,
                                                   r_lim=self.rmag_lim)              

                            # set redshift
                            lc.redshift = self.metadata[key]['true_z'].iloc[i]

                            # set light curve type
                            lc.sncode  = self.metadata[key]['true_target'].iloc[i]
                            lc.sntype = self.class_code[lc.sncode]

                            # set id
                            lc.id = snid
                            
                            # save results
                            self.write_bazin_to_file(lc, features_file, queryable)
    
