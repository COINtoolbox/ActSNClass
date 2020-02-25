import os
import numpy as np
import pandas as pd

from actsnclass import LightCurve


class PLAsTiCCPhotometry(object):
    """ """

    def __init__(self):
        self.bazin_header = 'id redshift type code sample uA uB ut0 ' + \
                            'utfall utrise gA gB gt0 ' + \
                            'gtfall gtrise rA rB rt0 rtfall rtrise iA ' + \
                            'iB it0 itfall itrise zA zB zt0 ztfall ztrise' + \
                            'YA YB Yt0 Ytfall Ytrise\n'
        self.max_epoch = 60674.363
        self.min_epoch = 59580.0338
        self.rmag_lim = 24
        self.class_code = {90: 'Ia', 67: '91bg', 52:'Iax', 42:'II', 62:'Ibc', 
             95: 'SLSN', 15:'TDE', 64:'KN', 88:'AGN', 92:'RRL', 65:'M-dwarf',
             16:'EB',53:'Mira', 6:'MicroL', 991:'MicroLB', 992:'ILOT', 
             993:'CART', 994:'PISN',995:'MLString'}
        self.metadata = {}

    def create_daily_file(self, output_dir: str,
                          day: int, header='Bazin'):
        """Create one file for a given day of the survey.

        The file contains only header for the features file.

        Parameters
        ----------
        output_dir: str
            Complete path to raw data directory.
        day: int
            Day passed since the beginning of the survey.
        header: str (optional)
            List of elements to be added to the header.
            Separate by 1 space.
            Default option uses header for Bazin features file.
        """

        features_file = output_dir + 'day_' + str(day) + '.dat'

        if header == 'Bazin':
            # add headers to files
            with open(features_file, 'w') as param_file:
                param_file.write(self.bazin_header)

        else:
            with open(features_file, 'w') as param_file:
                param_file.write(self.header)


    def filter_classes(self, path_to_data_dir: str, classes: list):
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



    def build_one_epoch(self, raw_data_dir: str, day_of_survey: int,
                        time_domain_dir: str, feature_method='Bazin',
                        dataset='PLAsTiCC'):

        
        # list of PLAsTiCC photometric files
        flist = ['plasticc_test_lightcurves_' + str(x).zfill(2) + '.csv.gz' 
                 for x in range(1, 12)]

        # add training file
        flist = flist + ['plasticc_train_lightcurves.csv.gz']

        

        for fname in flist:
            # read photometric points
            if '.tar.gz' in fname:
                tar = tarfile.open(fname, 'r:gz')
                name = tar.getmembers()[0]
                content = tar.extractfile(name).read()
                all_photo = pd.read_csv(io.BytesIO(content))
            else:
                all_photo = pd.read_csv(fname, index_col=False)

                if ' ' in all_photo.keys()[0]:
                    all_photo = pd.read_csv(fname, sep=' ', 
                                            index_col=False)




################################################
        # read file names
        file_list_all = os.listdir(raw_data_dir)
        lc_list = [elem for elem in file_list_all if 'DES_SN' in elem]

        # get name of file to store results of today
        features_file = time_domain_dir + 'day_' + str(day_of_survey) + '.dat'

        # count survivers
        count_surv = 0

        for i in range(len(lc_list)):
            print('Processed : ', i)

            lc = LightCurve()  # create light curve instance

            if dataset == 'SNPCC':
                lc.load_snpcc_lc(raw_data_dir + lc_list[i])
            else:
                raise ValueError('Only SNPCC data set is implemented!')

            # see which epochs are observed for until this day
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
                    count_surv = count_surv + 1
                    print('... ... ... Survived: ', count_surv)

                    # see if query is possible
                    queryable = \
                        lc.check_queryable(mjd=self.min_epoch + day_of_survey,
                                           r_lim=self.rmag_lim)

                    if queryable:
                        lc.sample = 'queryable'

                    # save features to file
                    with open(features_file, 'a') as param_file:
                        param_file.write(str(lc.id) + ' ' +
                                         str(lc.redshift) + ' ' +
                                         str(lc.sntype) + ' ')
                        param_file.write(str(lc.sncode) + ' ' +
                                         str(lc.sample) + ' ')
                        for item in lc.bazin_features[:-1]:
                            param_file.write(str(item) + ' ')
                        param_file.write(str(lc.bazin_features[-1]) + '\n')   
    
