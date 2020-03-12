# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 12 August 2019
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

from actsnclass import LightCurve

__all__ = ['SNPCCPhotometry']


class SNPCCPhotometry(object):
    """Handles photometric information for entire SNPCC data.

    This class only works for Bazin feature extraction method.

    Attributes
    ----------
    bazin_header: str
        Header to be added to features files for each day.
    max_epoch: float
        Maximum MJD for the entire data set.
    min_epoch: float
        Minimum MJD for the entire data set.
    rmag_lim: float
        Maximum r-band magnitude allowing a query.

    Methods
    -------
    get_lim_mjds(raw_data_dir)
        Get minimum and maximum MJD for complete sample.
    create_daily_file(raw_data_dir: str, day: int, output_dir: str, header: str)
        Create one file for a given day of the survey.
        Only populates the file with header. It will erase existing files!
    build_one_epoch(raw_data_dir: str, day_of_survey: int, time_domain_dir: str,
                    feature_method: str, dataset: str)
        Selects objects with observed points until given MJD, performs feature extraction
        and evaluate if query is possible. Save results to file.
    """
    def __init__(self):
        self.bazin_header = 'id redshift type code orig_sample queryable gA gB gt0 ' + \
                            'gtfall gtrise rA rB rt0 rtfall rtrise iA ' + \
                            'iB it0 itfall itrise zA zB zt0 ztfall ztrise\n'
        self.max_epoch = 56352
        self.min_epoch = 56171
        self.rmag_lim = 24

    def get_lim_mjds(self, raw_data_dir):
        """Get minimum and maximum MJD for complete sample.

        This function is not necessary if you are working with
        SNPCC data. The values are hard coded in the class.

        Parameters
        ----------
        raw_data_dir: str
            Complete path to raw data directory.

        Returns
        -------
        limits: list
            List of extreme MJDs for entire sample: [min_MJD, max_MJD].
        """

        # read file names
        file_list_all = os.listdir(raw_data_dir)
        lc_list = [elem for elem in file_list_all if 'DES_SN' in elem]

        # store MJDs
        min_day = []
        max_day = []

        for elem in lc_list:

            print('Searching obj n. ', lc_list.index(elem))

            lc = LightCurve()                        # create light curve instance
            lc.load_snpcc_lc(raw_data_dir + elem)                      # read data

            min_day.append(min(lc.photometry['mjd'].values))
            max_day.append(max(lc.photometry['mjd'].values))

            self.min_epoch = min(min_day)
            self.max_epoch = max(max_day)

        return [min(min_day), max(max_day)]

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

    def build_one_epoch(self, raw_data_dir: str, day_of_survey: int,
                        time_domain_dir: str, feature_method='Bazin',
                        dataset='SNPCC'):
        """Fit bazin for all objects with enough points in a given day.

        Generate 1 file containing best-fit Bazin parameters for a given
        day of the survey.

        Parameters
        ----------
        raw_data_dir: str
            Complete path to raw data directory
        day_of_survey: int
            Day since the beginning of survey. 
        time_domain_dir: str
            Output directory to store time domain files. 
        feature_method: str (optional)
            Feature extraction method.
            Only possibility is 'Bazin'.
        dataset: str (optional)
            Name of the data set. 
            Only possibility is 'SNPCC'.
        
        """

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
                                         str(lc.sample) + ' ' +
                                         str(queryable) + ' ')
                        for item in lc.bazin_features[:-1]:
                            param_file.write(str(item) + ' ')
                        param_file.write(str(lc.bazin_features[-1]) + '\n')



def main():
    return None


if __name__ == '__main__':
    main()
