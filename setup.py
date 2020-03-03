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

import setuptools


setuptools.setup(
    name='actsnclass',
    version='1.2',
    packages=setuptools.find_packages(),
    py_modules=['bazin',
                'build_snpcc_canonical',
                'classifiers',
                'data_base',
                'fit_lightcurves',
                'learn_loop',
                'metrics',
                'plot_results',
                'query_strategies',
                'time_domain'],
    scripts=['actsnclass/scripts/build_canonical.py',
             'actsnclass/scripts/build_time_domain.py',
             'actsnclass/scripts/fit_dataset.py',
             'actsnclass/scripts/make_diagnostic_plots.py',
             'actsnclass/scripts/run_loop.py',
             'actsnclass/scripts/run_time_domain.py'],
    url='https://github.com/COINtoolbox/ActSNClass',
    license='GNU3',
    author='Emille E. O.Ishida',
    author_email='emille@cosmostatistics-initiative.org',
    description='Active Learning for SN photometric classification'
)
