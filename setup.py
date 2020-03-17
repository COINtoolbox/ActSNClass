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
    py_modules=['actsnclass/bazin',
                'actsnclass/build_snpcc_canonical',
                'actsnclass/classifiers',
                'actsnclass/database',
                'actsnclass/fit_lightcurves',
                'actsnclass/learn_loop',
                'actsnclass/metrics',
                'actsnclass/plot_results',
                'actsnclass/query_strategies',
                'actsnclass/snana_fits_to_pd',
                'actsnclass/time_domain',
                'actsnclass/time_domain_PLAsTiC'],
    scripts=['actsnclass/scripts/build_canonical.py',
             'actsnclass/scripts/build_time_domain.py',
             'actsnclass/scripts/fit_dataset.py',
             'actsnclass/scripts/make_metrics_plots.py',
             'actsnclass/scripts/run_loop.py',
             'actsnclass/scripts/run_time_domain.py'],
    url='https://github.com/COINtoolbox/ActSNClass/tree/RESSPECT',
    license='GNU3',
    author='The RESSPECT team',
    author_email='contact@cosmostatistics-initiative.org',
    description='ActSNClass - Recommendation System for Spectroscopic Follow-up'
)
