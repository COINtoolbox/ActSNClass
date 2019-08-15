# Copyright 2019 actsnclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 8 August 2019
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

from .bazin import bazin, fit_scipy
from .build_snpcc_canonical import build_snpcc_canonical
from .build_snpcc_canonical import Canonical, plot_snpcc_train_canonical
from .classifiers import random_forest
from .database import DataBase
from .fit_lightcurves import fit_snpcc_bazin, LightCurve
from .learn_loop import learn_loop
from .metrics import accuracy, efficiency, fom, purity, get_snpcc_metric
from .query_strategies import random_sampling, uncertainty_sampling
from .plot_results import Canvas
from .scripts.build_canonical import main as build_canonical
from .scripts.build_time_domain import main as build_time_domain
from .scripts.fit_dataset import main as fit_dataset
from .scripts.make_diagnostic_plots import main as make_diagnostic_plots
from .scripts.run_loop import main as run_loop
from .scripts.run_time_domain import main as run_time_domain
from .time_domain import SNPCCPhotometry


__all__ = ['accuracy',
           'bazin',
           'build_canonical',
           'build_snpcc_canonical',
           'Canonical',
           'Canvas',
           'DataBase',
           'efficiency',
           'fit_dataset',
           'fit_scipy',
           'fit_snpcc_bazin',
           'fom',
           'get_snpcc_metric',
           'learn_loop',
           'LightCurve',
           'make_diagnostic_plots',
           'plot_snpcc_train_canonical',
           'purity',
           'random_forest',
           'random_sampling',
           'run_loop',
           'run_time_domain',
           'SNPCCPhotometry',
           'uncertainty_sampling']
