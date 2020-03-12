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

from .bazin import *
from .build_snpcc_canonical import *
from .classifiers import *
from .database import *
from .fit_lightcurves import *
from .learn_loop import *
from .metrics import *
from .query_strategies import *
from .plot_results import *
from .scripts.build_canonical import main as build_canonical
from .scripts.build_time_domain import main as build_time_domain
from .scripts.fit_dataset import main as fit_dataset
from .scripts.make_metrics_plots import main as make_metrics_plots
from .scripts.run_loop import main as run_loop
from .scripts.run_time_domain import main as run_time_domain
from .time_domain import *
from .time_domain_loop import *

__all__ = ['accuracy',
           'bazin',
           'build_canonical',
           'build_snpcc_canonical',
           'Canonical',
           'Canvas',
           'DataBase',
           'efficiency',
           'errfunc',
           'fit_dataset',
           'fit_scipy',
           'fit_snpcc_bazin',
           'fit_plasticc_bazin',
           'fit_resspect_bazin',
           'fom',
           'get_snpcc_metric',
           'gradient_boosted_trees',
           'knn',
           'learn_loop',
           'LightCurve',
           'mlp',
           'nbg',
           'PLAsTiCCPhotometry',
           'make_metrics_plots',
           'plot_snpcc_train_canonical',
           'purity',
           'random_forest',           
           'random_sampling',
           'run_loop',
           'run_time_domain',
           'SNPCCPhotometry',
           'svm',
           'time_domain_loop',
           'uncertainty_sampling']
