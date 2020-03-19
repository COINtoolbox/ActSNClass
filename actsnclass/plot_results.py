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

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = ['Canvas']

strategies = ['Canonical',
               'RandomSampling',
               'UncSampling',
               'UncSamplingEntropy',
               'UncSamplingLeastConfident',
               'UncSamplingMargin',
               'QBDMI',
               'QBDEntropy']


class Canvas(object):
    """Canvas object, handles and plot information from multiple strategies.

    Attributes
    ----------
    axis_label_size: int
        Size of font in axis labels.
    canonical: pd.DataFrame
        Data from Canonical strategy.
    fig_size: tuple
        Figure dimensions.
    rand_sampling: pd.DataFrame
        Data from Random Sampling strategy.
    tick_label_size: int
        Size of tick labels in both axis.
    ncolumns: int
        Number of columns in panel grid.
    nlines: int
        Number of lines in panel grid.
    nmetrics: int
        Number of metric elements (panels in plot).
    metrics_names: list
        List of names for metrics to be plotted.
    unc_sampling: pd.DataFrame
        Data from Uncertainty Sampling strategy.
    colors: dict
        Colors corresponding to each strategy.
        They were chosen to follow Mondrian's color palette.
        Do not change and try to keep the same palette in adding
        new elements.
    labels: dict
        Labels to appear on plot for each strategy.
    markers: dict
        Plot markers for each strategy.
    strategies: dict
        Dictionary connecting each data frame to its
        standard nomenclature (not the plot labels).

    Methods
    -------
    load_metrics(path_to_files: list, strategy_list: list)
        Load metrics and identify set of metrics.
    set_plot_dimensions()
        Set directives for plot sizes based on number of metrics.
    plot_metrics(output_plot_file: str, strategies_list: list)
        Generate plot for all metrics in files and strategies given as input.

    Examples
    --------
    Define input variables

    >>> path_to_files = ['results/metrics_canonical.dat',
    >>>                  'results/metrics_random.dat',
    >>>                  'results/metrics_unc.dat']
    >>> strategies_list = ['Canonical', 'RandomSampling', 'UncSampling']
    >>> output_plot = 'plots/metrics1_unc.png'

    Initiate the Canvas object, read and plot the results for
    each metric and strategy.

    >>> cv = Canvas()
    >>> cv.load_metrics(path_to_files=path_to_files,
    >>>                    strategies_list=strategies_list)
    >>> cv.set_plot_dimensions()
    >>> cv.plot_metrics(output_plot_file=output_plot,
    >>>                 strategies_list=strategies_list)

    """

    def __init__(self):
        self.axis_label_size = 30
        self.canonical = pd.DataFrame()
        self.fig_size = ()
        self.rand_sampling = pd.DataFrame()
        self.tick_label_size = 22
        self.ncolumns = 0
        self.nlines = 2
        self.line_width = 5
        self.nmetrics = 0
        self.metrics_names = []
        self.unc_sampling = pd.DataFrame()
        self.unc_sampling_entropy = pd.DataFrame()
        self.unc_sampling_least_confident = pd.DataFrame()
        self.unc_sampling_margin = pd.DataFrame()
        self.qbd_mi = pd.DataFrame()
        self.qbd_entropy = pd.DataFrame()
        self.colors = {'Canonical': '#dd0100',                 # red
                       'RandomSampling': '#fac901',            # yellow
                       'UncSampling': '#225095',               # blue
                       'UncSamplingEntropy': '#74eb34',        # lime green
                       'UncSamplingLeastConfident': '#eb34de', # pink
                       'UncSamplingMargin': '#ff8f05',         # orange
                       'QBDMI': '#0a4f08',                     # dark green
                       'QBDEntropy': '#434773'                 # grey blue
                       }
        self.labels = {'Canonical': 'Canonical',
                       'RandomSampling': 'Passive Learning',
                       'UncSampling': 'AL - Uncertainty Sampling',
                       'UncSamplingEntropy': 'Uncertainty Sampling Entropy',
                       'UncSamplingLeastConfident': 'Uncertainty Sampling LC',
                       'UncSamplingMargin': 'Uncertainty Sampling Margin',
                       'QBDMI': 'QBD with MI',
                       'QBDEntropy': 'QBD with Entropy'}
        self.markers = {'Canonical': '--',
                        'RandomSampling': ':',
                        'UncSampling': '-.',
                        'UncSamplingEntropy': '-.',
                        'UncSamplingLeastConfident': '-.',
                        'UncSamplingMargin': '-.',
                        'QBDMI': '-.',
                        'QBDEntropy': '-.'}
        self.strategies = {'Canonical': self.canonical,
                           'RandomSampling': self.rand_sampling,
                           'UncSampling': self.unc_sampling,
                           'UncSamplingEntropy': self.unc_sampling_entropy,
                           'UncSamplingLeastConfident': self.unc_sampling_least_confident,
                           'UncSamplingMargin': self.unc_sampling_margin,
                           'QBDMI': self.qbd_mi,
                           'QBDEntropy': self.qbd_entropy}

    def load_metrics(self, path_to_files: list, strategies_list: list):
        """Load and identify set of metrics.

        Populates attributes: canonical, unc_sampling or rand_sampling,
        depending on choice of 'sample'.

        Parameters
        ----------
        path_to_files : str
            List of paths to metrics files for different strategies.
        strategies_list: list
            List of all strategies to be included in the same plot.
            Current possibibilities are:
            ['canonical', 'rand_sampling', 'unc_sampling'].
        """

        # read data
        for i in range(len(strategies_list)):
            name = strategies_list[i]
            self.strategies[name] = pd.read_csv(path_to_files[i],
                                                sep=' ',
                                                index_col=False)

            # get metrics names
            if len(self.metrics_names) == 0:
                self.metrics_names = \
                    list(self.strategies[name].columns[1: -1])

    def set_plot_dimensions(self):
        """Set directives for plot sizes.

        Populates attributes: nmetrics, ncolumns, and fig_size.
        """

        # determine number of lines and columns in plot
        self.nmetrics = len(self.metrics_names)
        self.ncolumns = self.nmetrics // self.nlines + self.nmetrics % self.nlines
        self.fig_size = (10 * self.ncolumns, 7 * self.nlines)

    def plot_metrics(self,  output_plot_file: str, strategies_list: list,
                         lim_queries=None):
        """
        Generate plot for all metrics in files and strategies given as input.

        Parameters
        ----------
        output_plot_file : str
            Complete path to file to store plot.
        strategies_list: list
            List of all strategies to be included in the same plot.
            Current possibibilities are:
            ['canonical', 'rand_sampling', 'unc_sampling'].
        lim_queries: int or None (optional)
            If int, maximum number of queries to be plotted.
            If None no limits are imposed. Default is None.
        """

        # set of all matrices to be plotted
        all_data = [self.strategies[name] for name in strategies_list]

        # set color scheme
        sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})

        # hold axis
        axis = []

        plt.figure(figsize=self.fig_size)
        for i in range(self.nmetrics):
            ax = plt.subplot(self.nlines, self.ncolumns, i + 1)

            for j in range(len(all_data)):

                # get data
                x = all_data[j].values[:, 0]
                y = all_data[j].values[:, i + 1]

                if isinstance(lim_queries, int):
                    xflag = x <= lim_queries
                    xnew = x[xflag]
                    ynew = y[xflag]
                else:
                    xnew = x
                    ynew = y

                color = self.colors[strategies_list[j]]
                marker = self.markers[strategies_list[j]]

                # plot
                ax.plot(xnew, ynew, color=color, ls=marker, lw=self.line_width,
                        label=self.labels[strategies_list[j]])

            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([np.round(item, 2) for item in ax.get_yticks()],
                               fontsize=self.tick_label_size)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([int(item)
                               for item in ax.get_xticks()],
                               fontsize=self.tick_label_size)

            ax.set_xlabel('Number of queries', fontsize=self.axis_label_size)
            ax.set_ylabel(self.metrics_names[i], fontsize=self.axis_label_size)

            axis.append(ax)

        # handles legend
        handles, labels = axis[0].get_legend_handles_labels()
        ph = [plt.plot([], marker="", ls="")[0]]
        h = ph + handles
        rename_labels = labels
        lgd = axis[0].legend(h[1:], rename_labels, loc='upper center',
                             bbox_to_anchor=(1.025, 1.35),
                             ncol=3, fontsize=25, title='Strategy:')
        plt.setp(lgd.get_title(), fontsize='23')

        plt.subplots_adjust(left=0.075, right=0.95, top=0.875, bottom=0.075,
                            wspace=0.35, hspace=0.35)
        plt.savefig(output_plot_file)


def main():
    return None


if __name__ == '__main__':
    main()
