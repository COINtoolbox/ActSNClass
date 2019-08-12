# Copyright 2019 snactclass software
# Author: Emille E. O. Ishida
#         Based on initial prototype developed by the CRP #4 team
#
# created on 9 August 2019
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

__all__ = ['efficiency', 'purity', 'fom', 'accuracy', 'get_snpcc_metric']


def efficiency(label_pred: list, label_true: list, ia_flag=1):
    """Calculate efficiency.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: int (optional)
        Flag used to identify Ia objects. Default is 1.

    Returns
    -------
    efficiency: float
       Fraction of correctly classified SN Ia.

    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    tot_ia = sum([label_true[i] == ia_flag for i in range(len(label_true))])

    return float(cc_ia) / tot_ia


def purity(label_pred: list, label_true: list, ia_flag=1):
    """ Calculate purity.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: int (optional)
        Flag used to identify Ia objects. Default is 1.

    Returns
    -------
    Purity: float
        Fraction of true SN Ia in the final classified Ia sample.

    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    wr_nia = sum([label_pred[i] != label_true[i] and label_true[i] != ia_flag for i in range(len(label_pred))])

    if cc_ia + wr_nia > 0:
        return float(cc_ia) / (cc_ia + wr_nia)
    else:
        return 0


def fom(label_pred: list, label_true: list, ia_flag=1, penalty=3.0):
    """
    Calculate figure of merit.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: bool (optional)
        Flag used to identify Ia objects. Default is 1.
    penalty: float
        Weight given for non-Ias wrongly classified.

    Returns
    -------
    figure of merit: float
        Efficiency x pseudo-purity (purity with a penalty for false positives)

    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    wr_nia = sum([label_pred[i] != label_true[i] and label_true[i] != ia_flag for i in range(len(label_pred))])
    tot_ia = sum([label_true[i] == ia_flag for i in range(len(label_true))])

    if (cc_ia + penalty * wr_nia) > 0:
        return (float(cc_ia) / (cc_ia + penalty * wr_nia)) * float(cc_ia) / tot_ia
    else:
        return 0


def accuracy(label_pred: list, label_true: list):
    """Calculate accuracy.

    Parameters
    ----------
    label_pred: list
        predicted labels
    label_true: list
        true labels

    Returns
    -------
    Accuracy: float
        Global fraction of correct classifications.

    """

    cc = sum([label_pred[i] == label_true[i] for i in range(len(label_pred))])

    return cc / len(label_pred)


def get_snpcc_metric(label_pred: list, label_true: list, ia_flag=1,
                     wpenalty=3):
    """
    Calculate the metric parameters used in the SNPCC.

    Parameters
    ----------
    label_pred: list
        Predicted labels
    label_true: list
        True labels
    ia_flag: bool (optional)
        Flag used to identify Ia objects. Default is 1.
    wpenalty: float
        Weight given for non-Ias wrongly classified.


    Returns
    -------
    metric_names: list
        Name of elements in metrics: [accuracy, eff, purity, fom]
    metric_values: list
        list of calculated metrics values for each element

    """

    calc_eff = efficiency(label_pred=label_pred,
                          label_true=label_true, ia_flag=ia_flag)
    calc_purity = purity(label_pred=label_pred,
                         label_true=label_true, ia_flag=ia_flag)
    calc_fom = fom(label_pred=label_pred,
                   label_true=label_true, ia_flag=ia_flag, penalty=wpenalty)
    calc_accuracy = accuracy(label_pred=label_pred, label_true=label_true)

    metric_values = [calc_accuracy, calc_eff, calc_purity, calc_fom]
    metric_names = ['accuracy', 'efficiency', 'purity', 'fom']

    return metric_names, metric_values


def main():
    return None


if __name__ == '__main__':
    main()
