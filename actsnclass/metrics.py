"""Created by CRP #4 team between 20-27 Aug 2017

Standard diagnostic to access classification results.

- efficiency
    Fraction of type ia SNe found.
- purity
    Fraction of correct ia classifications.
- fom
    efficiency * pseudo-purity.
- accuracy
    Global fraction of correct classifications
"""


def efficiency(label_pred, label_true, ia_flag=1):
    """Calculate efficiency.

       input: label_pred, predicted labels
              label_true, true labels
              ia_flag, ia flag (optional, default is 0)

       output: efficiency

    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    tot_ia = sum([label_true[i] == ia_flag for i in range(len(label_true))])

    return float(cc_ia) / tot_ia


def purity(label_pred, label_true, ia_flag=1):
    """Calculate purity.

       input: label_pred, predicted labels
              label_true, true labels
              ia_flag, ia flag (optional, default is 0)

       output: purity
    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    wr_nia = sum([label_pred[i] != label_true[i] and label_true[i] != ia_flag for i in range(len(label_pred))])

    if cc_ia + wr_nia > 0:
        return float(cc_ia) / (cc_ia + wr_nia)
    else:
        return 0


def fom(label_pred, label_true, ia_flag=1, penalty=3):
    """
    Calculate figure of merit.

    input: label_pred, predicted labels
           label_true, true labels
           ia_flag, ia flag (optional, default is 0)
           penalty factor for wrong non-ia classification (optional, default is 3)

    output: figure of merity
    """

    cc_ia = sum([label_pred[i] == label_true[i] and label_true[i] == ia_flag for i in range(len(label_pred))])
    wr_nia = sum([label_pred[i] != label_true[i] and label_true[i] != ia_flag for i in range(len(label_pred))])
    tot_ia = sum([label_true[i] == ia_flag for i in range(len(label_true))])

    if (cc_ia + penalty * wr_nia) > 0:
        return (float(cc_ia) / (cc_ia + penalty * wr_nia)) * float(cc_ia) / tot_ia
    else:
        return 0


def accuracy(label_pred, label_true):
    """Calculate accuracy.

       input: label_pred, predicted labels
              label_true, true labels

       output: accuracy
    """

    cc = sum([label_pred[i] == label_true[i] for i in range(len(label_pred))])

    return cc / len(label_pred)