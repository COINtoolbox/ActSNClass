"""Created by CRP #4 team between 20-27 Aug 2017

Standard diagnostic to access classification results. 

- efficiency
    Fraction of type Ia SNe found.
- purity
    Fraction of correct Ia classifications.
- fom
    efficiency * pseudo-purity.
"""

def efficiency(label_pred, label_true, Ia_flag=0):
    """Calculate efficiency.
       
       input: label_pred, predicted labels
              label_true, true labels
              Ia_flag, Ia flag (optional, default is 0)

       output: efficiency
    
    """

    cc_Ia = sum([label_pred[i] == label_true[i] and label_true[i] == Ia_flag for i in range(len(label_pred))])
    tot_Ia = sum([label_true[i] == Ia_flag for i in range(len(label_true))])

    return float(cc_Ia)/tot_Ia


def purity(label_pred, label_true, Ia_flag=0):
    """Calculate purity.

       input: label_pred, predicted labels
              label_true, true labels
              Ia_flag, Ia flag (optional, default is 0)

       output: purity
    """

    cc_Ia = sum([label_pred[i] == label_true[i] and label_true[i] == Ia_flag for i in range(len(label_pred))])
    wr_nIa = sum([label_pred[i] != label_true[i] and label_true[i] != Ia_flag for i in range(len(label_pred))])

    if cc_Ia + wr_nIa > 0:
        return float(cc_Ia)/(cc_Ia + wr_nIa)
    else:
        return 0

def fom(label_pred, label_true, Ia_flag=0, penalty=3):
    """
    Calculate figure of merit.

    input: label_pred, predicted labels
           label_true, true labels
           Ia_flag, Ia flag (optional, default is 0)
           penalty factor for wrong non-Ia classification (optional, default is 3)

    output: figure of merity
    """

    cc_Ia = sum([label_pred[i] == label_true[i] and label_true[i] == Ia_flag for i in range(len(label_pred))])
    wr_nIa = sum([label_pred[i] != label_true[i] and label_true[i] != Ia_flag for i in range(len(label_pred))])
    tot_Ia = sum([label_true[i] == Ia_flag for i in range(len(label_true))])

    if (cc_Ia + penalty * wr_nIa) > 0:
        return (float(cc_Ia)/(cc_Ia + penalty * wr_nIa)) * float(cc_Ia)/tot_Ia
    else:
        return 0
