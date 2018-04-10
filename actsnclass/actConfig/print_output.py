"""Created by Emille E. O. Ishida in 7 Sep 2016.

Collection of output functions.

- save_results
    Save results from Active Learning Pipeline to file. 

"""
import numpy as np
import pandas as pd
import os

from actsnclass import read_snana_lc


def save_results(accuracies, effs, purs, foms, queryNames, queryClasses,
                 queryLabels, user_input, path_hi_snr, path_low_snr):
    """"Save results from Active Learning Pipeline to file. 

        ***************************** IMPORTANT: *****************************
        * this is extremelly taylored for post-SNPCC data format!            *
        * if you are using another data set, writing your own print function *
        * is advisable!                                                      *
        **********************************************************************

        input: accuracies, list - accuracy results
               effs, list - efficiency results
               purs, list - purity results
               foms, list - figure of merit results
               queryNames, list - identification of queried objects
               queryClasses, list - code of template used for simulations
               queryLabels, list - sn type for each queried object
               user_input, dict - user choices
                 keywords mus contain:
                     output_folder - path to output folder
                     diagnosticFile - complete path to output diagnostics
                     queryFile - complete path to output query properties

               path_hi_snr - complete path to high SNR identification
               path_low_snr - complete path to low SNR identification
                     
        output: 2 csv files:
                   diagnostics for each new query
                   characteristics of each queried object
    """

    snIbc = ['1','5','6','7','8','9','10','11','13','14','16','18','22',
             '23','29','45','28']
    snII = ['2','3','4','12','15','17','19','20','21','24','25','26','27',
            '30','31','32','33','34','35','36','37','38','39','40','41',
            '42','43','44']

    # Read high SNR ids
    op1 = open(path_hi_snr, 'r')
    lin1 = op1.readlines()
    op1.close()

    data1 = [elem.split() for elem in lin1]
    high_SNR_ids = [int(data1[j][0][6:]) for j in range(1, len(data1))]

    # Read low SNR ids
    op2 = open(path_low_snr, 'r')
    lin2 = op2.readlines()
    op2.close()

    data2 = [elem.split() for elem in lin2]
    low_SNR_ids = [int(data2[j][0][6:]) for j in range(1, len(data2))]

    # Create number of queries vector
    indx = range(0, len(foms))

    # Create data frame with diagnostics results
    diag_rows = np.vstack((indx, accuracies, effs, purs, foms))
    diag_columns = diag_rows.transpose()
    my_df = pd.DataFrame(diag_columns, columns=['nqueries', 'accuracy', 
                                                'eff', 'pur', 'fom'])
    my_df['nqueries'] = my_df['nqueries'].astype(int)

    # Check if output folder exists
    if not os.path.isdir(user_input['output_folder']):
        os.makedirs(user_input['output_folder'])

    my_df.to_csv(user_input['output_folder'] + user_input['diagnosticFile'], 
                 index=False, header=['nqueries', 'accuracy', 'eff',
                                      'pur', 'fom'])

    query_features = []
    for i in range(len(queryNames)):

        # Read raw data
        header = read_snana_lc(rawDataPath + queryNames[i]+'.DAT')[0]

        # get SN type
        if header['type'][0] == '0':
            sntype = 'Ia'
        elif header['type'][0] in snIbc:
            sntype = 'Ibc'
        elif header['type'][0] in snII:
            sntype = 'II'
        else:
            raise ValueError('Invalid SN type!')

        # identify SNR classification
        if header['snid'] in high_SNR_ids:
            SNR = 'high'
        elif header['snid'] in low_SNR_ids:
            SNR = 'low'
        else:
            raise ValueError('snid is not listed in SNR files!')

        # identify sn type
        if queryLabels[i] == 0:
            qlabel = 'nIa'
        elif queryLabels[i] == 1:
            qlabel = 'Ia'
        else:
            raise ValueError('Invalid Query Label!')

        if qlabel == 'Ia' and sntype in ['Ibc', 'II']:
            raise ValueError('Incompatible type between raw and script labels!')
        elif qlabel == 'nIa' and sntype == 'Ia':
            raise ValueError('Incompatible type between raw and script labels!')

        # Build entry in query file
        line = [str(i + 1), str(header['snid']), sntype, SNR, str(header['z'])]
        for k in range(4):
            line.append(str(header['pkmag'][k]))
        line.append(queryClasses[i])

        query_features.append(line)

    # Build data frame
    my_df2 = pd.DataFrame(query_features)

    # save to file
    my_df2.to_csv(user_input['output_folder'] + user_input['queryFile'], 
                  index=False, header=['nqueries', 'snid', 'sntype', 'SNR',
                                       'z', 'g_pkmag', 'r_pkmag', 'i_pkmag',
                                       'z_pkmag', 'qclass'])

