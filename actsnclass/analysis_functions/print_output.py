"""Created by Emille E. O. Ishida in 7 Sep 2016.

Collection of output functions.

- save_results
    Save results from Active Learning Pipeline to file. 

"""
import numpy as np
import pandas as pd
import os

<<<<<<< HEAD
from actsnclass.data_functions.read import read_snana_lc
=======
from actsnclass.data_functions.read_SNANA import read_snana_lc
>>>>>>> 117a0f0bcc417f5cc9ddd8b9d1c0cda6e93de7c2
from actsnclass.analysis_functions.ml_result import MLResult


def save_results(accuracies, effs, purs, foms, queryNames, queryClasses,
                 queryLabels, user_input, path_raw_data, Ia_label=1):
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
                     outputFolder - path to output folder
                     diagnosticFile - complete path to output diagnostics
                     queryFile - complete path to output query properties
               path_row_data, str - path to raw data folder
                     
        output: 2 csv files:
                   diagnostics for each new query
                   characteristics of each queried object
    """

    snIbc = ['1','5','6','7','8','9','10','11','13','14','16','18','22',
             '23','29','45','28']
    snII = ['2','3','4','12','15','17','19','20','21','24','25','26','27',
            '30','31','32','33','34','35','36','37','38','39','40','41',
            '42','43','44']

    filters = ['g', 'r', 'i', 'z']

    # Check if output folder exists
    if not os.path.isdir(user_input['outputFolder']):
        os.makedirs(user_input['outputFolder'])
    
    

    # save diagnostics results
    if user_input['choice'] in ['batch_random', 'batch_nlunc', 'batch_semi']:

        # horrible way to take into account the many ids in a batch
        # (repeating the diagnostics)
        [queryNames.insert(0, '---') for k in range(user_input['batchSize'])]
        foms_rep = [foms[i] for i in range(len(foms)) \
                            for j in range(user_input['batchSize'])]
        effs_rep = [effs[i] for i in range(len(effs)) \
                            for j in range(user_input['batchSize'])]
        purs_rep = [purs[i] for i in range(len(purs)) \
                            for j in range(user_input['batchSize'])]
        accuracies_rep = [accuracies[i] for i in range(len(foms)) \
                                     for j in range(user_input['batchSize'])]

        diagnostics = MLResult(queryNames, foms_rep, effs_rep, purs_rep, accuracies_rep)
            
    else:
        queryNames.insert(0, '---')
        diagnostics = MLResult(queryNames, foms, effs, purs, accuracies)
    
    # save to file
    diagnostics.save(user_input['outputFolder'] + user_input['diagnosticFile'])

    if user_input['choice'] in ['batch_random', 'batch_nlunc', 'batch_semi']:
        [queryNames.remove('---') for k in range(user_input['batchSize'])]
    else:
        queryNames.remove('---')


    query_features = []
    for i in range(len(queryNames)):

        # Read raw data
        lc = read_snana_lc(path_raw_data + queryNames[i]+'.DAT')

        # get SN type
        if lc[0]['type'][0] == '0':
            sntype = 'Ia'
        elif lc[0]['type'][0] in snIbc:
            sntype = 'Ibc'
        elif lc[0]['type'][0] in snII:
            sntype = 'II'
        else:
            raise ValueError('Invalid SN type!')

        # identify sn type
        if queryLabels[i] != Ia_label:
            qlabel = 'nIa'
        elif queryLabels[i] == Ia_label:
            qlabel = 'Ia'
        else:
            raise ValueError('Invalid Query Label!')

        if qlabel == 'Ia' and sntype in ['Ibc', 'II']:
            raise ValueError('Incompatible type between raw and script labels!')
        elif qlabel == 'nIa' and sntype == 'Ia':
            raise ValueError('Incompatible type between raw and script labels!')

        # Build entry in query file
        # line = [str(i + 1), str(header['snid']), sntype, SNR, str(header['z'])]
        line = [str(i + 1), str(lc[0]['snid']), sntype, str(lc[0]['z'])]
        for k in range(4):
            line.append(str(lc[0]['pkmag'][k]))

        for f in filters:
            line.append(str(np.mean(lc[1][f][:,3])))        

        line.append(queryClasses[i])

        query_features.append(line)

    # Build data frame
    my_df2 = pd.DataFrame(query_features)
 
    h1 = ['nqueries', 'snid', 'sntype', 'z', 'g_pkmag', 'r_pkmag',
          'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR', 'z_SNR', 'qclass']
 
    # save to file
    my_df2.to_csv(user_input['outputFolder'] + user_input['queryFile'], 
                  index=False, header=h1)

