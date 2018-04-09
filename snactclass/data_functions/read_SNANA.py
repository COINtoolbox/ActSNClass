"""
Created by Emille Ishida on 7 August 2017.

Simple function to read SNANA light curve files.

- read_snana_lc:
    Read post-SNPCC data.
"""

import numpy as np
import pandas as pd

def read_snana_lc(file_name, flux_as_DataFrame=False):
    """
    Read DES light curve in SNANA format.

    input:     file_name, str - path to light curve file
               flux_as_DataFrame, boolean - output flux as pandas DataFrame

    output:    header, dict - keywords: snid  - SN identification
                                        z     - redshift
                                        type  - SN type
                                        model - SN model
                                        pkmjd - simulated day of max
                                        sample - target or train
                                        columns - name of columns in each
                                                  filter matrix

               if flux_as_DataFrame:
                   measurements, dict - keywords:   g, r, i, z
                                        values:     np.arrays of measurements
                                                    column names in 
                                                    header['columns']
               else:
                  measurements, DataFrame - first layer: g, r, i, z
                                            second layer: header['columns']
    """

    filters=['g', 'r', 'i', 'z']

    # read light curve data
    with open(file_name, 'r') as op1:
        lin1 = op1.readlines()

    data1 = [elem.split() for elem in lin1]

    # get header data
    raw_data = dict([[line[0], line[1:]] for line in data1
                      if len(line) > 1 and line[0] != 'OBS:'])

    header = {}
    header['snid'] = int(raw_data['SNID:'][0])
    header['z'] = float(raw_data['SIM_REDSHIFT:'][0])
    header['type'] = raw_data['SIM_NON1a:']
    header['model'] = raw_data['SIM_MODEL:']
    header['pkmjd'] = float(raw_data['SIM_PEAKMJD:'][0])
    header['pkmag'] = [float(raw_data['SIM_PEAKMAG:'][j])
                       for j in range(len(filters))]
    header['columns'] = ['MJD', 'FLUXCAL', 'FLUXCALERR', 'SNR']
    header['SIM_COMMENT:'] = raw_data['SIM_COMMENT:']

    if raw_data['SNTYPE:'][0] == '-9':
        header['sample'] = 'target'
    else:
        header['sample'] = 'train'

    # get flux measurements
    flux = {}
    
    for f in filters:
        flist = []
        for line in data1:
            if len(line) > 2 and line[0] == 'OBS:' and line[2] == f:
                flist.append([float(line[1]), float(line[4]),
                              float(line[5]), float(line[6]), float(line[7]), float(line[8])])

        flux[f] = np.array(flist)

 
    if flux_as_DataFrame:

        data = {}
        for f in filters:
            data[f] = {}
            data[f]['MJD'] = flux[f][:,0]
            data[f]['FLUXCAL'] = flux[f][:,1]
            data[f]['FLUXCALERR'] = flux[f][:,2]
            data[f]['SNR'] = flux[f][:,3]
            data[f]['MAG'] = flux[f][:,4]
            data[f]['MAGERR'] = flux[f][:,5]

        data = pd.DataFrame(data)

        return header, data

    return header, flux
