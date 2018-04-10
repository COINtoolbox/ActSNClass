"""Created by the CRP #4 team between 20-27 Aug 2017.

Functions to deal with the naming conventions on post-SNPCC data.

- plt_lc_raw
    Generate plot with the observed points in a given light curve. 
"""


import pandas as pd
import matplotlib.pyplot as plt

def plot_lc_raw(lc):
    """
    Generate plot with the observed points in a given light curve. 

    input: lc, tuple - output from snactclass.read_SNANA.read_snana_lc 

    output: plot visualization
    """
    
    filters = ['g', 'r', 'i', 'z']

    data = {}
    for f in filters:
        data[f] = {}
        data[f]['MJD'] = lc[1][f][:,0]
        data[f]['FLUXCAL'] = lc[1][f][:,1]
        data[f]['FLUXCALERR'] = lc[1][f][:,2]

    data = pd.DataFrame(data)
    
    fig, ax = plt.subplots()

    color_wheel = {'r':'#dd0100','g':'#fac901','i':'#225095','z':'brown'}
    for i, (band, data) in enumerate(data.iteritems()):
        
        ax.errorbar(data['MJD'],
                    data['FLUXCAL'],
                    yerr=data['FLUXCALERR'],
                    color=color_wheel[band],
                    fmt='o',
                    label=band)
        
    ax.legend()
    ax.set_xlabel('MJD')
    ax.set_ylabel('Flux')

    plt.show()
    
