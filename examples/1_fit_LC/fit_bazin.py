"""
Developed by the CRP #4 team. 
CRP #4 was held in Clermont Ferrand, France in August 2017.

This script uses the Bazin et al., 2009 parametric fit and returns
the best fit parameters for each supernova in 4 DES filters.
"""



import numpy as np
import os

from snclass.util import read_user_input, read_snana_lc
from snclass.treat_lc import LC
from fit_lc_parametric import fit_scipy, bazin
import pylab as plt

# determinie minimum MJD
file_list = os.listdir('../../data/SIMGEN_PUBLIC_DES/')
lc_list = [elem for elem in file_list if 'DES_SN' in elem]

# read user input file
user_input = read_user_input('user.input')
spec_data = {}
spec_label = []
photo_data = {}
photo_label = []

# rather to generate plots
plot = False

xaxis = np.arange(-120, 350, 0.5)

for fname in lc_list:

    # read user input
    user_input['path_to_lc'] = [fname]
    user_input['photon_flag'] = ['FLUXCAL']
    user_input['photonerr_flag'] = ['FLUXCALERR']

    # read raw data
    lc_flux = read_snana_lc(user_input)

    data = {}
    for f in user_input['filters']:
        line = []
        for i in range(len(lc_flux[f])):
           line.append(lc_flux[f][i])
                    
        # we need at least 5 points to fit the light curve
        if len(line) >= 5:
            data[f] = np.array(line)

    if len(data.keys()) == 4:

        new_fit = []
    
        for f in user_input['filters']:
            # fit light curve
            best_fit = fit_scipy(data[f][:,0] - data[f][0][0], data[f][:,1])
            
            if sum([str(item) == 'nan' for item in best_fit]) == 0:
                for fit in best_fit:
                    new_fit.append(fit)

                snid = 'DES_SN' + lc_flux['SNID:'][0].zfill(6) 

                if lc_flux['SNTYPE:'][0] == '-9':
                    sample = 'target'
                else:
                    sample = 'train'

        if len(new_fit) == len(user_input['filters']) * len(best_fit):

            print lc_list.index(fname)

            # create minumum light curve
            if sample == 'train':
                spec_data[lc_flux['SNID:'][0]] = new_fit
                spec_label.append([snid, sample, lc_flux['SIM_NON1a:'][0]])
            else:
                photo_data[lc_flux['SNID:'][0]] = new_fit
                photo_label.append([snid, sample, lc_flux['SIM_NON1a:'][0]])

            if plot:
                plt.figure(figsize=(10,10))

                for i in range(len(user_input['filters'])):
                    yaxis = [bazin(item, *new_fit[(i*5):(i+1)*5]) for item in xaxis]
                    xdata = lc_flux[user_input['filters'][i]][:,0][:7]
                    ydata = lc_flux[user_input['filters'][i]][:,1][:7]
                    yerrdata = lc_flux[user_input['filters'][i]][:,2][:7]
                    
                    # we need at least 5 points to fit the light curve
                    plt.subplot(2,2,i+1)
                    if len(xdata) >= 5:                     
                        plt.errorbar(xdata, ydata, yerr=yerrdata, fmt='o', color='blue')
                    plt.plot(xaxis + lc_flux[user_input['filters'][i]][0][0], yaxis, color='red', lw=2)
                    plt.xlim(min(lc_flux[user_input['filters'][i]][:,0]), max(lc_flux[user_input['filters'][i]][:,0]))

                plt.show()
                plt.close('all')



op3 = open('../data/spec_fitparameters.dat', 'w')
op3.write('#gA gB gt0 gtfall gtrise rA rB rt0 rtfall rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ztrise\n')
for k in range(len(spec_label)):
    if len(spec_data[str(int(spec_label[k][0][6:]))]) > 0:
        for item in spec_data[str(int(spec_label[k][0][6:]))]:
            op3.write(str(item) + ' ')
        op3.write('\n')
op3.close()

op4 = open('../data/spec_labels.dat', 'w')            
op4.write('# ID train/target Class\n')
for line2 in spec_label:
    for item2 in line2:
        op4.write(item2 + ' ')
    op4.write('\n')
op4.close()

op5 = open('../data/photo_fitparameters.dat', 'w')
op5.write('#gA gB gt0 gtfall gtrise rA rB rt0 rtfall rtrise iA iB it0 itfall itrise zA zB zt0 ztfall ztrise\n')
for k in range(len(photo_label)):
    if len(photo_data[str(int(photo_label[k][0][6:]))]) > 0:
        for item in photo_data[str(int(photo_label[k][0][6:]))]:
            op5.write(str(item) + ' ')
        op5.write('\n')
op5.close()

op6 = open('../data/photo_labels.dat', 'w')            
op6.write('# ID train/target Class\n')
for line2 in photo_label:
    for item2 in line2:
        op6.write(item2 + ' ')
    op6.write('\n')
op6.close()


