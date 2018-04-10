"""
Developed by the CRP #4 team. 
CRP #4 was held in Clermont Ferrand, France in August 2017.

This script gets simulated peak magnitudes, redshift type and SNR
for the different data samples.
"""


import numpy as np
from snactclass import read_snana_lc
import pandas as pd


filters = ['g', 'r', 'i', 'z']

# read cross validation snids
op1 = open('../data/query_labels.dat', 'r')
lin1 = op1.readlines()
op1.close()

cv = [[elem.split()[0]] for elem in lin1[1:]]

# get other features for cross validation sample
cv_features = []
for line in cv:
    snlc = read_snana_lc('../../data/SIMGEN_PUBLIC_DES/' + line[0] + '.DAT')

    entry = [snlc[0]['snid'], snlc[0]['z'], snlc[0]['type'][0]]
    for item in snlc[0]['pkmag']:
        entry.append(item)

    for f in filters:
        entry.append(np.mean(snlc[1][f][:,3]))

    cv_features.append(entry)

# save to file
cv_pd  = pd.DataFrame(cv_features, columns=['snid', 'z', 'sntype', 'g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'])
cv_pd.to_csv('../data/query_features.csv', header=['snid', 'z', 'sntype','g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'], index=False)

####

# read publication snids
op2 = open('../data/target_labels.dat', 'r')
lin2 = op2.readlines()
op2.close()

pub = [[elem.split()[0]] for elem in lin2[1:]]

# get other features for publication sample
pub_features = []
for line in pub:

    snlc = read_snana_lc('../../data/SIMGEN_PUBLIC_DES/' + line[0] + '.DAT')

    entry = [snlc[0]['snid'], snlc[0]['z'], snlc[0]['type'][0]]
    for item in snlc[0]['pkmag']:
        entry.append(item)

    for f in filters:
        entry.append(np.mean(snlc[1][f][:,3]))

    pub_features.append(entry)

# save to file
pub_pd  = pd.DataFrame(pub_features, columns=['snid', 'z', 'sntype', 'g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'])
pub_pd.to_csv('../data/target_features.csv', header=['snid', 'z', 'sntype','g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'], index=False)


#####

# construct target features
target_features = cv_features

for item in pub_features:
    target_features.append(item)

# save to file
target_pd  = pd.DataFrame(target_features, columns=['snid', 'z', 'sntype', 'g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'])
target_pd.to_csv('../data/photo_features.csv', header=['snid', 'z', 'sntype','g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'], index=False)


##########

# read train validation snids
op1 = open('../data/train_labels.dat', 'r')
lin1 = op1.readlines()
op1.close()

cv = [[elem.split()[0]] for elem in lin1[1:]]

# get other features for cross validation sample
cv_features = []
for line in cv:

    snlc = read_snana_lc('../../data/SIMGEN_PUBLIC_DES/' + line[0] + '.DAT')

    entry = [snlc[0]['snid'], snlc[0]['z'], snlc[0]['type'][0]]
    for item in snlc[0]['pkmag']:
        entry.append(item)

    for f in filters:
        entry.append(np.mean(snlc[1][f][:,3]))

    cv_features.append(entry)

# save to file
cv_pd  = pd.DataFrame(cv_features, columns=['snid', 'z', 'sntype', 'g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'])
cv_pd.to_csv('../data/train_features.csv', header=['snid', 'z', 'sntype','g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag', 'g_SNR', 'r_SNR', 'i_SNR','z_SNR'], index=False)

    
