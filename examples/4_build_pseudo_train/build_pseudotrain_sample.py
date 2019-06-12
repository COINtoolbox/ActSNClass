"""
Developed by the CRP #4 team. 
CRP #4 was held in Clermont Ferrand, France in August 2017.

This script uses the Bazin et al., 2009 parametric fit and returns
the best fit parameters for each supernova in 4 DES filters.
"""


import pylab as plt
import csv
from sklearn.neighbors import NearestNeighbors, KernelDensity
import numpy as np


# read data
op3 = open('../data/target_features.csv', 'r')
lin3 = op3.readlines()
op3.close()

targetFeatures = [elem.split(',') for elem in lin3[1:]]

op7 = open('../data/train_features.csv', 'r')
lin7 = op7.readlines()
op7.close()

trainFeatures = [[float(item) for item in elem.split(',')] for elem in lin7[1:]]

op1 = open('../data/target_fitparameters.dat', 'r')
lin1 = op1.readlines()
op1.close()

pubLC = [elem.split() for elem in lin1[1:]]

op4 = open('../data/target_labels.dat', 'r')
lin4 = op4.readlines()
op4.close()

pubLabels = [elem.split() for elem in lin4]

pubDict = {}
for i in range(1, len(pubLabels)):
    pubDict[str(int(pubLabels[i][0][6:]))] = pubLC[i - 1]

op2 = open('../data/query_fitparameters.dat', 'r')
lin2 = op2.readlines()
op2.close()

cvLC = [elem.split() for elem in lin2[1:]]

op5 = open('../data/query_labels.dat', 'r')
lin5 = op5.readlines()
op5.close()

cvLabels = [elem.split() for elem in lin5]

cvDict = {}
for i in range(1, len(cvLabels)):
    cvDict[str(int(cvLabels[i][0][6:]))] = cvLC[i - 1]


# remove id and type and convert to float
targetLabels = []
targetFeatFloat = []
for line in targetFeatures:
    targetLabels.append(['DES_SN' + str(int(line[0])).zfill(6), 'target', str(int(line[2]))])
    line.remove(line[0])
    line.remove(line[1])

    line2 = [float(item) for item in line]
    targetFeatFloat.append(line2)

trainLabels = []
trainFeatFloat = []
for line in trainFeatures:
    trainLabels.append(['DES_SN' + str(int(line[0])).zfill(6), 'train', str(int(line[2]))])
    line.remove(line[0])
    line.remove(line[1])

    line2 = [float(item) for item in line]
    trainFeatFloat.append(line2)

# get the types in training sample
types_train = {}
for line in trainLabels:
    if str(line[-1]) not in types_train.keys():
        types_train[str(line[-1])] = 1

    else:
        types_train[str(line[-1])] = types_train[str(line[-1])] + 1

# separate target labels by type
data_photo = {}
for sntype in types_train.keys():
    data_photo[sntype] = []
    data_photo[sntype + '_id'] = []
    for j in range(len(targetLabels)):
        if targetLabels[j][-1] == sntype:
            data_photo[sntype].append(targetFeatFloat[j])
            data_photo[sntype + '_id'].append(targetLabels[j])

# gather new data to mimic the training sample
newTrainLabels = []
newTrainFeatures = []

for l in range(len(trainFeatFloat)):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(data_photo[trainLabels[l][-1]])
    distances, indices = nbrs.kneighbors(np.array(trainFeatFloat[l]).reshape(1, -1))

    for k in range(1):
        m = indices[0][k]

        newTrainLabels.append(data_photo[trainLabels[l][-1] + '_id'][m])
        newTrainFeatures.append(data_photo[trainLabels[l][-1]][m])

        data_photo[trainLabels[l][-1]].remove(data_photo[trainLabels[l][-1]][m])
        data_photo[trainLabels[l][-1] + '_id'].remove(data_photo[trainLabels[l][-1] + '_id'][m])


op1 = open('../data/pseudotrain_labels.dat', 'w')
op1.write(lin4[0])
for line in newTrainLabels:
    for item in line[:-1]:
        op1.write(str(item) + ' ')
    op1.write(str(line[-1]) + '\n')

op1.close()

newTrainLC = []
for line in newTrainLabels:
    print len(newTrainLC) + 1
    done = False
    if str(int(line[0][6:])) in cvDict.keys():
        newTrainLC.append(cvDict[str(int(line[0][6:]))])
        done = True

    if done == False and str(int(line[0][6:])) in pubDict.keys():
        newTrainLC.append(pubDict[str(int(line[0][6:]))])
        done = True

    """
    for i in range(1, len(cvLC)):
        if str(int(cvLabels[i][0][6:])) == line[0]:
            newTrainLC.append(cvLC[i])
            done = True
            break
        if done == False:
            for j in range(1, len(pubLC)):
                if str(int(pubLabels[j][0][6:])) == line[0]:
                    newTrainLC.append(pubLC[j])
                    done = True
                    break
    """

    if done == False:
        raise ValueError('SN ' + line[0] + ' not found!')
    

op9 = open('../data/pseudotrain_fitparameters.txt', 'w')
op9.write(lin2[0])
for line in newTrainLC:
    for elem in line[:-1]:
        op9.write(str(elem) +  ' ')
    op9.write(str(line[-1]) + '\n')
op9.close()

import pylab as plt
from actsnclass import read_snana_lc
import pandas as pd

features = []
for line in newTrainLabels:
    name = line[0] + '.DAT'
    lc = read_snana_lc('../../data/SIMGEN_PUBLIC_DES/' + name)

    f1 = [lc[0]['snid'], lc[0]['z'], lc[0]['type'][0]]
    for item in lc[0]['pkmag']:
        f1.append(item)
    features.append(f1)

my_pd  = pd.DataFrame(features, columns=['snid', 'z', 'sntype', 'g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag'])
my_pd.to_csv('../data/pseudotrain_features.csv', header=['snid', 'z', 'sntype','g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag'], index=False)


old_train_features = []
for line in trainLabels:
    name = line[0] + '.DAT'
    lc = read_snana_lc('../../data/SIMGEN_PUBLIC_DES/' + name)

    f1 = [lc[0]['snid'], lc[0]['z'], lc[0]['type'][0]]
    for item in lc[0]['pkmag']:
        f1.append(item)
    old_train_features.append(f1)
    

my_pd2  = pd.DataFrame(old_train_features, columns=['snid', 'z', 'sntype','g_pkmag', 'r_pkmag', 'i_pkmag', 'z_pkmag'])

sntypes_oldtrain = {}
sntypes_newtrain = {}
for line in trainLabels:
    if line[-1] not in sntypes_oldtrain.keys():
        sntypes_oldtrain[line[-1]] = 1
    else:
        sntypes_oldtrain[line[-1]] = sntypes_oldtrain[line[-1]] + 1

for line in newTrainLabels:
    if line[-1] not in sntypes_newtrain.keys():
        sntypes_newtrain[line[-1]] = 1
    else:
        sntypes_newtrain[line[-1]] = sntypes_newtrain[line[-1]] + 1


z_newtrain = my_pd['z']
z_oldtrain = my_pd2['z']


znewtrain_axis = np.linspace(min(z_newtrain)-0.25, max(z_newtrain)+0.25, 1000)[:,np.newaxis]
kde_znewtrain = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.array(z_newtrain).reshape(-1,1))
log_dens_znewtrain = kde_znewtrain.score_samples(znewtrain_axis)

zoldtrain_axis = np.linspace(min(z_oldtrain)-0.25, max(z_oldtrain)+0.25, 1000)[:,np.newaxis]
kde_zoldtrain = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(np.array(z_oldtrain).reshape(-1,1))
log_dens_zoldtrain = kde_zoldtrain.score_samples(zoldtrain_axis)

gpknewtrain_axis = np.linspace(min(my_pd['g_pkmag'])-0.25, max(my_pd['g_pkmag'])+0.25, 1000)[:,np.newaxis]
kde_gpknewtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(my_pd['g_pkmag']).reshape(-1,1))
log_dens_gpknewtrain = kde_gpknewtrain.score_samples(gpknewtrain_axis)

gpkoldtrain_axis = np.linspace(min(my_pd2['g_pkmag'])-0.25, max(my_pd2['g_pkmag'])+0.25, 1000)[:,np.newaxis]
kde_gpkoldtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(my_pd2['g_pkmag']).reshape(-1,1))
log_dens_gpkoldtrain = kde_gpkoldtrain.score_samples(gpkoldtrain_axis)

rpknewtrain_axis = np.linspace(min(my_pd['r_pkmag'])-0.25, max(my_pd['r_pkmag'])+0.25, 1000)[:,np.newaxis]
kde_rpknewtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(my_pd['r_pkmag']).reshape(-1,1))
log_dens_rpknewtrain = kde_rpknewtrain.score_samples(rpknewtrain_axis)

rpkoldtrain_axis = np.linspace(min(my_pd2['r_pkmag'])-0.25, max(my_pd2['r_pkmag'])+0.25, 1000)[:,np.newaxis]
kde_rpkoldtrain = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(my_pd2['r_pkmag']).reshape(-1,1))
log_dens_rpkoldtrain = kde_rpkoldtrain.score_samples(rpkoldtrain_axis)


plt.figure()
plt.subplot(1,3,1)
plt.plot(zoldtrain_axis, np.exp(log_dens_zoldtrain), label='old train')
plt.plot(znewtrain_axis, np.exp(log_dens_znewtrain), label='new train')
plt.xlabel('redshift')
plt.legend()

plt.subplot(1,3,2)
plt.plot(gpkoldtrain_axis, np.exp(log_dens_gpkoldtrain), label='old train')
plt.plot(gpknewtrain_axis, np.exp(log_dens_gpknewtrain), label='new train')
plt.xlabel('g_peakmag')
plt.xlim(15,35)
plt.legend()

plt.subplot(1,3,3)
plt.plot(rpkoldtrain_axis, np.exp(log_dens_rpkoldtrain), label='old train')
plt.plot(rpknewtrain_axis, np.exp(log_dens_rpknewtrain), label='new train')
plt.xlabel('g_peakmag')
plt.xlim(15,35)
plt.legend()

plt.savefig('pseudotrain.png')


