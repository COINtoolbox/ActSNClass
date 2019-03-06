import numpy as np
import seaborn as sns
import pylab as plt


QBC = False
points = False
output_file = 'diag_fullsample_fullLC.png'

batchSize = 1
diag_path = ['random_RF/diag_random_RF_fullsample_canonical.csv', 
             'random_RF/diag_random_RF_fullsample.csv', 
             'unc_RF/diag_unc_RF_fullsample.csv']

acc = []
eff = []
pur = []
fom = []




for j in range(len(diag_path)):
    # read diagnostics
    op1 = open(diag_path[j], 'r')
    lin1 = op1.readlines()
    op1.close()

    data_str = [elem.split(',') for elem in lin1]
    acc.append(np.array([[int(data_str[k][0]), float(data_str[k][2])] for k in range(1, len(data_str), batchSize)]))
    eff.append(np.array([[int(data_str[k][0]), float(data_str[k][3])] for k in range(1, len(data_str), batchSize)]))
    pur.append(np.array([[int(data_str[k][0]), float(data_str[k][4])] for k in range(1, len(data_str), batchSize)]))
    fom.append(np.array([[int(data_str[k][0]), float(data_str[k][5])] for k in range(1, len(data_str),batchSize)]))


acc = np.array(acc)
eff = np.array(eff)
pur = np.array(pur)
fom = np.array(fom)

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

fig = plt.figure(figsize=(20,14))
ax1 = plt.subplot(2,2,1)
if points:
    l0 = ax1.scatter(acc[0][:,0], acc[0][:,1], color='#dd0100', marker='v', s=70, label='Canonical')
    l1 = ax1.scatter(acc[1][:,0], acc[1][:,1], color='#fac901', marker='o', s=50, label='Passive Learning')
    l2 = ax1.scatter(acc[2][:,0], acc[2][:,1], color='#225095', marker='^', s=70, label='AL: Uncertainty sampling')
    l10 = ax1.plot([1],[1], color='white', label='             ')
    if QBC:
        l3 = ax1.scatter(acc[3][:,0], acc[3][:,1], color='#30303a', marker='s', s=50, label='AL: Query by Commitee')
else:
    l0 = ax1.plot(acc[0][:,0], acc[0][:,1], color='#dd0100', ls=':', lw=5.0, label='Canonical')
    l1 = ax1.plot(acc[1][:,0], acc[1][:,1], color='#fac901', ls='--', lw=5.0, label='Passive Learning')
    l2 = ax1.plot(acc[2][:,0], acc[2][:,1], color='#225095', ls='-.', lw=5.0, label='AL: Uncertainty sampling')
    if QBC:
        l3 = ax1.plot(acc[3][:,0], acc[3][:,1], color='#30303a', lw=5.0, label='AL: QBC1')
        l4 = ax1.plot(acc[4][:,0], acc[4][:,1], color='cyan', ls='-.', lw=5.0, label='AL: QBC2')
        l5 = ax1.plot(acc[5][:,0], acc[5][:,1], color='grey', ls='--', lw=5.0, label='AL: QBC3')


ax1.set_yticks(ax1.get_yticks())
ax1.set_yticklabels([round(item, 2) for item in ax1.get_yticks()], fontsize=22)
ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels([int(item) for item in ax1.get_xticks()], fontsize=22)
ax1.set_xlabel('Number of queries', fontsize=30)
ax1.set_ylabel('Accuracy', fontsize=30)
ax1.set_xlim(0, 1000)

ax2 = plt.subplot(2,2,2)
if points:
    ax2.scatter(eff[0][:,0], eff[0][:,1], color='#dd0100', marker='v', s=70)
    ax2.scatter(eff[1][:,0], eff[1][:,1], color='#fac901', marker='o', s=50)
    ax2.scatter(eff[2][:,0], eff[2][:,1], color='#225095', marker='^', s=70)
    if QBC:
        ax2.scatter(eff[3][:,0], eff[3][:,1], color='#30303a', marker='s', s=50)
else:
    ax2.plot(eff[0][:,0], eff[0][:,1], color='#dd0100', ls=':', lw=5.0)
    ax2.plot(eff[1][:,0], eff[1][:,1], color='#fac901', ls='--', lw=5)
    ax2.plot(eff[2][:,0], eff[2][:,1], color='#225095', ls='-.', lw=5)
    if QBC:
        ax2.plot(eff[3][:,0], eff[3][:,1], color='#30303a', lw=5.0)
        ax2.plot(eff[4][:,0], eff[4][:,1], color='cyan', lw=5.0)
        ax2.plot(eff[5][:,0], eff[5][:,1], color='grey', lw=5.0)

ax2.set_xlabel('Number of queries', fontsize=30)
ax2.set_ylabel('Efficiency', fontsize=30)
ax2.set_xticks(ax2.get_xticks())
ax2.set_xticklabels([int(item) for item in ax2.get_xticks()], fontsize=22)
ax2.set_yticklabels([round(item, 2) for item in ax2.get_yticks()], fontsize=22)
ax2.set_xlim(0, 1000)

ax3 = plt.subplot(2,2,3)
if points:
    ax3.scatter(pur[0][:,0], pur[0][:,1], color='#dd0100', marker='v', s=70)
    ax3.scatter(pur[1][:,0], pur[1][:,1], color='#fac901', marker='o', s=50)
    ax3.scatter(pur[2][:,0], pur[2][:,1], color='#225095', marker='^', s=70)
    if QBC:
        ax3.scatter(pur[3][:,0], pur[3][:,1], color='#30303a', marker='s', s=50)
else:
    ax3.plot(pur[0][:,0], pur[0][:,1], color='#dd0100', ls=':', lw=5.0)
    ax3.plot(pur[1][:,0], pur[1][:,1], color='#fac901', ls='--', lw=5.0)
    ax3.plot(pur[2][:,0], pur[2][:,1], color='#225095', ls='-.', lw=5.0)
    if QBC:
        ax3.plot(pur[3][:,0], pur[3][:,1], color='#30303a', lw=5.0)
        ax3.plot(pur[4][:,0], pur[4][:,1], color='cyan', lw=5.0)
        ax3.plot(pur[5][:,0], pur[5][:,1], color='grey', lw=5.0)

ax3.set_xlabel('Number of queries', fontsize=30)
ax3.set_ylabel('Purity', fontsize=30)
ax3.set_xticks(ax3.get_xticks())
ax3.set_xticklabels([int(item) for item in ax3.get_xticks()], fontsize=22)
ax3.set_yticklabels([round(item,2) for item in ax3.get_yticks()], fontsize=22)
ax3.set_xlim(0, 1000)


ax4 = plt.subplot(2,2,4)
if points:
    ax4.scatter(fom[0][:,0], fom[0][:,1], color='#dd0100', marker='v', s=70)
    ax4.scatter(fom[1][:,0], fom[1][:,1], color='#fac901', marker='o', s=50)
    ax4.scatter(fom[2][:,0], fom[2][:,1], color='#225095', marker='^', s=70)
    if QBC:
        ax4.scatter(fom[3][:,0], fom[3][:,1], color='#30303a', marker='s', s=50)
else:
    ax4.plot(fom[0][:,0], fom[0][:,1], color='#dd0100', ls=':', lw=5.0)
    ax4.plot(fom[1][:,0], fom[1][:,1], color='#fac901', ls='--', lw=5.0)
    ax4.plot(fom[2][:,0], fom[2][:,1], color='#225095', ls='-.', lw=5.0)
    if QBC:
        ax4.plot(fom[3][:,0], fom[3][:,1], color='#30303a', lw=5.0)
        ax4.plot(fom[4][:,0], fom[4][:,1], color='cyan', lw=5.0)
        ax4.plot(fom[5][:,0], fom[5][:,1], color='grey', lw=5.0)

ax4.set_xlabel('Number of queries', fontsize=30)
ax4.set_ylabel('Figure of merit', fontsize=30)
ax4.set_xticks(ax4.get_xticks())
ax4.set_xticklabels([int(item) for item in ax4.get_xticks()], fontsize=22)
ax4.set_yticks(ax4.get_yticks())
ax4.set_yticklabels([round(item, 2) for item in ax4.get_yticks()], fontsize=22)
ax4.set_xlim(0, 1000)

handles, labels = ax1.get_legend_handles_labels()
ph = [plt.plot([], marker="", ls="")[0]]
h = ph + handles
l = labels
lgd = ax1.legend(h[1:], l, loc='upper center', bbox_to_anchor=(1.025,1.35), ncol=3, fontsize=25, title='Strategies:')
plt.setp(lgd.get_title(),fontsize='23')


plt.subplots_adjust(left=0.075, right=0.95, top=0.875, bottom=0.075, wspace=0.35, hspace=0.35)
plt.savefig(output_file)

