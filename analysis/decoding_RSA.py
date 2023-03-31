import numpy as np
from neurora.rdm_cal import eegRDM_bydecoding
from neurora.rsa_plot import plot_tbyt_decoding_acc, plot_tbytsim_withstats, plot_rdm
from neurora.corr_cal_by_rdm import rdms_corr
from neurora.rdm_corr import rdm_correlation_spearman
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

"""for i in range(10):
    if i != 8:
        real = np.load('eeg_data/test/st_sub' + str(i + 1).zfill(2) + '.npy')
        real = np.transpose(real, (1, 2, 0, 3))
        # data: [nconditions * ntrials * nchannels * nts]

        real = np.reshape(real, [200, 1, 80, 17, 200])

        real_eegRDMs = eegRDM_bydecoding(real, sub_opt=1, time_win=10, time_step=10, nfolds=4, nrepeats=5)[0]

        np.save('Analysis_results/decoding_RSA/realeegRDMs_Sub' + str(i+1) + '.npy', real_eegRDMs)


for i in range(10):
    if i != 8:
        fake_eegRDMs = np.zeros([8, 20, 200, 200])
        index = 0
        for j in range(10):
            if j != 8 and j != i:
                fake = fake = np.load('generated_fullmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                fake = np.transpose(fake, (1, 2, 0, 3))
                # data: [nconditions * ntrials * nchannels * nts]

                fake = np.reshape(fake, [200, 1, 80, 17, 200])

                fake_eegRDMs[index] = eegRDM_bydecoding(fake, sub_opt=1, time_win=10, time_step=10, nfolds=4, nrepeats=5)[0]

                index += 1

        np.save('Analysis_results/decoding_RSA/fakeeegRDMs_fromSub' + str(i+1) + '.npy', fake_eegRDMs)"""

# real - acc

realaccs = np.zeros([9, 20])
index1 = 0
for i1 in range(10):
    if i1 != 8:
        realRDMs = np.load('Analysis_results/decoding_RSA/realeegRDMs_Sub' + str(i1+1) + '.npy')
        for j in range(200):
            for k in range(200):
                if j > k:
                    realaccs[index1] += realRDMs[:, j, k]
        index1 += 1
realaccs = realaccs/19900

plot_tbyt_decoding_acc(realaccs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[0.4, 0.9],
                       cbpt=False, avgshow=True)

# fake - acc

fakeaccs = np.zeros([72, 20])
index1 = 0
for i1 in range(10):
    if i1 != 8:
        fakeRDMs = np.load('Analysis_results/decoding_RSA/fakeeegRDMs_fromSub' + str(i1+1) + '.npy')
        for i2 in range(8):
            for j in range(200):
                for k in range(200):
                    if j > k:
                        fakeaccs[index1*8+i2] += fakeRDMs[i2, :, j, k]
        index1 += 1
fakeaccs = fakeaccs/19900

plot_tbyt_decoding_acc(fakeaccs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[0.4, 0.9], cbpt=False,
                       avgshow=True)

# real - acc - image 1 vs 2

"""realaccs = np.zeros([9, 20])
index1 = 0
for i1 in range(10):
    if i1 != 8:
        realRDMs = np.load('Analysis_results/decoding_RSA/realeegRDMs_Sub' + str(i1+1) + '.npy')
        print(realRDMs.shape)
        realaccs[index1] = realRDMs[:, 0, 1]
        index1 += 1

plot_tbyt_decoding_acc(realaccs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[0.45, 0.7],
                       cbpt=False, avgshow=True)"""

# fake - acc image 1 vs 2

"""fakeaccs = np.zeros([72, 20])
index1 = 0
for i1 in range(10):
    if i1 != 8:
        fakeRDMs = np.load('Analysis_results/decoding_RSA/fakeeegRDMs_fromSub' + str(i1+1) + '.npy')
        for i2 in range(8):
            fakeaccs[index1*8+i2] += fakeRDMs[i2, :, 0, 1]
        index1 += 1

plot_tbyt_decoding_acc(fakeaccs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[0.45, 0.7], cbpt=False,
                       avgshow=True)"""

"""fakeaccs = np.zeros([9, 20])
index1 = 0
for i1 in range(10):
    if i1 != 8:
        fakeRDMs = np.load('Analysis_results/decoding_RSA/fakeeegRDMs_fromSub' + str(i1+1) + '.npy')
        index2 = 0
        for i2 in range(9):
            if i2 != index1:
                for j in range(200):
                    for k in range(200):
                        if j > k:
                            fakeaccs[i2] += fakeRDMs[index2, :, j, k]/8
                index2 += 1
        index1 += 1
fakeaccs = fakeaccs/19900

plot_tbyt_decoding_acc(fakeaccs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[0.4, 1.0], cbpt=False,
                       avgshow=True)"""

# EEG RDMs - Sub 01
"""fakeRDMs = np.load('Analysis_results/decoding_RSA/fakeeegRDMs_fromSub2.npy')[0]
print(fakeRDMs.shape)
for i in range(20):
    plot_rdm(fakeRDMs[i], percentile=True)"""

# DCNN RDMs

""" i in range(8):
    model_rdm = np.load('Analysis_results/decoding_RSA/alexnet_ly' + str(i+1) + '_RDM.npy')
    plot_rdm(model_rdm, percentile=True)"""

# RDM similarity between real and fake

"""sims = np.zeros([72, 20])

index1 = 0
for i1 in range(10):
    if i1 != 8:
        realRDMs = np.load('Analysis_results/decoding_RSA/realeegRDMs_Sub' + str(i1+1) + '.npy')
        fakeRDMs = np.load('Analysis_results/decoding_RSA/fakeeegRDMs_fromSub' + str(i1+1) + '.npy')
        print(realRDMs.shape)
        print(fakeRDMs.shape)
        index2 = 0
        for i2 in range(9):
            if i2 != index1:
                for t in range(20):
                    sims[index1*8+index2, t] = rdm_correlation_spearman(realRDMs[t], fakeRDMs[index2, t])[0]
                index2 += 1
        index1 += 1

np.save('Analysis_results/decoding_RSA/sims_betweenrealandfake.npy', sims)"""
"""sims = np.load('Analysis_results/decoding_RSA/sims_betweenrealandfake.npy')

sims_avg = np.average(sims, axis=0)
sims_err = np.zeros([20])
for t in range(20):
    sims_err[t] = np.std(sims[:, t], ddof=1)/np.sqrt(72)

t = np.arange(0, 200, 10)
plt.errorbar(t, sims_avg, color='black', yerr=sims_err, elinewidth=0.2, capthick=0.4, marker='o', markersize=0.1, capsize=2)
plt.ylabel('Similarity', fontsize=18)
plt.xlabel('Time (ms)', fontsize=18)
plt.savefig('Analysis_results/decoding_RSA/sims_betweenrealandfake.jpg', dpi=300)"""


# Get Animate-Inanimate RDM
"""meta = np.load('images/image_metadata.npy', allow_pickle=True)

tsv_data = pd.read_csv('images/things_concepts.tsv', sep='\t', header=0)
names_fromtsv = np.array(tsv_data)[:, 1]
categories_fromtsv = np.array(tsv_data)[:, 17]

names = []
v = np.zeros([200])
n = 0
for i in range(200):
    name = meta.item()['test_img_concepts'][i][6:]
    index = np.where(names_fromtsv == name)[0]
    category = categories_fromtsv[index][0]
    if 'animal' in category or 'bird' in category or 'insect' in category or 'dog' in category or \
            'music player' in category or 'snake' in category or 'crustacean' in category:
        v[i] = 1
        n += 1
print(n)
animate_rdm = np.zeros([200, 200])
for i in range(200):
    for j in range(200):
        if v[i] != v[j]:
            animate_rdm[i, j] = 1
np.save('Analysis_results/decoding_RSA/animate_RDM.npy', animate_rdm)"""

# Get Low-level RDM
"""lw_rdm = np.zeros([200, 200])

for i in range(200):
    for j in range(200):
        if i > j:
            img1 = np.array(Image.open('images/' + str(i) + '.jpg'))
            img2 = np.array(Image.open('images/' + str(j) + '.jpg'))
            lw_rdm[i, j] = 1 - pearsonr(np.reshape(img1, [750000]), np.reshape(img2, [750000]))[0]
            lw_rdm[j, i] = lw_rdm[i, j]

np.save('Analysis_results/decoding_RSA/lowlevel_RDM.npy', lw_rdm)"""

# RSA with Animate-inanimate RDM

"""allrealRDMs = np.zeros([9, 20, 200, 200])
allfakeRDMs = np.zeros([72, 20, 200, 200])

index1 = 0
for i in range(10):
    if i != 8:
        allrealRDMs[index1] = np.load('Analysis_results/decoding_RSA/realeegRDMs_Sub' + str(i+1) + '.npy')
        fakeRDMs = np.load('Analysis_results/decoding_RSA/fakeeegRDMs_fromSub' + str(i+1) + '.npy')
        index2 = 0
        index3 = 0
        for j in range(10):
            if j != 8 and j != i:
                allfakeRDMs[index1*8+index2] = fakeRDMs[index2]
                index2 += 1
        index1 += 1

np.save('Analysis_results/decoding_RSA/allrealRDMs.npy', allrealRDMs)
np.save('Analysis_results/decoding_RSA/allfakeRDMs.npy', allfakeRDMs)

allrealRDMs = np.load('Analysis_results/decoding_RSA/allrealRDMs.npy')
allfakeRDMs = np.load('Analysis_results/decoding_RSA/allfakeRDMs.npy')

real_corrs = np.zeros([9, 9, 20])
fake_corrs = np.zeros([9, 72, 20])

lw_rdm = np.load('Analysis_results/decoding_RSA/lowlevel_RDM.npy')
real_corrs[0] = rdms_corr(lw_rdm, allrealRDMs)[:, :, 0]
fake_corrs[0] = rdms_corr(lw_rdm, allfakeRDMs)[:, :, 0]

for i in range(8):
    model_rdm = np.load('Analysis_results/decoding_RSA/alexnet_ly' + str(i+1) + '_RDM.npy')
    real_corrs[i+1] = rdms_corr(model_rdm, allrealRDMs)[:, :, 0]
    fake_corrs[i+1] = rdms_corr(model_rdm, allfakeRDMs)[:, :, 0]

np.save('Analysis_results/decoding_RSA/rsa_results_real.npy', real_corrs)
np.save('Analysis_results/decoding_RSA/rsa_results_fake.npy', fake_corrs)"""

"""real_corrs = np.load('Analysis_results/decoding_RSA/rsa_results_real.npy')
fake_corrs = np.load('Analysis_results/decoding_RSA/rsa_results_fake.npy')

labels = ['Low-level', 'Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'Layer7', 'Layer8']
colors = ['indigo', 'rebeccapurple', 'darkviolet', 'darkorchid', 'mediumorchid', 'orchid', 'violet', 'plum', 'thistle']

x = np.arange(5, 205, 10)

for i in range(9):
    plt.plot(x, np.average(real_corrs, axis=1)[-i], color=colors[-i], linewidth=5, label=labels[-i], alpha=0.8)
    for t in range(20):
        tvalue, pvalue = ttest_1samp(real_corrs[-i, :, t], 0, alternative='greater')
        if pvalue < 0.05:
            plt.plot(x[t], 0.585-i*0.012, 'o', color=colors[-i], alpha=0.8, markersize=3)

plt.legend()
plt.ylim(-0.015, 0.6)
plt.xlabel('Time (ms)', fontsize=15)
plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200], fontsize=10)
plt.ylabel('Representational Similarity', fontsize=16)
plt.yticks(fontsize=10)
plt.title('Real data', fontsize=18)
plt.savefig('Analysis_results/decoding_RSA/real_rsa.jpg', dpi=300)
plt.show()

for i in range(9):
    plt.plot(x, np.average(fake_corrs, axis=1)[-i], color=colors[-i], linewidth=5, label=labels[-i], alpha=0.8)
    for t in range(20):
        tvalue, pvalue = ttest_1samp(fake_corrs[-i, :, t], 0, alternative='greater')
        if pvalue < 0.05:
            plt.plot(x[t], 0.585-i*0.012, 'o', color=colors[-i], alpha=0.8, markersize=3)

plt.legend()
plt.ylim(-0.015, 0.6)
plt.xlabel('Time (ms)', fontsize=15)
plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 200], fontsize=10)
plt.ylabel('Representational Similarity', fontsize=16)
plt.yticks(fontsize=10)
plt.title('Generated data', fontsize=18)
plt.savefig('Analysis_results/decoding_RSA/fake_rsa.jpg', dpi=300)
plt.show()"""