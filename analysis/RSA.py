import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from neurora.rsa_plot import plot_rdm, plot_tbytsim_withstats
from neurora.rdm_cal import eegRDM
from neurora.corr_cal_by_rdm import rdms_corr

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
np.save('Analysis_results/RSA/animate_RDM.npy', animate_rdm)"""

# Get Low-level RDM
"""lw_rdm = np.zeros([200, 200])

for i in range(200):
    for j in range(200):
        if i > j:
            img1 = np.array(Image.open('images/' + str(i) + '.jpg'))
            img2 = np.array(Image.open('images/' + str(j) + '.jpg'))
            lw_rdm[i, j] = 1 - pearsonr(np.reshape(img1, [750000]), np.reshape(img2, [750000]))[0]
            lw_rdm[j, i] = lw_rdm[i, j]

np.save('Analysis_results/RSA/lowlevel_RDM.npy', lw_rdm)"""

# Get real EEG rdms
"""real = np.zeros([200, 9, 1, 17, 200])
index = 0
for i in range(10):
    if i != 8:
        subreal = np.load('eeg_data/test/sub' + str(i + 1).zfill(2) + '.npy')
        subreal = np.transpose(subreal, (1, 0, 2))
        real[:, index] = np.reshape(subreal, [200, 1, 17, 200])
        index += 1
realRDMs = eegRDM(real, time_opt=1, time_step=10, time_win=10)
np.save('Analysis_results/RSA/realRDMs.npy', realRDMs)"""

# Get fake EEG rdms (from Sub01)

"""for i in range(10):
    if i != 8:
        fake = np.zeros([200, 8, 1, 17, 200])
        index = 0
        for j in range(10):
            if j != i and j != 8:
                subfake = np.load('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                subfake = np.transpose(subfake, (1, 0, 2))
                fake[:, index] = np.reshape(subfake, [200, 1, 17, 200])
                index += 1
        fakeRDMs = eegRDM(fake, time_opt=1, time_step=10, time_win=10)
        np.save('Analysis_results/RSA/fakeRDMs_fromSub' + str(i + 1) + '.npy', fakeRDMs)"""

# Correlation with Low-level RDM

lw_rdm = np.load('Analysis_results/RSA/Lowlevel_RDM.npy')
realRDMs = np.load('Analysis_results/RSA/realRDMs.npy')
print(realRDMs.shape)

real_lw_corrs = rdms_corr(lw_rdm, realRDMs)[:, :, 0]
print(real_lw_corrs.shape)

fake_lw_corrs = np.zeros([72, 20])
index = 0
for i in range(10):
    if i != 8:
        fakeRDMs = np.load('Analysis_results/RSA/fakeRDMs_fromSub' + str(i+1) + '.npy')
        print(fakeRDMs.shape)
        fake_lw_corrs[index*8:index*8+8] = rdms_corr(lw_rdm, fakeRDMs)[:, :, 0]
        index += 1

plot_tbytsim_withstats(real_lw_corrs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[-0.05, 0.1], avgshow=True)
plot_tbytsim_withstats(fake_lw_corrs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[-0.05, 0.1], avgshow=True)

animate_rdm = np.load('Analysis_results/RSA/animate_RDM.npy')

real_animate_corrs = rdms_corr(animate_rdm, realRDMs)[:, :, 0]
print(real_animate_corrs.shape)

fake_animate_corrs = np.zeros([72, 20])
index = 0
for i in range(10):
    if i != 8:
        fakeRDMs = np.load('Analysis_results/RSA/fakeRDMs_fromSub' + str(i+1) + '.npy')
        print(fakeRDMs.shape)
        fake_animate_corrs[index*8:index*8+8] = rdms_corr(animate_rdm, fakeRDMs)[:, :, 0]
        index += 1

plot_tbytsim_withstats(real_animate_corrs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[-0.05, 0.1], avgshow=True)
plot_tbytsim_withstats(fake_animate_corrs, start_time=0, end_time=0.2, time_interval=0.01, xlim=[0, 0.2], ylim=[-0.05, 0.1], avgshow=True)
