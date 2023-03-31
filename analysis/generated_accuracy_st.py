import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# overall corr matrix

"""corrs = np.ones([9, 9])
index1 = 0
for i in range(10):
    if i != 8:
        index2 = 0
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                fake = np.average(fake, axis=2)
                real = np.load('eeg_data/test/st_sub' + str(j + 1).zfill(2) + '.npy')
                real = np.average(real, axis=2)
                corrs[index1, index2] = spearmanr(fake.flatten(), real.flatten())[0]
                print(index1, index2)
            if j != 8:
                index2 += 1
        index1 += 1

print(corrs)

plt.imshow(corrs, extent=(0, 1, 0, 1), clim=(0, 1))
subs = ['Sub01', 'Sub02', 'Sub03', 'Sub04', 'Sub05', 'Sub06', 'Sub07', 'Sub08', 'Sub10']
step = float(1/9)
for i in range(9):
    for j in range(9):
        plt.text(i*step+0.5*step, 1-j*step-0.5*step, float('%.4f' % corrs[j, i]),
                 ha='center', va='center', fontsize=8)
        x = np.arange(0.5*step, 1+0.5*step, step)
        y = np.arange(1-0.5*step, -0.5*step, -step)
        plt.xticks(x, subs, fontsize=9)
        plt.yticks(y, subs, fontsize=9)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
font = {'size': 14}
cb.set_label('Generated Accuracy (Overall)', fontdict=font)
plt.xlabel('Target Subject', fontsize=13)
plt.ylabel('Source Subject', fontsize=13)
plt.savefig('Analysis_results/generated_accuracy_st/overall_corr_martix.jpg', dpi=300)"""

# overall, pattern, profile corrs of the full model

"""overall_corrs = np.zeros([72])
pattern_corrs = np.zeros([72])
profile_corrs = np.zeros([72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                fake = np.average(fake, axis=2)
                real = np.load('eeg_data/test/st_sub' + str(j + 1).zfill(2) + '.npy')
                real = np.average(real, axis=2)
                overall_corrs[index] = spearmanr(fake.flatten(), real.flatten())[0]
                for k in range(200):
                    pattern_corrs[index] += spearmanr(fake[:, k].flatten(), real[:, k].flatten())[0]/200
                    profile_corrs[index] += spearmanr(fake[:, :, k].flatten(), real[:, :, k].flatten())[0]/200
                index += 1

y = np.zeros([3, 72])
y[0] = overall_corrs
y[1] = pattern_corrs
y[2] = profile_corrs
np.save('Analysis_results/avgtrials_comparisons_st/3corrs.npy', y)
y = np.load('Analysis_results/avgtrials_comparisons_st/3corrs.npy')
y_avg = np.average(y, axis=1)
y_err = np.zeros([3])
for i in range(3):
    y_err[i] = np.std(y[i], ddof=1)/np.sqrt(72)

x = np.array([0, 1, 2])
w = 0.3
fig, ax = plt.subplots()
for i in range(3):
    ax.scatter(x[i] + np.random.random(y[i].size)*w/8-w/16, y[i], s=15, color='silver', edgecolors='grey', alpha=0.5, zorder=1)
ax.bar(x, y_avg, color='white', edgecolor='black', yerr=y_err, width=w, capsize=6, ecolor='black', visible=False, zorder=2)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(x, ['Overall', 'Pattern', 'Profile'], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Generated Accuracy', fontsize=18)
plt.savefig('Analysis_results/generated_accuracy_st/corrs_bar.jpg', dpi=300)"""

# Time-by-time correlation
"""corrs = np.zeros([200, 72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                fake = np.average(fake, axis=2)
                real = np.load('eeg_data/test/st_sub' + str(j + 1).zfill(2) + '.npy')
                real = np.average(real, axis=2)
                print(real.shape)
                for k in range(200):
                    for t in range(200):
                        corrs[t, index] += spearmanr(fake[:, k, t], real[:, k, t])[0]/200
                index += 1
np.save('Analysis_results/generated_accuracy_st/tbyt_corrs.npy', corrs)"""
corrs = np.load('Analysis_results/generated_accuracy_st/tbyt_corrs.npy')
corrs_avg = np.average(corrs, axis=1)
corrs_err = np.zeros([200])
for t in range(200):
    corrs_err[t] = np.std(corrs[t], ddof=1)/np.sqrt(72)
t = np.arange(0, 200, 1)
plt.errorbar(t, corrs_avg, yerr=corrs_err, color='black', ecolor='black', elinewidth=0.2, capthick=0.4, marker='o', markersize=0.1, capsize=2)
plt.ylabel('Generated Accuracy', fontsize=18)
plt.xlabel('Time (ms)', fontsize=18)
plt.savefig('Analysis_results/generated_accuracy_st/tbyt_corrs.jpg', dpi=300)
plt.show()

# Channel-by-channel correlation
"""corrs = np.zeros([17, 72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                fake = np.average(fake, axis=2)
                real = np.load('eeg_data/test/st_sub' + str(j + 1).zfill(2) + '.npy')
                real = np.average(real, axis=2)
                print(real.shape)
                for k in range(200):
                    for ch in range(17):
                        corrs[ch, index] += spearmanr(fake[ch, k, :], real[ch, k, :])[0]/200
                index += 1
np.save('Analysis_results/generated_accuracy_st/chbych_corrs.npy', corrs)
corrs = np.load('Analysis_results/generated_accuracy_st/chbych_corrs.npy')
corrs_avg = np.average(corrs, axis=1)
corrs_err = np.zeros([17])
for ch in range(17):
    corrs_err[ch] = np.std(corrs[ch], ddof=1)/np.sqrt(72)
ch = np.arange(0, 17, 1)
plt.errorbar(ch, corrs_avg, yerr=corrs_err, color='black', ecolor='black', marker='o', markersize=0.1, capsize=2)
channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5',
            'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
plt.xticks(ch, channels, fontsize=10, rotation=60, ha='right')
plt.ylabel('Generated Accuracy', fontsize=18)
plt.savefig('Analysis_results/generated_accuracy_st/chbych_corrs.jpg', dpi=300)
plt.show()"""