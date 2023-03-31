import numpy as np
from scipy.stats import spearmanr, ttest_rel
import matplotlib.pyplot as plt

# overall corr matrix - full model

"""corrs = np.ones([9, 9])
index1 = 0
for i in range(10):
    if i != 8:
        index2 = 0
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                corrs[index1, index2] = spearmanr(fake.flatten(), real.flatten())[0]
                print(index1, index2)
            if j != 8:
                index2 += 1
        index1 += 1

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
plt.savefig('Analysis_results/generated_accuracy/overall_corr_martix.jpg', dpi=300)"""

# overall corr matrix - linear model

"""corrs = np.ones([9, 9])
index1 = 0
for i in range(10):
    if i != 8:
        index2 = 0
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_linearmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                corrs[index1, index2] = spearmanr(fake.flatten(), real.flatten())[0]
                print(index1, index2)
            if j != 8:
                index2 += 1
        index1 += 1

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
plt.savefig('Analysis_results/generated_accuracy/overall_corr_martix_linearmodel.jpg', dpi=300)"""

# overall, pattern, profile corrs of the full model

"""overall_corrs = np.zeros([72])
pattern_corrs = np.zeros([72])
profile_corrs = np.zeros([72])
overall_corrs_linear = np.zeros([72])
pattern_corrs_linear = np.zeros([72])
profile_corrs_linear = np.zeros([72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                fake_linear = np.load('generated_linearmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                overall_corrs[index] = spearmanr(fake.flatten(), real.flatten())[0]
                overall_corrs_linear[index] = spearmanr(fake_linear.flatten(), real.flatten())[0]
                for k in range(200):
                    pattern_corrs[index] += spearmanr(fake[:, k].flatten(), real[:, k].flatten())[0]/200
                    profile_corrs[index] += spearmanr(fake[:, :, k].flatten(), real[:, :, k].flatten())[0]/200
                    pattern_corrs_linear[index] += spearmanr(fake_linear[:, k].flatten(), real[:, k].flatten())[0]/200
                    profile_corrs_linear[index] += spearmanr(fake_linear[:, :, k].flatten(), real[:, :, k].flatten())[0]/200
                index += 1

x = np.array([0, 1, 2])
w = 0.3

y = np.zeros([3, 72])
y[0] = overall_corrs
y[1] = pattern_corrs
y[2] = profile_corrs
np.save('Analysis_results/avgtrials_comparisons/3corrs.npy', y)
y = np.load('Analysis_results/avgtrials_comparisons/3corrs.npy')
y_avg = np.average(y, axis=1)
y_err = np.zeros([3])
for i in range(3):
    y_err[i] = np.std(y[i], ddof=1)/np.sqrt(72)
fig, ax = plt.subplots()
for i in range(3):
    if i == 0:
        ax.scatter(x[i]+w/2 + np.random.random(y[i].size)*w/8-w/16, y[i], s=15, color='yellowgreen',
                   edgecolors='olive', alpha=0.5, zorder=1, label='EEG2EEG Model')
    else:
        ax.scatter(x[i] + w / 2 + np.random.random(y[i].size) * w / 8 - w / 16, y[i], s=15, color='yellowgreen',
                   edgecolors='olive', alpha=0.5, zorder=1)
ax.bar(x+w/2, y_avg, color='darkolivegreen', edgecolor='darkolivegreen', yerr=y_err, width=w, capsize=6, ecolor='darkolivegreen', visible=False, zorder=2)

y_linear = np.zeros([3, 72])
y_linear[0] = overall_corrs_linear
y_linear[1] = pattern_corrs_linear
y_linear[2] = profile_corrs_linear
np.save('Analysis_results/avgtrials_comparisons/3corrs_linear.npy', y_linear)
y_linear = np.load('Analysis_results/avgtrials_comparisons/3corrs_linear.npy')
y_linear_avg = np.average(y_linear, axis=1)
y_linear_err = np.zeros([3])
for i in range(3):
    y_linear_err[i] = np.std(y_linear[i], ddof=1)/np.sqrt(72)
for i in range(3):
    if i == 0:
        ax.scatter(x[i]-w/2 + np.random.random(y_linear[i].size)*w/8-w/16, y_linear[i], s=15, color='lightgreen',
                   edgecolors='limegreen', alpha=0.5, zorder=1, label='Linear model')
    else:
        ax.scatter(x[i] - w / 2 + np.random.random(y_linear[i].size) * w / 8 - w / 16, y_linear[i], s=15, color='lightgreen',
                   edgecolors='limegreen', alpha=0.5, zorder=1)
ax.bar(x-w/2, y_linear_avg, color='forestgreen', edgecolor='forestgreen', yerr=y_linear_err, width=w, capsize=6, ecolor='forestgreen', visible=False, zorder=2)

print(ttest_rel(y[0], y_linear[0]))
print(ttest_rel(y[1], y_linear[1]))
print(ttest_rel(y[2], y_linear[2]))

ax = plt.gca()
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(x, ['Overall', 'Pattern', 'Profile'], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Generated Accuracy', fontsize=18)
plt.ylim(0.1, 0.9)
plt.plot([-0.15, 0.15], [0.85, 0.85], color='black')
plt.text(-0.08, 0.84, '***', fontsize=16)
plt.plot([0.85, 1.15], [0.86, 0.86], color='black')
plt.text(0.92, 0.85, '***', fontsize=16)
plt.plot([1.85, 2.15], [0.7, 0.7], color='black')
plt.text(1.92, 0.69, '***', fontsize=16)
plt.savefig('Analysis_results/generated_accuracy/corrs_bar.jpg', dpi=300)
plt.show()"""

# Time-by-time correlation
"""corrs = np.zeros([200, 72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                print(real.shape)
                for k in range(200):
                    for t in range(200):
                        corrs[t, index] += spearmanr(fake[:, k, t], real[:, k, t])[0]/200
                index += 1
np.save('Analysis_results/generated_accuracy/tbyt_corrs.npy', corrs)"""
"""corrs = np.load('Analysis_results/generated_accuracy/tbyt_corrs.npy')
corrs_avg = np.average(corrs, axis=1)
corrs_err = np.zeros([200])
for t in range(200):
    corrs_err[t] = np.std(corrs[t], ddof=1)/np.sqrt(72)"""

"""corrs_linear = np.zeros([200, 72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake_linear = np.load('generated_linearmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                print(real.shape)
                for k in range(200):
                    for t in range(200):
                        corrs_linear[t, index] += spearmanr(fake_linear[:, k, t], real[:, k, t])[0]/200
                index += 1
np.save('Analysis_results/generated_accuracy/tbyt_corrs_linear.npy', corrs_linear)"""
"""corrs_linear = np.load('Analysis_results/generated_accuracy/tbyt_corrs_linear.npy')
corrs_avg_linear = np.average(corrs_linear, axis=1)
corrs_err_linear = np.zeros([200])
for t in range(200):
    corrs_err_linear[t] = np.std(corrs_linear[t], ddof=1)/np.sqrt(72)

t = np.arange(0, 200, 1)
plt.errorbar(t, corrs_avg, yerr=corrs_err, color='yellowgreen', ecolor='olive', elinewidth=0.2, capthick=0.4, marker='o', markersize=0.1, capsize=2, label='EEG2EEG model')
plt.errorbar(t, corrs_avg_linear, yerr=corrs_err_linear, color='lightgreen', ecolor='limegreen', elinewidth=0.2, capthick=0.4, marker='o', markersize=0.1, capsize=2, label='Linear model')
plt.ylabel('Generated Accuracy', fontsize=18)
plt.xlabel('Time (ms)', fontsize=18)
plt.legend()
plt.savefig('Analysis_results/generated_accuracy/tbyt_corrs.jpg', dpi=300)"""

# Channel-by-channel correlation

"""corrs = np.zeros([17, 72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake = np.load('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                print(real.shape)
                for k in range(200):
                    for ch in range(17):
                        corrs[ch, index] += spearmanr(fake[ch, k, :], real[ch, k, :])[0]/200
                index += 1
np.save('Analysis_results/generated_accuracy/chbych_corrs.npy', corrs)"""
corrs = np.load('Analysis_results/generated_accuracy/chbych_corrs.npy')
corrs_avg = np.average(corrs, axis=1)
corrs_err = np.zeros([17])

"""corrs_linear = np.zeros([17, 72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake_linear = np.load('generated_linearmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                print(real.shape)
                for k in range(200):
                    for ch in range(17):
                        corrs_linear[ch, index] += spearmanr(fake_linear[ch, k, :], real[ch, k, :])[0]/200
                index += 1
np.save('Analysis_results/generated_accuracy/chbych_corrs_linear.npy', corrs_linear)"""
corrs_linear = np.load('Analysis_results/generated_accuracy/chbych_corrs_linear.npy')
corrs_linear_avg = np.average(corrs_linear, axis=1)
corrs_linear_err = np.zeros([17])

for ch in range(17):
    corrs_err[ch] = np.std(corrs[ch], ddof=1)/np.sqrt(72)
    corrs_linear_err[ch] = np.std(corrs_linear[ch], ddof=1)/np.sqrt(72)
ch = np.arange(0, 17, 1)
plt.errorbar(ch, corrs_avg, yerr=corrs_err, color='yellowgreen', ecolor='olive', marker='o', markersize=0.1, capsize=2, label='EEG2EEG model')
plt.errorbar(ch, corrs_linear_avg, yerr=corrs_linear_err, color='lightgreen', ecolor='limegreen', marker='o', markersize=0.1, capsize=2, label='Linear model')
channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5',
            'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
plt.xticks(ch, channels, fontsize=10, rotation=60, ha='right')
plt.ylabel('Generated Accuracy', fontsize=18)
plt.legend()
plt.savefig('Analysis_results/generated_accuracy/chbych_corrs.jpg', dpi=300)