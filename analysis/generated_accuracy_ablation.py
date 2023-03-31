import numpy as np
from scipy.stats import spearmanr, ttest_rel
import matplotlib.pyplot as plt

# overall, pattern, profile corrs of the full model

x = np.array([0, 1, 2])
w = 0.15

fig, ax = plt.subplots()

y_linear = np.load('Analysis_results/avgtrials_comparisons/3corrs_linear.npy')
y_linear_avg = np.average(y_linear, axis=1)
y_linear_err = np.zeros([3])
for i in range(3):
    y_linear_err[i] = np.std(y_linear[i], ddof=1)/np.sqrt(72)
ax.bar(x-1.5*w, y_linear_avg, color='white', edgecolor='lightblue', yerr=y_linear_err, width=w, hatch='///', capsize=6, ecolor='lightblue', zorder=2, label='Linear model')

"""overall_corrs_nocombination = np.zeros([72])
pattern_corrs_nocombination = np.zeros([72])
profile_corrs_nocombination = np.zeros([72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake_nocombination = np.load('generated_nocombination/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                overall_corrs_nocombination[index] = spearmanr(fake_nocombination.flatten(), real.flatten())[0]
                for k in range(200):
                    pattern_corrs_nocombination[index] += spearmanr(fake_nocombination[:, k].flatten(), real[:, k].flatten())[0]/200
                    profile_corrs_nocombination[index] += spearmanr(fake_nocombination[:, :, k].flatten(), real[:, :, k].flatten())[0]/200
                index += 1

y_nocombination = np.zeros([3, 72])
y_nocombination[0] = overall_corrs_nocombination
y_nocombination[1] = pattern_corrs_nocombination
y_nocombination[2] = profile_corrs_nocombination
np.save('Analysis_results/avgtrials_comparisons/3corrs_nocombination.npy', y_nocombination)"""
y_nocombination = np.load('Analysis_results/avgtrials_comparisons/3corrs_nocombination.npy')
y_nocombination_avg = np.average(y_nocombination, axis=1)
y_nocombination_err = np.zeros([3])
for i in range(3):
    y_nocombination_err[i] = np.std(y_nocombination[i], ddof=1)/np.sqrt(72)
ax.bar(x-0.5*w, y_nocombination_avg, color='white', edgecolor='deepskyblue', yerr=y_nocombination_err, width=w, hatch='///', capsize=6, ecolor='deepskyblue', zorder=2, label='Non-Combination')


"""overall_corrs_nocosineloss = np.zeros([72])
pattern_corrs_nocosineloss = np.zeros([72])
profile_corrs_nocosineloss = np.zeros([72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                fake_nocosineloss = np.load('generated_nocosineloss/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                overall_corrs_nocosineloss[index] = spearmanr(fake_nocosineloss.flatten(), real.flatten())[0]
                for k in range(200):
                    pattern_corrs_nocosineloss[index] += spearmanr(fake_nocosineloss[:, k].flatten(), real[:, k].flatten())[0]/200
                    profile_corrs_nocosineloss[index] += spearmanr(fake_nocosineloss[:, :, k].flatten(), real[:, :, k].flatten())[0]/200
                index += 1

y_nocosineloss = np.zeros([3, 72])
y_nocosineloss[0] = overall_corrs_nocosineloss
y_nocosineloss[1] = pattern_corrs_nocosineloss
y_nocosineloss[2] = profile_corrs_nocosineloss
np.save('Analysis_results/avgtrials_comparisons/3corrs_nocosineloss.npy', y_nocosineloss)"""
y_nocosineloss = np.load('Analysis_results/avgtrials_comparisons/3corrs_nocosineloss.npy')
y_nocosineloss_avg = np.average(y_nocosineloss, axis=1)
y_nocosineloss_err = np.zeros([3])
for i in range(3):
    y_nocosineloss_err[i] = np.std(y_nocosineloss[i], ddof=1)/np.sqrt(72)
ax.bar(x+0.5*w, y_nocosineloss_avg, color='white', edgecolor='dodgerblue', yerr=y_nocosineloss_err, width=w, hatch='///', capsize=6, ecolor='dodgerblue', zorder=2, label='Non-CosineLoss')

y = np.load('Analysis_results/avgtrials_comparisons/3corrs.npy')
y_avg = np.average(y, axis=1)
y_err = np.zeros([3])
for i in range(3):
    y_err[i] = np.std(y[i], ddof=1)/np.sqrt(72)
ax.bar(x+1.5*w, y_avg, color='white', edgecolor='navy', yerr=y_err, width=w, hatch='///', capsize=6, ecolor='navy', zorder=2, label='EEG2EEG model')

print(ttest_rel(y[0], y_nocombination[0]))
print(ttest_rel(y[1], y_nocombination[1]))
print(ttest_rel(y[2], y_nocombination[2]))

ax = plt.gca()
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(x, ['Overall', 'Pattern', 'Profile'], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Generated Accuracy', fontsize=18)
plt.ylim(0.4, 0.8)
"""plt.ylim(0.1, 0.9)
plt.plot([-0.15, 0.15], [0.85, 0.85], color='black')
plt.text(-0.08, 0.84, '***', fontsize=16)
plt.plot([0.85, 1.15], [0.86, 0.86], color='black')
plt.text(0.92, 0.85, '***', fontsize=16)
plt.plot([1.85, 2.15], [0.7, 0.7], color='black')
plt.text(1.92, 0.69, '***', fontsize=16)"""
#plt.savefig('Analysis_results/generated_accuracy/corrs_bar.jpg', dpi=300)
plt.show()

print(np.average(y_linear[0]), np.std(y_linear[0], ddof=1))
print(np.average(y_nocombination[0]), np.std(y_nocombination[0], ddof=1))
print(np.average(y_nocosineloss[0]), np.std(y_nocosineloss[0], ddof=1))
print(np.average(y[0]), np.std(y[0], ddof=1))
