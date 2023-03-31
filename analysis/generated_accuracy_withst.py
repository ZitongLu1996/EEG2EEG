import numpy as np
from scipy.stats import spearmanr, ttest_rel
import matplotlib.pyplot as plt

# overall, pattern, profile corrs

x = np.array([0, 1, 2])
w = 0.3

y = np.load('Analysis_results/avgtrials_comparisons/3corrs.npy')
y_avg = np.average(y, axis=1)
y_err = np.zeros([3])
for i in range(3):
    y_err[i] = np.std(y[i], ddof=1)/np.sqrt(72)
fig, ax = plt.subplots()
for i in range(3):
    if i == 0:
        ax.scatter(x[i]+w/2 + np.random.random(y[i].size)*w/8-w/16, y[i], s=15, color='yellowgreen',
                   edgecolors='olive', alpha=0.5, zorder=1, label='ERP level')
    else:
        ax.scatter(x[i] + w / 2 + np.random.random(y[i].size) * w / 8 - w / 16, y[i], s=15, color='yellowgreen',
                   edgecolors='olive', alpha=0.5, zorder=1)
ax.bar(x+w/2, y_avg, color='darkolivegreen', edgecolor='darkolivegreen', yerr=y_err, width=w, capsize=6, ecolor='darkolivegreen', visible=False, zorder=2)

y_st = np.load('Analysis_results/avgtrials_comparisons_st/3corrs.npy')
y_st_avg = np.average(y_st, axis=1)
y_st_err = np.zeros([3])
for i in range(3):
    y_st_err[i] = np.std(y_st[i], ddof=1)/np.sqrt(72)
for i in range(3):
    if i == 0:
        ax.scatter(x[i]-w/2 + np.random.random(y_st[i].size)*w/8-w/16, y_st[i], s=15, color='lightgreen',
                   edgecolors='limegreen', alpha=0.5, zorder=1, label='Single trial level')
    else:
        ax.scatter(x[i] - w / 2 + np.random.random(y_st[i].size) * w / 8 - w / 16, y_st[i], s=15, color='lightgreen',
                   edgecolors='limegreen', alpha=0.5, zorder=1)
ax.bar(x-w/2, y_st_avg, color='forestgreen', edgecolor='forestgreen', yerr=y_st_err, width=w, capsize=6, ecolor='forestgreen', visible=False, zorder=2)

print(ttest_rel(y[0], y_st[0]))
print(ttest_rel(y[1], y_st[1]))
print(ttest_rel(y[2], y_st[2]))

ax = plt.gca()
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(x, ['Overall', 'Pattern', 'Profile'], fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Generated Accuracy', fontsize=18)
plt.savefig('Analysis_results/generated_accuracy_withst/corrs_bar.jpg', dpi=300)
plt.show()

for i in range(3):
    print(np.average(y[i]), np.std(y[i], ddof=1))
    print(np.average(y_st[i]), np.std(y_st[i], ddof=1))

# Time-by-time correlation

"""corrs = np.load('Analysis_results/generated_accuracy/tbyt_corrs.npy')
corrs_avg = np.average(corrs, axis=1)
corrs_err = np.zeros([200])
for t in range(200):
    corrs_err[t] = np.std(corrs[t], ddof=1)/np.sqrt(72)

corrs_st = np.load('Analysis_results/generated_accuracy_st/tbyt_corrs.npy')
corrs_avg_st = np.average(corrs_st, axis=1)
corrs_err_st = np.zeros([200])
for t in range(200):
    corrs_err_st[t] = np.std(corrs_st[t], ddof=1)/np.sqrt(72)

t = np.arange(0, 200, 1)
plt.errorbar(t, corrs_avg, yerr=corrs_err, color='yellowgreen', ecolor='olive', elinewidth=0.2, capthick=0.4, marker='o', markersize=0.1, capsize=2, label='ERP level')
plt.errorbar(t, corrs_avg_st, yerr=corrs_err_st, color='lightgreen', ecolor='limegreen', elinewidth=0.2, capthick=0.4, marker='o', markersize=0.1, capsize=2, label='Single trial level')
plt.ylabel('Generated Accuracy', fontsize=18)
plt.xlabel('Time (ms)', fontsize=18)
plt.legend()
plt.savefig('Analysis_results/generated_accuracy_withst/tbyt_corrs.jpg', dpi=300)"""

# Channel-by-channel correlation

"""corrs = np.load('Analysis_results/generated_accuracy/chbych_corrs.npy')
corrs_avg = np.average(corrs, axis=1)
corrs_err = np.zeros([17])

corrs_st = np.load('Analysis_results/generated_accuracy_st/chbych_corrs.npy')
corrs_st_avg = np.average(corrs_st, axis=1)
corrs_st_err = np.zeros([17])

for ch in range(17):
    corrs_err[ch] = np.std(corrs[ch], ddof=1)/np.sqrt(72)
    corrs_st_err[ch] = np.std(corrs_st[ch], ddof=1)/np.sqrt(72)
ch = np.arange(0, 17, 1)
plt.errorbar(ch, corrs_avg, yerr=corrs_err, color='yellowgreen', ecolor='olive', marker='o', markersize=0.1, capsize=2, label='ERP level')
plt.errorbar(ch, corrs_st_avg, yerr=corrs_st_err, color='lightgreen', ecolor='limegreen', marker='o', markersize=0.1, capsize=2, label='Single trial level')
channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5',
            'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
plt.xticks(ch, channels, fontsize=10, rotation=60, ha='right')
plt.ylabel('Generated Accuracy', fontsize=18)
plt.legend()
plt.savefig('Analysis_results/generated_accuracy_withst/chbych_corrs.jpg', dpi=300)"""