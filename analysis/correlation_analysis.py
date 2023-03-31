import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import seaborn as sb

# SubA -> SubB vs. SubB -> SubA

AB = np.zeros([3, 36])
BA = np.zeros([3, 36])
corrs = np.load('Analysis_results/avgtrials_comparisons/3corrs.npy')

index = 0
indexAB = 0
indexBA = 0
for i in range(9):
    for j in range(9):
        if i < j:
            AB[:, indexAB] = corrs[:, index]
            indexAB += 1
            index += 1
        if j > i:
            BA[:, indexBA] = corrs[:, index]
            indexBA += 1
            index += 1
    
print(pearsonr(AB[0], BA[0]))
print(pearsonr(AB[1], BA[1]))
print(pearsonr(AB[2], BA[2]))

cons = ['overall', 'pattern', 'profile']

for i in range(3):
    data = {'Generated Accuracy (from Sub A to Sub B)': AB[i],
            'Generated Accuracy (from Sub B to Sub A)': BA[i],}
    df = pd.DataFrame(data, columns=['Generated Accuracy (from Sub A to Sub B)', 'Generated Accuracy (from Sub B to Sub A)'])
    sb.regplot(data = data, x = 'Generated Accuracy (from Sub A to Sub B)', y = 'Generated Accuracy (from Sub B to Sub A)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Generated Accuracy (from Sub A to Sub B)', fontsize=16)
    plt.ylabel('Generated Accuracy (from Sub B to Sub A)', fontsize=16)
    plt.title(cons[i], fontsize=20)
    plt.savefig('Analysis_results/correlation_analysis/corrs_ABvsBA_' + cons[i] + '.jpg', dpi=300)
    plt.show()

# variation vs. generated accuracy

vars = np.zeros([72])
vars_9subs = np.load('Analysis_results/data_variability/data_variability.npy')

index = 0
for i in range(9):
    for j in range(9):
        if i != j:
            vars[index] = vars_9subs[j]
            index += 1

corrs = np.load('Analysis_results/avgtrials_comparisons/3corrs.npy')

print(pearsonr(corrs[0], vars))
print(pearsonr(corrs[1], vars))
print(pearsonr(corrs[2], vars))

cons = ['overall', 'pattern', 'profile']

for i in range(3):
    data = {'Data Variability of Sub B': vars,
            'Generated Accuracy (from Sub A to Sub B)': corrs[i],}
    df = pd.DataFrame(data, columns=['Data Variability of Sub B', 'Generated Accuracy (from Sub A to Sub B)'])
    sb.regplot(data = data, x = 'Data Variability of Sub B', y = 'Generated Accuracy (from Sub A to Sub B)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('EEG Variability of the target subject', fontsize=16)
    plt.ylabel('Generated Accuracy', fontsize=16)
    plt.title(cons[i], fontsize=20)
    plt.savefig('Analysis_results/correlation_analysis/corrs_VarsBvsAcc_' + cons[i] + '.jpg', dpi=300)
    plt.show()



vars = np.zeros([72])
vars_9subs = np.load('Analysis_results/data_variability/data_variability.npy')

index = 0
for i in range(9):
    for j in range(9):
        if i != j:
            vars[index] = vars_9subs[i]
            index += 1

corrs = np.load('Analysis_results/avgtrials_comparisons/3corrs.npy')

print(pearsonr(corrs[0], vars))
print(pearsonr(corrs[1], vars))
print(pearsonr(corrs[2], vars))

cons = ['overall', 'pattern', 'profile']

for i in range(3):
    data = {'Data Variability of Sub A': vars,
            'Generated Accuracy (from Sub A to Sub B)': corrs[i],}
    df = pd.DataFrame(data, columns=['Data Variability of Sub A', 'Generated Accuracy (from Sub A to Sub B)'])
    sb.regplot(data = data, x = 'Data Variability of Sub A', y = 'Generated Accuracy (from Sub A to Sub B)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('EEG Variability of the source subject', fontsize=16)
    plt.ylabel('Generated Accuracy', fontsize=16)
    plt.title(cons[i], fontsize=20)
    plt.savefig('Analysis_results/correlation_analysis/corrs_VarsAvsAcc_' + cons[i] + '.jpg', dpi=300)
    plt.show()

# correlation between AB vs. generated accuracy

"""overall_corrs = np.zeros([72])
pattern_corrs = np.zeros([72])
profile_corrs = np.zeros([72])
index = 0
for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                print(i, j)
                data2 = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                data1 = np.load('eeg_data/test/sub' + str(i + 1).zfill(2) + '.npy')
                overall_corrs[index] = spearmanr(data1.flatten(), data2.flatten())[0]
                for k in range(200):
                    pattern_corrs[index] += spearmanr(data1[:, k].flatten(), data2[:, k].flatten())[0]/200
                    profile_corrs[index] += spearmanr(data1[:, :, k].flatten(), data2[:, :, k].flatten())[0]/200
                index += 1

corrs_AB = np.zeros([3, 72])
corrs_AB[0] = overall_corrs
corrs_AB[1] = pattern_corrs
corrs_AB[2] = profile_corrs

np.save('Analysis_results/avgtrials_comparisons/3corrs_AB.npy', corrs_AB)"""
corrs_AB = np.load('Analysis_results/avgtrials_comparisons/3corrs_AB.npy')

corrs = np.load('Analysis_results/avgtrials_comparisons/3corrs.npy')

print(pearsonr(corrs[0], corrs_AB[0]))
print(pearsonr(corrs[1], corrs_AB[1]))
print(pearsonr(corrs[2], corrs_AB[2]))

cons = ['overall', 'pattern', 'profile']

for i in range(3):
    data = {'Similarity between Sub A and Sub B': corrs_AB[i],
            'Generated Accuracy (from Sub A to Sub B)': corrs[i],}
    df = pd.DataFrame(data, columns=['Similarity between Sub A and Sub B', 'Generated Accuracy (from Sub A to Sub B)'])
    sb.regplot(data = data, x = 'Similarity between Sub A and Sub B', y = 'Generated Accuracy (from Sub A to Sub B)')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('EEG Similarity between two subjects', fontsize=16)
    plt.ylabel('Generated Accuracy', fontsize=16)
    plt.title(cons[i], fontsize=20)
    plt.savefig('Analysis_results/correlation_analysis/corrs_SimABvsAcc_' + cons[i] + '.jpg', dpi=300)
    plt.show()