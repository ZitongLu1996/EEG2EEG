import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

"""for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_fullmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (2, 0, 1, 3))
                fake = fake*std2 + mean2
                np.save('generated_fullmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)
                print(i+1, j+1)"""

"""for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_linearmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (2, 0, 1, 3))
                fake = fake*std2 + mean2
                np.save('generated_linearmodel/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)
                print(i+1, j+1)"""

"""for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_nocosineloss/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (2, 0, 1, 3))
                fake = fake*std2 + mean2
                np.save('generated_nocosineloss/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)
                print(i+1, j+1)"""

for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_nocombination/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (2, 0, 1, 3))
                fake = fake*std2 + mean2
                np.save('generated_nocombination/st_Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)
                print(i+1, j+1)