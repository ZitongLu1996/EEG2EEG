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
                fake = np.load('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (1, 0, 2))
                fake = fake*std2 + mean2
                np.save('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)"""

"""for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_linearmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (1, 0, 2))
                fake = fake*std2 + mean2
                np.save('generated_linearmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)"""

"""for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_nocosineloss/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (1, 0, 2))
                fake = fake*std2 + mean2
                np.save('generated_nocosineloss/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)"""

for i in range(10):
    if i != 8:
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_nocombination/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                fake = np.transpose(fake, (1, 0, 2))
                fake = fake*std2 + mean2
                np.save('generated_nocombination/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '_final.npy', fake)