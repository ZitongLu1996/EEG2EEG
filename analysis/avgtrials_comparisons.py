import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# full model
for i in range(10):
    if i != 8:
        fig, axs = plt.subplots(4, 2, figsize=(8, 6), constrained_layout=True)
        fig.suptitle('From Sub' + str(i+1), fontsize=16)
        t = np.arange(0, 200, 1)
        index = 0
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_fullmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                fake = np.average(fake, axis=(0, 1))
                fake = fake*std2 + mean2
                real = np.average(real, axis=(0, 1))
                r = spearmanr(fake, real)[0]
                index1 = int(index / 2)
                index2 = index - index1 * 2

                axs[index1, index2].plot(t, real, label='real')
                axs[index1, index2].plot(t, fake, label='generated')
                axs[index1, index2].set_title('To Sub' + str(j+1), fontsize=14)

                index = index + 1
        plt.legend()
        plt.savefig('Analysis_results/avgtrials_comparisons/FromSub' + str(i+1) + '.jpg', dpi=200)
        plt.show()

# linear model
for i in range(10):
    if i != 8:
        fig, axs = plt.subplots(4, 2, figsize=(8, 6), constrained_layout=True)
        fig.suptitle('From Sub' + str(i+1), fontsize=16)
        t = np.arange(0, 200, 1)
        index = 0
        for j in range(10):
            if i != j and j != 8:
                data2 = np.load('eeg_data/train/sub' + str(j + 1).zfill(2) + '.npy')
                mean2 = np.average(data2)
                std2 = np.std(data2)
                fake = np.load('generated_linearmodel/Sub' + str(i + 1) + 'ToSub' + str(j + 1) + '.npy')
                real = np.load('eeg_data/test/sub' + str(j + 1).zfill(2) + '.npy')
                fake = np.average(fake, axis=(0, 1))
                fake = fake*std2 + mean2
                real = np.average(real, axis=(0, 1))
                r = spearmanr(fake, real)[0]
                index1 = int(index / 2)
                index2 = index - index1 * 2

                axs[index1, index2].plot(t, real, label='real')
                axs[index1, index2].plot(t, fake, label='generated')
                axs[index1, index2].set_title('To Sub' + str(j+1), fontsize=14)

                index = index + 1
        plt.legend()
        plt.savefig('Analysis_results/avgtrials_comparisons/FromSub' + str(i+1) + '_linearmodel.jpg', dpi=200)
        plt.show()