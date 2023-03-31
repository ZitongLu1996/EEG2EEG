import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

t = np.arange(0, 200, 1)
data2 = np.load('eeg_data/train/sub02.npy')
mean2 = np.average(data2)
std2 = np.std(data2)
#data2 = (data2 - mean2)/std2

fake = np.load('generated/Sub3ToSub2_best.npy')
fake = fake * std2 + mean2
real = np.load('eeg_data/test/sub02.npy')

data2 = np.average(data2, axis=(0, 1))
fake = np.average(fake, axis=(0, 1))
real = np.average(real, axis=(0, 1))
r = spearmanr(fake, real)[0]

#plt.plot(t, data2, label='train')
plt.plot(t, fake, label='fake')
plt.plot(t, real, label='real')

plt.legend()
plt.show()