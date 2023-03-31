import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

for i in range(10):
    for j in range(10):
        if i > j and i != 8 and j != 8:
            data1 = np.load('eeg_data/test/sub' + str(i+1).zfill(2) + '.npy')
            data2 = np.load('eeg_data/test/sub' + str(j+1).zfill(2) + '.npy')
            data1 = np.transpose(data1, (1, 0, 2))
            data2 = np.transpose(data2, (1, 0, 2))
            data1 = np.reshape(data1, [200, 17*200])
            data2 = np.reshape(data2, [200, 17*200])
            corr = 0
            for k in range(200):
                r = spearmanr(data1[k], data2[k])[0]
                if np.isnan(r):
                    r = 0
                corr += r
            corr = corr/200

            print('Sub' + str(i+1).zfill(2) + ' vs. Sub' + str(j+1).zfill(2) + ': ', corr)