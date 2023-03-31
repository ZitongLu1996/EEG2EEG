import numpy as np
from sklearn.metrics.pairwise import paired_distances

subs = ['01', '02', '03', '04', '05',
        '06', '07', '08', '10']
channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5',
            'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
variabilities = np.zeros([9])
subi = 0
for sub in subs:
    data = np.zeros([17, 16540, 4, 200])
    conditioni = np.zeros([16540], dtype=int)
    for s in range(4):
        rawdata = np.load('../sub-' + sub + '/ses-' + str(s+1).zfill(2) + '/raw_eeg_training.npy', allow_pickle=True)
        stim = rawdata.item()['raw_eeg_data'][63]
        ch_names = rawdata.item()['ch_names']
        ch_indexes = np.zeros([17], dtype=int)
        i = 0
        for ch in channels:
            ch_indexes[i] = int(np.where(np.array(ch_names) == ch)[0])
            i = i + 1
        selecteddata = rawdata.item()['raw_eeg_data'][ch_indexes]
        for j in range(16540):
            condition_indexes = np.where(stim == j+1)[0]
            if len(condition_indexes) == 2:
                data[:, j, conditioni[j]] = selecteddata[:, condition_indexes[0]:condition_indexes[0]+200]
                data[:, j, conditioni[j]+1] = selecteddata[:, condition_indexes[1]:condition_indexes[1]+200]
                conditioni[j] = conditioni[j] + len(condition_indexes)

    for i in range(16540):
        for j in range(4):
            for k in range(4):
                if j > k:
                    data1 = data[:, i, j].flatten()
                    data2 = data[:, i, k].flatten()
                    data1 = np.reshape(data1, [1, 3400])
                    data2 = np.reshape(data2, [1, 3400])
                    variabilities[subi] += paired_distances(data1, data2)[0]

    print(variabilities[subi]/99240)
    subi += 1

np.save('Analysis_results/data_variability/data_variability.npy', variabilities)
