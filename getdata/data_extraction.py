import numpy as np

"""subs = ['01', '02', '03', '04', '05',
        '06', '07', '08', '10']
channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5',
            'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
for sub in subs:
    data = np.zeros([17, 16540, 4, 200])
    baseline = np.zeros([17, 16540, 4, 50])
    conditioni = np.zeros([16540], dtype=int)
    for s in range(4):
        rawdata = np.load('../sub-' + sub + '/ses-' + str(s+1).zfill(2) + '/raw_eeg_training.npy', allow_pickle=True)
        stim = rawdata.item()['raw_eeg_data'][63]
        print(max(stim))
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
                baseline[:, j, conditioni[j]] = selecteddata[:, condition_indexes[0]-50:condition_indexes[0]]
                baseline[:, j, conditioni[j]+1] = selecteddata[:, condition_indexes[1]-50:condition_indexes[1]]
                conditioni[j] = conditioni[j] + len(condition_indexes)
    data = np.average(data, axis=2)
    baseline = np.average(baseline, axis=(2, 3))
    for i in range(17):
        for j in range(16540):
            data[i, j] = data[i, j] - baseline[i, j]
    np.save('eeg_data/train/sub' + sub + '.npy', data)"""

subs = ['01', '02', '03', '04', '05',
        '06', '07', '08', '10']
channels = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P7', 'P5',
            'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
for sub in subs:
    data = np.zeros([17, 200, 80, 200])
    baseline = np.zeros([17, 200, 80, 50])
    conditioni = np.zeros([200], dtype=int)
    for s in range(4):
        rawdata = np.load('../sub-' + sub + '/ses-' + str(s+1).zfill(2) + '/raw_eeg_test.npy', allow_pickle=True)
        stim = rawdata.item()['raw_eeg_data'][63]
        print(max(stim))
        index = 0
        for i in range(len(stim)):
            if stim[i] == 199:
                index += 1
        print(index)
        ch_names = rawdata.item()['ch_names']
        ch_indexes = np.zeros([17], dtype=int)
        i = 0
        for ch in channels:
            ch_indexes[i] = int(np.where(np.array(ch_names) == ch)[0])
            i = i + 1
        selecteddata = rawdata.item()['raw_eeg_data'][ch_indexes]
        for j in range(200):
            condition_indexes = np.where(stim == j+1)[0]
            for k in range(20):
                data[:, j, conditioni[j]+k] = selecteddata[:, condition_indexes[k]:condition_indexes[k]+200]
                baseline[:, j, conditioni[j]+k] = selecteddata[:, condition_indexes[k]-50:condition_indexes[k]]
            conditioni[j] = conditioni[j] + 20
    baseline = np.average(baseline, axis=3)
    for i in range(17):
        for j in range(200):
            for k in range(80):
                data[i, j, k] = data[i, j, k] - baseline[i, j, k]
    np.save('eeg_data/test/st_sub' + sub + '.npy', data)