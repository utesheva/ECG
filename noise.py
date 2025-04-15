import pywt
from matplotlib import pyplot as plt
import mne
import scipy.signal as signal
import numpy as np
import pdb
from QRS import Pan_Tompkins_QRS, heart_rate

NAME = 'Noise.edf'
FS = 250

record = mne.io.read_raw_edf(NAME, preload=True)
info = record.info
channels = record.ch_names
print(info)

record_1, times=record.get_data(return_times=True)

if __name__ == '__main__':
    plt.plot(record_1[0][:5000], label = 'Noise', color='grey')

    plt.show()

