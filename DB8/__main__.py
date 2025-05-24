import sys
import mne
from . import main
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    NAME = sys.argv[1]
    RECORD = mne.io.read_raw_edf(NAME, preload=True)
    INFO = RECORD.info
    FS = int(INFO['sfreq'])

    channels = RECORD.ch_names

    record_1, times = RECORD.get_data(return_times=True, picks=channels[0])

    result = main(record_1[0], FS)

    plt.xticks(np.arange(0, len(result)+1, 150))
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.plot(record_1[0], label='Исходный сигнал', color='red')
    plt.plot(result, label='Результат', color='blue')
    plt.legend()

    plt.show()
