import pywt
from matplotlib import pyplot as plt
import mne
import scipy.signal as signal
import numpy as np
import pdb
from ecgdetectors import Detectors
from QRS import Pan_Tompkins_QRS, heart_rate

NAME = 'Rh10010.edf'
FS = 250

record = mne.io.read_raw_edf(NAME, preload=True)
info = record.info
channels = record.ch_names
print(info)

record_1, times=record.get_data(return_times=True, picks='sig')

def preprocess(syg):
    butter = signal.butter(2, 100, 'lowpass', fs=FS, output='sos')
    syg_butter = signal.sosfiltfilt(butter, syg)
    iir_b, iir_a = signal.iirnotch(50, Q=50, fs=FS)
    syg_iir = signal.filtfilt(iir_b, iir_a, syg_butter)
    butter2 = signal.butter(2, 2, 'highpass', fs=FS, output='sos')    
    syg_butter2 = signal.sosfiltfilt(butter, syg_iir)
    return syg_butter2

preprocessed = np.array(preprocess(record_1[0]))

QRS_detector = Pan_Tompkins_QRS(FS)
QRS_detector.solve(preprocessed)

hr = heart_rate(preprocessed, FS)
result = hr.find_r_peaks()
result = np.array(result)

result = result[result > 0]

heartRate = (60*FS)/np.average(np.diff(result[1:]))
print("Heart Rate",heartRate, "BPM")

print(len(times), len(result))
plt.plot(times, record_1[0], label = 'Before', color='green')
plt.plot(times, preprocessed, label = 'Preprocessed', color='blue')
#plt.scatter(result, preprocessed[result], color='red', s=50, marker='*')
plt.legend()
plt.show()

