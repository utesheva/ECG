import pywt
from matplotlib import pyplot as plt
import mne
import scipy.signal as signal
import numpy as np
import pdb
from QRS import Pan_Tompkins_QRS, heart_rate
from scipy.stats import pearsonr

NAME = 'Rh10010.edf'
FS = 250

record = mne.io.read_raw_edf(NAME, preload=True)
info = record.info
channels = record.ch_names
print(info)

record_1, times=record.get_data(return_times=True, picks='sig')


def preprocess(syg):
    global FS
    butter = signal.butter(2, 100, 'lowpass', fs=FS, output='sos')
    syg_butter = signal.sosfiltfilt(butter, syg)
    iir_b, iir_a = signal.iirnotch(50, Q=50, fs=FS)
    syg_iir = signal.filtfilt(iir_b, iir_a, syg_butter)
    butter2 = signal.butter(2, 2, 'highpass', fs=FS, output='sos')    
    syg_butter2 = signal.sosfiltfilt(butter, syg_iir)
    return syg_butter2

def find_similar_beats(sig, peaks, cur, xcorr_thr, nHB_thr):
    beat_len = len(cur)
    similar = [] 
    for peak in peaks:
        start = max(0, peak - beat_len // 2)
        end = min(len(sig), start + beat_len)
        beat = sig[start:end]
        if len(beat) == len(cur):
            corr = pearsonr(cur, beat)[0]
            print(corr)
            if corr > xcorr_thr:
                similar.append(beat)
                if len(similar) >= nHB_thr:
                    break

    return np.array(similar) if similar else np.array([cur])

def generate_AS(sig, peaks):
    xcorr_thr = 0.97
    nHB_thr = 7

    aux_signal = np.zeros_like(sig)
    nHB = 0
    for i, peak in enumerate(peaks):
        while nHB < nHB_thr and nHB != 1:
            start = max(0, peak - int(0.25 * np.median(np.diff(peaks))))
            if i + 1 < len(peaks):
                end = peaks[i+1]
            else:
                end = len(sig)
            cur = sig[start:end]
            similar_beats = find_similar_beats(sig, peaks, cur, xcorr_thr, nHB_thr)
            nHB = len(similar_beats)
            if xcorr_thr >= 0.91:
                xcorr_thr -= 0.02
            else:
                nHB_thr -= 3
                xcorr_thr = 0.97
        if nHB >= 11:
            MA = 5
        else:
            MA = 16 - nHB
        aux_hb = np.mean(similar_beats, axis=0)
        aux_hb = np.convolve(aux_hb, np.ones(MA)/MA, mode='same')
        aux_signal[start:end] = aux_hb
    return aux_signal

def irm(sig, peaks, cnt=0):
    global FS
    print('Iteration', cnt)
    auxilary_signal = generate_AS(sig, peaks)

    noise = sig - auxilary_signal
    butter = signal.butter(2, 10, 'highpass', fs=FS, output='sos')
    noise_butter = signal.sosfiltfilt(butter, noise)
    
    ob = sig - noise_butter
    
    if cnt == 0:
        signal_power = np.sum(np.square(sig ** 2))
        noise_power = np.sum(np.square(noise_butter ** 2))
        snr = 10 * np.log10(signal_power / noise_power)
        print('SNR', snr)
        if snr > 16: 
            pass
        elif snr > 8:
            ob = irm(ob, peaks, cnt=1)
        else:
            ob = irm(ob, peaks, cnt=2)
    elif cnt == 2:
        return irm(ob, peaks, cnt=1)

    return ob


def postprocess(sig):
    butter = signal.butter(2, 2, 'lowpass', fs=FS, output='sos')
    low = signal.sosfiltfilt(butter, sig)
    return sig + low

if __name__ == '__main__':
    preprocessed = np.array(preprocess(record_1[0][:5000]))

    QRS_detector = Pan_Tompkins_QRS(FS)
    QRS_detector.solve(preprocessed)

    hr = heart_rate(preprocessed, FS)
    result = hr.find_r_peaks()
    result = np.array(result)

    result = result[result > 0]

    heartRate = (60*FS)/np.average(np.diff(result[1:]))
    print("Heart Rate", heartRate, "BPM")

    irm_signal = irm(preprocessed, result)
    postprocessed = postprocess(irm_signal)
    plt.xticks(np.arange(0, len(preprocessed)+1, 150))
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.plot(record_1[0][:5000], label = 'Before', color='grey')
    plt.plot(preprocessed, label = 'Preprocessed', color='blue')
    plt.scatter(result, preprocessed[result], color='red', s=50, marker='*')
    plt.plot(irm_signal, label = 'IRM stage', color='green')
    plt.plot(postprocessed)
    plt.legend()
    plt.show()

