from matplotlib import pyplot as plt
import mne
import scipy.signal as signal
import numpy as np
from QRS import Pan_Tompkins_QRS, heart_rate
import sys
import time


def preprocess(syg, FS):
    butter = signal.butter(2, 100, 'lowpass', fs=FS, output='sos')
    syg_butter = signal.sosfiltfilt(butter, syg)
    iir_b, iir_a = signal.iirnotch(50, Q=50, fs=FS)
    syg_iir = signal.filtfilt(iir_b, iir_a, syg_butter)
    butter2 = signal.butter(5, 2, 'highpass', fs=FS, output='sos') 
    syg_butter2 = signal.sosfiltfilt(butter2, syg_iir)
    return syg_butter2


def find_similar_beats(sig, peaks, cur, xcorr_thr, nHB_thr, HB_start):
    beat_len = len(cur)
    similar = []
    if cur.size == 0:
        return similar
    for i, hb in enumerate(HB_start):
        start = hb
        end = min(start + beat_len, len(sig))
    
        beat = sig[start:end]
        xcorr = np.correlate(cur, beat, 'full')
        if xcorr[beat_len - 1] >= xcorr_thr* np.max(np.correlate(cur, cur)):
            similar.append(beat)
        if len(similar) >= nHB_thr:
            break

    return similar


def generate_AS(sig, peaks, FS):
    xcorr_thr = 0.97
    nHB_thr = 7
    aux_signal = np.zeros_like(sig)
    HB_start = [max(0, peak - int(0.25 * np.median(np.diff(peaks)))) for peak in peaks][1:]
    for i, hb in enumerate(HB_start):
        nHB = 0
        start = hb
        end = len(sig) if i + 1 == len(HB_start) else HB_start[i+1]
        cur = sig[start:end]
        similar_beats = find_similar_beats(sig, peaks, cur, xcorr_thr, nHB_thr, HB_start)
        nHB = len(similar_beats)
        
        while nHB < nHB_thr and nHB != 1:
            if xcorr_thr >= 0.91:
                xcorr_thr -= 0.02
            else:
                nHB_thr -= 3
                xcorr_thr = 0.97
            similar_beats = find_similar_beats(sig, peaks, cur, xcorr_thr, nHB_thr, HB_start)
            nHB = len(similar_beats)

        if nHB >= 11:
            MA = 5
        else:
            MA = 15

        if nHB == 0:
            continue

        aux_hb = np.mean(similar_beats, axis=0)
        r = int(0.25 * np.median(np.diff(peaks)))
        segment1 = int(max(0, r - int(0.04*FS)))
        segment2 = int(min(len(aux_hb), r + int(0.04*FS)))
        #print("Segments:", segment1, " ", segment2, "\n")
        aux_MA = min(MA, len(aux_hb))
        aux_hb[:segment1] =np.convolve(aux_hb[:segment1], np.ones(aux_MA)/aux_MA, mode='same')
        # Избегаем Value Error, если остаток блока меньше окна усреднения
        # Также не проводим свертку пустого блока
        if segment2 < aux_hb.size:
            aux_aux_hb = np.convolve(aux_hb[segment2:], np.ones(MA)/MA, mode='same')
            lng = aux_hb[segment2:].size
            if lng >= MA:
                aux_hb[segment2:] = aux_aux_hb
            else:
                aux_hb[segment2:] = aux_aux_hb[:lng]
        aux_signal[start:end] = aux_hb[:end-start]
    return aux_signal


def irm(sig, peaks, FS, cnt=0):
    auxilary_signal = generate_AS(sig, peaks, FS)

    noise = sig - auxilary_signal
    butter = signal.butter(2, 10, 'highpass', fs=FS, output='sos')
    noise_butter = signal.sosfiltfilt(butter, noise)
    
    ob = sig - noise_butter
    if cnt == 0:
        signal_power = np.sum(np.square(sig ** 2))
        noise_power = np.sum(np.square(noise_butter ** 2))
        snr = 10 * np.log10(signal_power / noise_power)
        if snr > 16:
            it = 1
        elif snr > 8:
            it = 2
            t2 = time.time()
            ob = irm(ob, peaks, FS, cnt=1)
        else:
            it = 3
            start = time.time()
            ob = irm(ob, peaks, FS, cnt=2)
            ob = irm(ob, peaks, FS, cnt=2)
    return ob


def postprocess(sig_before, sig_irm, FS):
    butter = signal.butter(5, 2, 'lowpass', fs=FS, output='sos')
    low = signal.sosfiltfilt(butter, sig_before)
    res = sig_irm + low
    return res


def main(record, freq):
    preprocessed = np.array(preprocess(record, freq))

    QRS_detector = Pan_Tompkins_QRS(freq)
    QRS_detector.solve(preprocessed)

    hr = heart_rate(preprocessed, freq)

    result = hr.find_r_peaks()
    result1 = []
    for i in range(len(result)-1):
        if result[i] != result[i+1]:
            result1.append(result[i])
    result = np.array(result1)

    result = result[result > 0]

    heartRate = (60*freq)/np.average(np.diff(result[1:]))

    irm_signal = irm(preprocessed, result, freq)
    postprocessed = postprocess(record, irm_signal, freq)
    return postprocessed


if __name__ == '__main__':
    NAME = sys.argv[1]
    t1 = time.time()
    RECORD = mne.io.read_raw_edf(NAME, preload=True)
    t2 = time.time()
    INFO = RECORD.info
    FS = int(INFO['sfreq'])

    channels = RECORD.ch_names

    record_1, times = RECORD.get_data(return_times=True, picks=channels[0])
    record_1 = record_1[:60*FS]
    preprocessed = np.array(preprocess(record_1[0], FS))

    QRS_detector = Pan_Tompkins_QRS(FS)
    QRS_detector.solve(preprocessed)

    hr = heart_rate(preprocessed, FS)

    result = hr.find_r_peaks()
    result1 = []
    for i in range(len(result)-1):
        if result[i] != result[i+1]:
            result1.append(result[i])
    result = np.array(result1)

    result = result[result > 0]

    heartRate = (60*FS)/np.average(np.diff(result[1:]))
    print("Heart Rate", heartRate, "BPM")

    irm_signal = irm(preprocessed, result, FS)
    postprocessed = postprocess(record_1[0], irm_signal, FS)

    plt.xticks(np.arange(0, len(preprocessed)+1, 150))
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.plot(record_1[0], label='Исходный сигнал', color='red')
    plt.plot(preprocessed, label='Первый этап', color='#a7a6aa')
    plt.scatter(result, preprocessed[result], color='green', s=50, marker='*', label='R-пики')

    plt.plot(irm_signal, label='Второй этап', color='#57565c')
    plt.plot(postprocessed, label='Результат', color='blue')
    plt.legend()

    plt.show()
    '''
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.plot(record_1[0][:5000], label='Исходный сигнал', color='red')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.plot(record_1[0][:5000], label='Исходный сигнал', color='red')
    plt.legend()

    plt.savefig('IRM.py')
    '''

