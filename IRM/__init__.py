import scipy.signal as signal
import numpy as np
from QRS_detection import Pan_Tompkins_QRS, heart_rate


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
        if xcorr[beat_len-1] >= xcorr_thr * np.max(np.correlate(cur, cur)):
            similar.append(beat)
        if len(similar) >= nHB_thr:
            break

    return similar


def generate_AS(sig, peaks, FS):
    xcorr_thr = 0.97
    nHB_thr = 7
    aux_signal = np.zeros_like(sig)
    HB_start = [max(0, peak-int(0.25 * np.median(np.diff(peaks)))) for peak in peaks][1:]
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
        aux_MA = min(MA, len(aux_hb))
        aux_hb[:segment1] =np.convolve(aux_hb[:segment1], np.ones(aux_MA) / aux_MA, mode='same')
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
            pass
        elif snr > 8:
            ob = irm(ob, peaks, FS, cnt=1)
        else:
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
    print('Heart rate:', heartRate)

    irm_signal = irm(preprocessed, result, freq)
    postprocessed = postprocess(record, irm_signal, freq)
    return postprocessed
