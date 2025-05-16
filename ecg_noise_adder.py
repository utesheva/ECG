# Добавление шума с заданным SNR
# с использованием исходных текстов П. Слолвьёва и А. Карлстедта

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import mne
import argparse
import os
import pdb
from scipy.signal import resample
import IRM
from scipy.ndimage import uniform_filter1d

def load_edf_signal(file_path, channel_name=None):
    """
    Загрузка сигнала из EDF-файла.
    file_path: путь к EDF файлу
    channel_name: имя канала (если None — берётся первый)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EDF file '{file_path}' not found.")

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    channels = raw.ch_names
    print(channels)
    fs = int(raw.info['sfreq'])

    if channel_name is None:
        channel_name = raw.ch_names[0]

    if channel_name not in raw.ch_names:
        raise ValueError(
            f"Канал '{channel_name}' не найден. Доступные каналы: {raw.ch_names}")

    data, _ = raw.get_data(picks=channel_name, return_times=True)
    ecg = data[0]
    return ecg, fs

def compute_snr(signal, noise):
    """Computing SNR"""
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    return 10 * np.log10(power_signal / power_noise)

def add_noise(ecg, fs, noise, noise_fs, snr):
    if fs != noise_fs:
        noise = resample(noise, fs)

    if len(ecg) < len(noise):
        noise=noise[:len(ecg)]
    else:
        noise1 = np.array([])
        while len(noise1) < len(ecg):
            noise1 = np.concatenate((noise1, noise))
        noise = noise1[:len(ecg)]

    power_signal = np.mean(ecg ** 2)
    power_noise = np.mean(noise ** 2)
    scale=np.sqrt(power_signal/(power_noise*10**(snr/10)))

    noisy_ecg = ecg + scale*noise
    return noisy_ecg


def main(directory="clean", channel=None, noise_file='emg1.edf'):
    """Get mse and snr of the set"""
    edf_files = [f for f in os.listdir(directory) if f.endswith('.edf')]
    results = []
    noise, noise_fs = load_edf_signal(noise_file)
    for edf_file in edf_files:
        print(f"Файл: {edf_file}")
        ecg, fs = load_edf_signal(os.path.join(directory, edf_file), channel)
        ecg = ecg[:fs*60]
        print('len of signal:', len(ecg))
        for snr in range(1, 31):
            noisy_ecg = add_noise(ecg, fs, noise, noise_fs, snr)
            filtered_ecg, it = IRM.main(noisy_ecg, fs)
            mse = np.mean((filtered_ecg - ecg) ** 2)
            results.append((snr, mse, edf_file))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal', type=str, help='Путь к EDF-файлу сигнала')
    parser.add_argument('--noise', type=str,
                        help='Путь к EDF-файлу шума')
    parser.add_argument('-c', '--sig_channel', type=str,
                        help='Имя канала EDF для полезного сигнала (по умолчанию первый)')
    parser.add_argument('--snr' , type=int,
                        help='Желаемое отношение сигнал/шум в децибелах')
    args = parser.parse_args()

    
    #mne.export.export_raw(fname="noised.edf", raw=mne.io.RawArray(data=data, info=mne.create_info(ch_names=1, sfreq=fs, ch_types='ecg')),fmt='edf', overwrite=True)
    results = main()
    results.sort(key=lambda x: x[0])
    snrs = [r[0] for r in results]
    mses = [r[1] for r in results]
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, mses, marker='o', markersize=5)
    plt.xlabel('SNR (dB)')
    plt.ylabel('MSE')
    plt.title('Зависимость MSE от SNR при удалении EMG шума')
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig("mse_vs_snr_1.png")
    plt.show()
    '''
    mse_smooth = uniform_filter1d(mses, size=5)

    plt.figure(figsize=(10, 6))
    plt.plot(snrs, mse_smooth, label="Сглаженный MSE", color="navy")
    plt.scatter(snrs, mses, s=20, color="skyblue", label="Исходные данные")

    plt.grid(True)
    plt.title("Зависимость MSE от SNR при удалении EMG шума")
    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig('snr.png')
    #mne.export.export_raw(fname="noised.edf", raw=mne.io.RawArray(data=data, info=mne.create_info(ch_names=1, sfreq=fs, ch_types='ecg')),fmt='edf', overwrite=True)
