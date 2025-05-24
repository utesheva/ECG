# Добавление шума с заданным SNR
# с использованием исходных текстов П. Слолвьёва и А. Карлстедта

import numpy as np
import matplotlib.pyplot as plt
import mne
import argparse
import os
from scipy.signal import resample
from scipy.ndimage import uniform_filter1d

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import IRM

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


def main(directory='tests', channel=None, noise_file='emg1.edf'):
    """Get mse and snr of the set"""
    edf_files = [f for f in os.listdir(directory) if f.endswith('.edf')]
    results = []
    noise, noise_fs = load_edf_signal(noise_file)
    
    for edf_file in edf_files:
        print(f"Файл: {edf_file}")
        ecg, fs = load_edf_signal(os.path.join(directory, edf_file), channel)
        ecg = ecg[:fs*60]
        
        for snr in range(1, 31):
            noisy_ecg = add_noise(ecg, fs, noise, noise_fs, snr)
            filtered_ecg = IRM.main(noisy_ecg, fs)
            mse = np.mean((filtered_ecg - ecg) ** 2)
            results.append((snr, mse, edf_file))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--direсtory', type=str, help='Путь к директории с EDF-файлами')
    parser.add_argument('-n', '--noise', type=str,
                        help='Путь к EDF-файлу шума')
    args = parser.parse_args()
    
    results = main(directory=args.direсtory, noise_file=args.noise)
    results.sort(key=lambda x: x[0])
    snrs = [r[0] for r in results]
    mses = [r[1] for r in results]
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
