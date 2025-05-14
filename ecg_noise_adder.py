# Добавление шума с заданным SNR
# с использованием исходных текстов П. Слолвьёва и А. Карлстедта

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import mne
import argparse
import os
import pdb

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

    if args.signal:
        ecg, fs = load_edf_signal(args.signal)
    if args.noise:
        noise, fs_noise = load_edf_signal(args.noise)

    if fs != fs_noise:
        raise ValueError ("current version of this program requires the same sampling rate for signal and noise")
    #Выравнивание длины шума по сигналу
    if len(ecg) < len(noise):
        noise=noise[:len(ecg)]
    else:
        noise1 = np.array([])
        while len(noise1) < len(ecg):
            noise1 = np.concatenate((noise1, noise))
        noise = noise1[:len(ecg)]

    power_signal = np.mean(ecg ** 2)
    power_noise = np.mean(noise ** 2)
    scale=np.sqrt(power_signal/(power_noise*10**(args.snr/10)))

    noisy_ecg = ecg + scale*noise
    data=np.array([noisy_ecg])
    pdb.set_trace()
#    data[0]=noisy_ecg
# Запись выходного файла
    mne.export.export_raw(fname="noised.edf", raw=mne.io.RawArray(data=data, info=mne.create_info(ch_names=1, sfreq=fs, ch_types='ecg')),
                      fmt='edf', overwrite=True)

    t = np.arange(len(ecg)) / fs
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_ecg, label='сигнал+шум')
    plt.title('Исходный зашумлённый сигнал')
    plt.legend()

    plt.plot(t, ecg, '--', alpha=0.7, label='Исходный ЭКГ')
    plt.title('Результат фильтрации')
    plt.legend()
    plt.show()
