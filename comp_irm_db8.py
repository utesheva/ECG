import IRM
import db8
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def load_edf_signal(file_path, channel_name=None):
    """
    Загрузка сигнала из EDF-файла.
    file_path: путь к EDF файлу
    channel_name: имя канала (если None — берётся первый)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EDF file '{file_path}' not found.")

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])

    if channel_name is None:
        channel_name = raw.ch_names[0]

    if channel_name not in raw.ch_names:
        raise ValueError(
            f"Канал '{channel_name}' не найден. Доступные каналы: {raw.ch_names}")

    data, _ = raw.get_data(picks=channel_name, return_times=True)
    ecg = data[0]
    return ecg, fs

def main(file="tests/P10_1_Ag-AgCl.edf", channel=None):
    results = {}
    ecg, fs = load_edf_signal(file, channel)
    ecg = ecg[:fs * 10]  # 30 секунд
    filtered_ecg_1 = IRM.main(ecg, fs)
    filtered_ecg_2 = db8.main(ecg, fs)
    plt.subplot(3, 1, 1)
    plt.plot(ecg, label = 'Исходный сигнал', color='red')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(filtered_ecg_2, label = 'Сигнал после фильтрации DB-8', color='blue')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(filtered_ecg_1, label = 'Сигнал после фильтрации IRM', color='blue')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.legend()

    plt.savefig('Compare.png')

if __name__ == '__main__':
    main()
