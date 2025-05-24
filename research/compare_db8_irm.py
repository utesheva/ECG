import IRM
import db8
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import sys

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

def main(file, channel=None):
    ecg, fs = load_edf_signal(file, channel)
    ecg = ecg[:fs * 60]  # 60 секунд
    irm = IRM.main(ecg, fs)
    db = db8.main(ecg, fs)
    plt.subplot(3, 1, 1)
    plt.plot(ecg, label = 'Исходный сигнал', color='red')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(db, label = 'Сигнал после фильтрации DB-8', color='blue')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(irm, label = 'Сигнал после фильтрации IRM', color='blue')
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Compare DB-8 and IRM.png')

if __name__ == '__main__':
    main(sys.argv[1])

