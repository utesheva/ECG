import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import time
import IRM
import db8
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def main(directory="tests", channel=None):
    edf_files = [f for f in os.listdir(directory) if f.endswith('.edf')]
    results = []
    for edf_file in edf_files:
        ecg, fs = load_edf_signal(os.path.join(directory, edf_file), channel)
        ecg = ecg[:fs*60]
        t1 = time.time()
        filtered_irm = IRM.main(ecg, fs)
        t2 = time.time()
        results.append((t2-t1) / len(ecg) * 1000)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='Путь к директоии с EDF-файлами')
    parser.add_argument('-c', '--sig_channel', type=str,
                        help='Имя канала EDF для полезного сигнала (по умолчанию первый)')
    args = parser.parse_args()
    res = main(directory=args.directory, channel=args.sig_channel)
    print(sum(res) / len(res))
