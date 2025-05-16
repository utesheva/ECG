import time
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

def main(directory="SimECG/tests", channel=None):
    """Get mse and snr of the set"""
    edf_files = [f for f in os.listdir(directory) if f.endswith('.edf')]
    results = {'irm1':[], 'irm2': [], 'irm3': [], 'db8':[]}
    for edf_file in edf_files:
        ecg, fs = load_edf_signal(os.path.join(directory, edf_file), channel)
        ecg = ecg[:fs * 60]  # 60 секунд
        t1 = time.time()
        filtered_irm, it = IRM.main(ecg, fs)
        t2 = time.time()
        filtered_db8 = db8.main(ecg, fs)
        t3 = time.time()
        if it == 1:
            results['irm1'] = results['irm1'] + [(t2 - t1) / len(ecg)]
        if it == 2:
            results['irm2'] = results['irm2'] + [(t2 - t1) / len(ecg)]
        if it == 3:        
            results['irm3'] = results['irm3'] + [(t2 - t1) / len(ecg)]
        results['db8'] = results['db8'] + [(t3 - t2) / len(ecg)]
    return results

if __name__ == '__main__':
    res = main()
    print('IRM1 time:', sum(res['irm1']) / len(res['irm1']))
    print('IRM2 time:', sum(res['irm2']) / len(res['irm2']))
    print('IRM3 time:', sum(res['irm3']) / len(res['irm3']))
    print('DB8 time', sum(res['db8']) / len(res['db8']))
    res2 = main(directory='clean')
    res['irm1'] = res['irm1'] + res2['irm1']
    res['irm2'] = res['irm2'] + res2['irm2']
    res['irm3'] = res['irm3'] + res2['irm3']
    print('IRM1 time:', sum(res['irm1']) / len(res['irm1']))
    print('IRM2 time:', sum(res['irm2']) / len(res['irm2']))
    print('IRM3 time:', sum(res['irm3']) / len(res['irm3']))
