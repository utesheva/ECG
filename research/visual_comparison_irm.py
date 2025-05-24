import IRM
import os
import mne
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
    results = {}
    for edf_file in edf_files:
        print(f"Файл: {edf_file}")
        ecg, fs = load_edf_signal(os.path.join(directory, edf_file), channel)
        ecg = ecg[:fs * 30]  # 60 секунд
        filtered_ecg = IRM.main(ecg, fs)
        results[edf_file] = [ecg, filtered_ecg]
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, help='Путь к директории с EDF-файлами сигнала')
    parser.add_argument('-c', '--sig_channel', type=str,
                        help='Имя канала EDF для полезного сигнала (по умолчанию первый)')
    args = parser.parse_args()

    results = main(directory=args.directory, channel=args.sig_channel)
    k = 1
    pic = 0
    plt.figure(figsize=(25,35))
    for name in results:
        plt.subplot(10, 1, k)
        plt.plot(results[name][0], label = 'Сигнал с шумом', color='red')
        plt.plot(results[name][1], label = 'Сигнал после фильтрации', color='blue')
        plt.xlabel('Samples')
        plt.ylabel('MLIImV')
        plt.legend()
        plt.title(name)
        k += 1
        if k == 11:
            plt.tight_layout()
            plt.savefig(f'{pic}.png')
            plt.clf()
            k = 1
            pic += 1
    plt.savefig(f'{pic}.png')
