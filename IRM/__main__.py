from . import main
import mne
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='Путь к EDF-файлу')
    parser.add_argument('-d', '--duration', type=str, help='Желаемая длина сигнала для обработки(по умолчанию весь сигнал)')
    parser.add_argument('-o', '--output', type=str, help='Название файла, в котором будет сохранён результат без расширения(по умолчанию result)')
    args = parser.parse_args()
    name = args.file
    if not os.path.exists(str(name)):
        raise FileNotFoundError(f"EDF file '{file_path}' not found.")
    record = mne.io.read_raw_edf(name, preload=True)
    info = record.info
    fs = int(info['sfreq'])

    channels = record.ch_names

    record_1, times = record.get_data(return_times=True, picks=channels[0])
    if args.duration:
        l = min(len(record_1[0]), int(args.duration) * fs)
    else:
        l = len(record_1[0])
    record_1 = record_1[0][:l]
    result = main(record_1, fs)

    plt.figure().set_figwidth(l / fs * 2)
    plt.xticks(np.arange(0, len(result)+1, 150))
    plt.xlabel('Samples')
    plt.ylabel('MLIImV')
    plt.plot(record_1, label='Исходный сигнал', color='red')
    plt.plot(result, label='Результат', color='blue')
    plt.legend()
    if args.output:
        plt.savefig(f'{args.output}.png')
    else:
        plt.savefig('result.png')
