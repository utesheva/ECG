import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import IRM
import spkit as sp
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample

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


def compute_snr(signal, noise):
    """Computing SNR"""
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    return 10 * np.log10(power_signal / power_noise)


def main(directory="tests", channel=None):
    """Get mse and snr of the set"""
    edf_files = [f for f in os.listdir(directory) if f.endswith('.edf')]
    results = []
    noise, fs = load_edf_signal('Noise.edf', 'ECG CM1')
    noise = np.array(noise[int(19.2*fs):23*fs])
    for edf_file in edf_files:
        print(f"Файл: {edf_file}")
        ecg, fs = load_edf_signal(os.path.join(directory, edf_file), channel)
        ecg = ecg[:fs * 60]  # 60 секунд
        noise = resample(noise, fs)
        noise = np.tile(noise, int(np.ceil(len(ecg)/len(noise))))[:len(ecg)]
        noisy_ecg = ecg + noise
        filtered_ecg = IRM.main(noisy_ecg, fs)
        plt.plot(noisy_ecg)
        plt.plot(filtered_ecg)
        plt.show()
        mse = np.mean((filtered_ecg - ecg) ** 2)
        snr = compute_snr(noisy_ecg, noise)
        results.append((snr, mse, edf_file))
    return results


if __name__ == '__main__':
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
    plt.show()

