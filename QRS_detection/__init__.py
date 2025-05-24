import scipy.signal as sg
import numpy as np

class Pan_Tompkins_QRS():
    def __init__(self, fs):
        self.fs = fs

    def band_pass_filter(self,signal):
        b, a = sg.butter(1, [5, 15], btype='bandpass', fs=self.fs)
        return sg.lfilter(b, a, signal)

    def derivative(self,signal):
        signal = np.pad(signal, (2, 2), mode='constant', constant_values=0)
        result = signal.copy()
        for index in range(2, len(signal) - 2):
            result[index] = (-2 * signal[index-1] - signal[index-2] 
                             + 2 * signal[index+1] + signal[index+2]) * self.fs / 8
        return result[2:-2]

    def squaring(self,signal):
        return np.square(signal)

    def moving_window_integration(self,signal):
        result = signal.copy()
        win_size = round(0.150 * self.fs)
        summary = 0

        for j in range(win_size):
            summary += signal[j]/win_size
            result[j] = summary
    
        for index in range(win_size,len(signal)):  
            summary += signal[index]/win_size
            summary -= signal[index-win_size]/win_size
            result[index] = summary

        return result

    def solve(self,signal):

        global bpass
        bpass = self.band_pass_filter(signal.copy())

        global der
        der = self.derivative(bpass.copy())

        global sqr
        sqr = self.squaring(der.copy())

        global mwin
        mwin = self.moving_window_integration(sqr.copy())

        return mwin

class heart_rate():

    def __init__(self,signal,samp_freq):
        self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for i in range(6))
        self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (0 for i in range(8))

        self.T_wave = False
        self.m_win = mwin
        self.b_pass = bpass
        self.samp_freq = samp_freq
        self.signal = signal
        self.win_150ms = round(0.15*self.samp_freq)

        self.RR_Low_Limit = 0
        self.RR_High_Limit = 0
        self.RR_Missed_Limit = 0
        self.RR_Average1 = 0

    def approx_peak(self):
        slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

        for i in range(round(0.5*self.samp_freq) + 1,len(slopes)-1):
            if (slopes[i] > slopes[i-1]) and (slopes[i+1] < slopes[i]):
                self.peaks.append(i)


    def adjust_rr_interval(self,ind):
        self.RR1 = np.diff(self.peaks[max(0,ind - 8) : ind + 1])/self.samp_freq

        self.RR_Average1 = np.mean(self.RR1)
        RR_Average2 = self.RR_Average1

        if (ind >= 8):
            for i in range(0, 8):
                if (self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit):
                    self.RR2.append(self.RR1[i])

                    if (len(self.RR2) > 8):
                        self.RR2.remove(self.RR2[0])
                        RR_Average2 = np.mean(self.RR2)

        if (len(self.RR2) > 7 or ind < 8):
            self.RR_Low_Limit = 0.92 * RR_Average2
            self.RR_High_Limit = 1.16 * RR_Average2
            self.RR_Missed_Limit = 1.66 * RR_Average2


    def searchback(self,peak_val,RRn,sb_win):
        if (RRn > self.RR_Missed_Limit):
            win_rr = self.m_win[peak_val - sb_win + 1 : peak_val + 1]
            coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]
            if (len(coord) > 0):
                for pos in coord:
                    if (win_rr[pos] == max(win_rr[coord])):
                        x_max = pos
                        break
            else:
                x_max = None
            if (x_max is not None):
                self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1

                win_rr = self.b_pass[x_max - self.win_150ms: min(len(self.b_pass) -1, x_max)]
                coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]

                if (len(coord) > 0):
                    for pos in coord:
                        if (win_rr[pos] == max(win_rr[coord])):
                            r_max = pos
                            break
                else:
                    r_max = None

                if (r_max is not None):
                    if self.b_pass[r_max] > self.Threshold_F2:
                        self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1
                        self.r_locs.append(r_max)


    def find_t_wave(self,peak_val,RRn,ind,prev_ind):
        if (self.m_win[peak_val] >= self.Threshold_I1):
            if (ind > 0 and 0.20 < RRn < 0.36):
                curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms/2) : peak_val + 1]))
                last_slope = max(np.diff(self.m_win[self.peaks[prev_ind] - round(self.win_150ms/2) : self.peaks[prev_ind] + 1]))
                if (curr_slope < 0.5*last_slope):
                    self.T_wave = True
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI

            if (not self.T_wave):
                if (self.probable_peaks[ind] > self.Threshold_F1):
                    self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI
                    self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF
                    self.r_locs.append(self.probable_peaks[ind])

                else:
                    self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

        elif (self.m_win[peak_val] < self.Threshold_I1) or (self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
            self.NPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def adjust_thresholds(self,peak_val,ind):
        if (self.m_win[peak_val] >= self.Threshold_I1):
            self.SPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.SPKI

            if (self.probable_peaks[ind] > self.Threshold_F1):
                self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF
                self.r_locs.append(self.probable_peaks[ind])
            else:
                self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

        elif (self.m_win[peak_val] < self.Threshold_I2) or (self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
            self.NPKI = 0.125 * self.m_win[peak_val]  + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF


    def update_thresholds(self):
        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False

    def ecg_searchback(self):
        self.r_locs = np.unique(np.array(self.r_locs).astype(int))
        win_200ms = round(0.2*self.samp_freq)
        for r_val in self.r_locs:
            coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + win_200ms + 1), 1)
            if (len(coord) > 0):
                for pos in coord:
                    if (self.signal[pos] == max(self.signal[coord])):
                        x_max = pos
                        break
            else:
                x_max = None
            if (x_max is not None):
                self.result.append(x_max)


    def find_r_peaks(self):
        self.approx_peak()

        for ind in range(len(self.peaks)):
            peak_val = self.peaks[ind]
            win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms), min(self.peaks[ind] + self.win_150ms, len(self.b_pass)-1), 1)
            max_val = max(self.b_pass[win_300ms], default = 0)
            if (max_val != 0):
                x_coord = np.asarray(self.b_pass == max_val).nonzero()
                self.probable_peaks.append(x_coord[0][0])
            if (ind < len(self.probable_peaks) and ind != 0):
                self.adjust_rr_interval(ind)
                if (self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit):
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2
                RRn = self.RR1[-1]
                self.searchback(peak_val, RRn, round(RRn*self.samp_freq))
                self.find_t_wave(peak_val,RRn,ind,ind-1)
            else:
                self.adjust_thresholds(peak_val,ind)
            self.update_thresholds()
        self.ecg_searchback()
        return self.result



