# ==========================================
# Digital Filters for ECG Signal (CSV I/O)
# ==========================================

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load ECG from CSV
# -------------------------------
# Assumes CSV has one column named 'ECG' or change accordingly
input_csv = "/content/drive/MyDrive/ECG_input.csv"
ecg_df = pd.read_csv(input_csv)

# Convert to NumPy array
ecg_signal = ecg_df['ECG'].values
fs = 500  # Sampling frequency (Hz) - adjust if needed
t = np.arange(len(ecg_signal)) / fs

# -------------------------------
# Step 2: Design digital filters
# -------------------------------

def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

# -------------------------------
# Step 3: Apply filters
# -------------------------------

# High-pass filter (remove baseline wander)
b, a = butter_highpass(0.5, fs)
ecg_hp = filtfilt(b, a, ecg_signal)

# Low-pass filter (remove high-frequency noise)
b, a = butter_lowpass(45, fs)
ecg_lp = filtfilt(b, a, ecg_hp)

# Notch filter (remove 50 Hz powerline interference)
f0 = 50.0  # Frequency to remove
Q = 30.0   # Quality factor
b, a = iirnotch(f0, Q, fs)
ecg_filtered = filtfilt(b, a, ecg_lp)

# -------------------------------
# Step 4: Save filtered ECG to CSV
# -------------------------------
output_csv = "/content/drive/MyDrive/ECG_filtered.csv"
filtered_df = pd.DataFrame({'ECG_filtered': ecg_filtered})
filtered_df.to_csv(output_csv, index=False)
print(f"Filtered ECG saved to: {output_csv}")

# -------------------------------
# Step 5: Optional Plot
# -------------------------------
plt.figure(figsize=(12,4))
plt.plot(t, ecg_signal, label='Raw ECG', alpha=0.5)
plt.plot(t, ecg_filtered, label='Filtered ECG', color='red')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("ECG Signal Filtering")
plt.legend()
plt.show()
