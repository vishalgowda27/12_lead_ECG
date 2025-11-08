import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sp_signal
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Parameters
MITDB_PATH = r"E:\12 lead ecg\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0"
records = ["100", "101", "103", "105", "106", "108", "109", "111", "112", "113",
           "114", "115", "116", "117", "118", "119", "121", "122", "123", "124",
           "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
           "213", "214", "215", "217", "219", "220", "221", "222", "223", "228",
           "230", "231", "232", "233", "234"]

fs = 360  # Sampling frequency
half_window = int(0.1 * fs)
step = 4

X = []
y = []

print("Extracting features...")

for record in records:
    try:
        record_path = os.path.join(MITDB_PATH, record)
        ecg_signal, fields = wfdb.rdsamp(record_path, channels=[0])
        annotation = wfdb.rdann(record_path, 'atr')

        ecg = ecg_signal[:, 0]
        ecg = (ecg - np.mean(ecg)) / np.std(ecg)
        true_labels = np.zeros(len(ecg), dtype=int)
        true_labels[annotation.sample] = 1

        for i in range(half_window, len(ecg) - half_window, step):
            segment = ecg[i - half_window:i + half_window]
            X.append(segment)
            y.append(true_labels[i])

    except Exception as e:
        print(f"Error in record {record}: {e}")
        continue

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=int)
X = X[..., np.newaxis]

print(f"Total segments extracted: {X.shape[0]}")

# Model
print("Training model...")

model = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(256, kernel_size=3, activation='relu'),
    Conv1D(512, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    LSTM(64, return_sequences=True),
    LSTM(128),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X, y, epochs=10, batch_size=128, validation_split=0.1)

# Functions
def hilbert_transform(signal):
    analytic_signal = sp_signal.hilbert(signal)
    amplitude = np.abs(analytic_signal)
    phase = np.angle(analytic_signal)
    return amplitude, phase

def wvd(signal, fs):
    f, t, Zxx = sp_signal.stft(signal, fs, nperseg=256)
    return f, t, np.abs(Zxx)**2

# Visualization
print("Generating plots...")

sample_record = "100"
record_path = os.path.join(MITDB_PATH, sample_record)
ecg_signal, fields = wfdb.rdsamp(record_path, channels=[0])
annotation = wfdb.rdann(record_path, 'atr')

ecg = ecg_signal[:, 0]
ecg = (ecg - np.mean(ecg)) / np.std(ecg)
time = np.arange(len(ecg)) / fs
r_peaks = annotation.sample

# Apply HT
instantaneous_amplitude, instantaneous_phase = hilbert_transform(ecg)

# Apply WVD
f, t, Zxx = wvd(ecg, fs)

# Plot Raw ECG with R-peaks
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(time, ecg, label="Raw ECG")
plt.scatter(r_peaks / fs, ecg[r_peaks], color='red', s=10, label='R-peaks')
plt.title(f"Raw ECG with R-peaks (Record {sample_record})")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)

# Plot Instantaneous Amplitude
plt.subplot(2, 1, 2)
plt.plot(time, instantaneous_amplitude, label="Instantaneous Amplitude (HT)", color='purple')
plt.title("Hilbert Transform: Instantaneous Amplitude")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot WVD
plt.figure(figsize=(14, 6))
plt.pcolormesh(t, f, 10 * np.log10(Zxx), shading='auto')
plt.title('Wigner-Ville Distribution (STFT Approximation)')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.colorbar(label='Power [dB]')
plt.tight_layout()
plt.show()

# RR Intervals
rr_intervals = np.diff(r_peaks) / fs
rr_times = r_peaks[1:] / fs

plt.figure(figsize=(14, 3))
plt.plot(rr_times, rr_intervals, marker='o')
plt.title("R-R Intervals (seconds)")
plt.xlabel("Time (s)")
plt.ylabel("Interval (s)")
plt.grid(True)
plt.tight_layout()
plt.show()

# QRS Widths
qrs_widths = []
qrs_times = []
for r in r_peaks:
    start = max(r - int(0.05 * fs), 0)
    end = min(r + int(0.05 * fs), len(ecg))
    width = (end - start) / fs
    qrs_widths.append(width)
    qrs_times.append(r / fs)

plt.figure(figsize=(14, 3))
plt.plot(qrs_times, qrs_widths, marker='x')
plt.title("QRS Complex Widths")
plt.xlabel("Time (s)")
plt.ylabel("Width (s)")
plt.grid(True)
plt.tight_layout()
plt.show()

# PVC Detection
pvc_threshold = 0.6
pvc_flags = rr_intervals < pvc_threshold
pvc_indices = np.where(pvc_flags)[0] + 1
pvc_peaks = r_peaks[pvc_indices]

plt.figure(figsize=(14, 4))
plt.plot(time, ecg, label='ECG Signal')
plt.scatter(r_peaks / fs, ecg[r_peaks], color='green', s=10, label='R-peaks')
plt.scatter(pvc_peaks / fs, ecg[pvc_peaks], color='orange', s=30, marker='x', label='PVC Suspected')
plt.title(f"PVC Detection (Threshold = {pvc_threshold}s)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluation
print("Evaluating...")

y_pred_prob = model.predict(X).flatten()
y_pred = (y_pred_prob > 0.3).astype(int)

cm = confusion_matrix(y, y_pred)

if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = np.nan

accuracy = accuracy_score(y, y_pred)
sensitivity = recall_score(y, y_pred, zero_division=0)
specificity = recall_score(y, y_pred, pos_label=0, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
f2 = f1_score(y, y_pred, beta=2, zero_division=0)
ppv = precision_score(y, y_pred, zero_division=0)
npv = precision_score(y, y_pred, pos_label=0, zero_division=0)

# Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not R-peak", "R-peak"],
            yticklabels=["Not R-peak", "R-peak"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# Final Metrics
print("\n----- OVERALL EVALUATION -----")
print(f"Accuracy                : {accuracy:.4f}")
print(f"Sensitivity (Recall)    : {sensitivity:.4f}")
print(f"Specificity             : {specificity:.4f}")
print(f"F1 Score                : {f1:.4f}")
print(f"F2 Score                : {f2:.4f}")
print(f"Positive Predictive Val : {ppv:.4f}")
print(f"Negative Predictive Val : {npv:.4f}")

