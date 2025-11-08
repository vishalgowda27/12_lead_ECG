import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, BatchNormalization,
                                     GlobalAveragePooling1D, Dense, Dropout,
                                     LSTM, Reshape, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------- Preprocessing Filters ---------------------------
def highpass_filter(signal, fs, cutoff=0.5, order=16):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='high')
    return filtfilt(b, a, signal)

def lowpass_filter(signal, fs, cutoff=40.0, order=16):
    nyquist = 0.5 * fs
    b, a = butter(order, cutoff / nyquist, btype='low')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, freq=50.0, Q=30.0):
    nyquist = 0.5 * fs
    b, a = iirnotch(freq / nyquist, Q)
    return filtfilt(b, a, signal)

def preprocess_ecg(signal, fs):
    signal = highpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = lowpass_filter(signal, fs)
    return signal

# --------------------------- Feature Extraction ---------------------------
def detect_r_peaks(signal, fs):
    distance = int(0.2 * fs)
    peaks, _ = find_peaks(signal, distance=distance, prominence=0.5)
    return peaks

def calculate_rr_intervals(r_peaks, fs):
    rr_intervals = np.diff(r_peaks) / fs
    return rr_intervals

def calculate_qrs_width(signal, r_peaks, fs):
    qrs_widths = []
    for r in r_peaks:
        left = max(0, r - int(0.05 * fs))
        right = min(len(signal), r + int(0.05 * fs))
        qrs_widths.append((right - left) / fs)
    return np.array(qrs_widths)

# --------------------------- Data Loader ---------------------------
def load_mitbih_data(records_dir, records):
    all_signals = []
    all_labels = []
    fs = 360

    for rec in records:
        record_path = os.path.join(records_dir, str(rec))
        try:
            signal, fields = wfdb.rdsamp(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except Exception as e:
            print(f"Error loading {record_path}: {e}")
            continue

        sig = preprocess_ecg(signal[:, 0], fs)
        r_peaks = detect_r_peaks(sig, fs)

        for r in r_peaks:
            if r - 100 < 0 or r + 100 > len(sig):
                continue
            beat = sig[r - 100:r + 100]
            all_signals.append(beat)
            idx = np.argmin(np.abs(annotation.sample - r))
            label = annotation.symbol[idx]
            all_labels.append(1 if label == 'V' else 0)

    return np.array(all_signals), np.array(all_labels)

# --------------------------- Model Definition ---------------------------
def build_cnn_lstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))

    model.add(Conv1D(512, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling1D())

    model.add(Reshape((1, 512)))  # Corrected shape
    model.add(LSTM(64))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --------------------------- Evaluation ---------------------------
def evaluate_model(model, X, y, title=""):
    y_pred_probs = model.predict(X, batch_size=32)
    y_pred = (y_pred_probs > 0.5).astype(int)

    accuracy = accuracy_score(y, y_pred)
    sensitivity = recall_score(y, y_pred)
    specificity = recall_score(y, y_pred, pos_label=0)
    precision = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    cm = confusion_matrix(y, y_pred)
    npv = 0
    if cm.shape == (2, 2):
        npv = cm[0, 0] / (cm[0, 0] + cm[1, 0])

    try:
        auc = roc_auc_score(y, y_pred_probs)
    except ValueError:
        auc = 0.5

    print(f"\n--- {title} ---")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"NPV:          {npv:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"AUC:          {auc:.4f}")
    print("Confusion Matrix:\n", cm)

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-PVC', 'PVC'], yticklabels=['Non-PVC', 'PVC'])
    plt.title(f"{title} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# --------------------------- Training History Plot ---------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# --------------------------- Main Execution ---------------------------
if __name__ == "__main__":
    records_dir = "mit-bih-arrhythmia-database-1.0.0"
    records = [230, 209, 108, 100, 220, 117, 104, 114, 102, 207, 107, 213, 219, 205, 233, 210,
               232, 228, 111, 201, 231, 105, 203, 123, 212, 113, 223, 119, 116, 103, 215, 121,
               118, 109, 106, 214, 234, 115, 101, 112, 122, 200, 124, 222, 208, 202, 221, 217]
    fs = 360

    print("Loading data...")
    X, y = load_mitbih_data(records_dir, records)
    X = X.reshape(-1, 200, 1)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("Building model...")
    model = build_cnn_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Training model...")
    history = model.fit(X_train, y_train, epochs=15, batch_size=32,
                        validation_split=0.1, callbacks=[early_stop], verbose=1)

    # Evaluation
    evaluate_model(model, X_train, y_train, title="Training Data Evaluation(PVC)")
    evaluate_model(model, X_test, y_test, title="Testing Data Evaluation")

    # Training plots
    plot_training_history(history)

    # --------------------------- Plotting ECG Features ---------------------------
    print("Plotting ECG Features...")
    try:
        signal, fields = wfdb.rdsamp(os.path.join(records_dir, str(records[0])))
        raw = signal[:, 0]
        filtered = preprocess_ecg(raw, fs)
        r_peaks = detect_r_peaks(filtered, fs)
        rr_intervals = calculate_rr_intervals(r_peaks, fs)
        qrs_widths = calculate_qrs_width(filtered, r_peaks, fs)
        t = np.arange(len(raw)) / fs

        # Raw ECG
        plt.figure(figsize=(12, 4))
        plt.plot(t, raw, label='Raw ECG')
        plt.title("Raw ECG Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
        plt.show()

        # Filtered ECG
        plt.figure(figsize=(12, 4))
        plt.plot(t, filtered, label='Filtered ECG')
        plt.plot(r_peaks / fs, filtered[r_peaks], 'ro')
        plt.title("Filtered ECG with R-peaks")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.legend()
        plt.show()

        # RR Intervals
        plt.figure(figsize=(8, 3))
        plt.plot(rr_intervals, marker='o')
        plt.title("RR Intervals")
        plt.ylabel("Interval (s)")
        plt.grid()
        plt.show()

        # QRS Widths
        plt.figure(figsize=(8, 3))
        plt.plot(qrs_widths, marker='o')
        plt.title("QRS Widths")
        plt.ylabel("Width (s)")
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Error in ECG feature plotting: {e}")
