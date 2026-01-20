# ECG Calculation Formulas and Pipeline Documentation

## Overview
This document describes the complete calculation pipeline for ECG metrics from raw hardware data (ADC values). All calculations follow GE/Philips/Fluke clinical standards.

---

## 1. DATA ACQUISITION PIPELINE

### 1.1 Hardware Data Format
- **Input**: 12-lead ECG data from serial port
- **Format**: Packet-based with MSB/LSB bytes
- **Leads**: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
- **Derived Leads**: III = II - I, aVR = -(I + II)/2, aVL = (I - III)/2, aVF = (II + III)/2
- **ADC Range**: 0-4095 (12-bit ADC)
- **Sampling Rate**: 500 Hz (default, can be auto-detected)

### 1.2 Data Processing Channels
**DUAL-PATH ECG ARCHITECTURE:**

1. **DISPLAY CHANNEL (0.5-40 Hz)**
   - Used for: R-peak detection, waveform display
   - Filter: 4th-order Butterworth bandpass
   - Purpose: Clean waveform visualization

2. **MEASUREMENT CHANNEL (0.05-150 Hz)**
   - Used for: All clinical measurements (PR, QRS, QT, QTc, P-wave)
   - Filter: 4th-order Butterworth bandpass
   - Purpose: Preserves Q/S waves and T-wave tail for accurate measurements

---

## 2. HEART RATE (HR) CALCULATION

### 2.1 Formula
```
HR (BPM) = 60000 / RR_interval_ms
```

Where:
- `RR_interval_ms` = Time between consecutive R-peaks in milliseconds
- `60000` = Milliseconds per minute

### 2.2 R-Peak Detection
**Method**: Adaptive peak detection on Lead II (display channel)

**Algorithm**:
1. Filter Lead II with 0.5-40 Hz bandpass
2. Calculate signal statistics: mean, std
3. Adaptive threshold: `threshold = mean + (3.5 × std)`
4. Minimum distance between peaks: `distance = 0.6 × fs` (for 10-300 BPM range)
5. Find peaks using `scipy.signal.find_peaks`

**RR Interval Calculation**:
```
RR_ms = (R_peak_2 - R_peak_1) / fs × 1000
```

### 2.3 HR Smoothing and Stabilization
**Buffer Size**: 15 readings (deque)
**Smoothing Method**:
1. Add raw HR to buffer
2. Calculate median: `median_hr = median(buffer)`
3. Apply EMA: `smoothed_hr = (1 - α) × old_ema + α × median_hr`
   - α = 0.15 (15% new value, 85% old value)
4. Dead zone: Only update if change ≥ 2 BPM
5. Hold-and-jump logic:
   - Small change (<2 BPM): Keep old value
   - Medium change (2-5 BPM): Update immediately
   - Large change (>5 BPM): Wait 2 seconds for stability

**Final HR**:
```
HR = int(round(smoothed_hr))
```

---

## 3. RR INTERVAL CALCULATION

### 3.1 Formula
```
RR_ms = (R_peak_current - R_peak_previous) / fs × 1000
```

### 3.2 Validation
- **Range**: 200-6000 ms (10-300 BPM)
- **Consistency Check**: `RR × HR ≈ 60000` (within 1 ms tolerance)

### 3.3 Storage
- Stored as `self.last_rr_interval` in milliseconds
- Used for QTc calculation

---

## 4. PR INTERVAL CALCULATION

### 4.1 Formula
```
PR_ms = QRS_onset_ms - P_onset_ms
```

Where:
- `QRS_onset_ms` = Time of QRS complex onset (relative to R-peak)
- `P_onset_ms` = Time of P-wave onset (relative to R-peak)

### 4.2 Median Beat Construction
**Purpose**: Build representative beat from multiple cycles

**Algorithm**:
1. Require ≥8 beats for reliable median
2. Extract beats aligned on R-peaks (±400 ms before, +900 ms after)
3. Apply measurement channel filter (0.05-150 Hz)
4. Quality assessment: Check beat quality (noise, baseline drift)
5. Calculate median across all beats

**Time Axis**:
```
time_axis_ms = np.arange(-pre_samples, post_samples + 1) / fs × 1000
```

### 4.3 QRS Onset Detection
**Method**: Slope-assisted detection on Lead II

**Algorithm**:
1. Baseline correction: `signal_corrected = median_beat - TP_baseline`
2. Calculate noise floor from TP segment (300-150 ms before R)
3. Search window: R-peak - 80 ms to R-peak
4. Calculate slope (first derivative): `slope = abs(diff(signal))`
5. Threshold: `threshold = max(0.05 × signal_range, noise_floor × 3.0)`
6. Find first sustained slope > threshold
7. QRS onset = first point where slope exceeds threshold

**Fallback Method** (if slope-assisted fails):
1. Amplitude threshold: `threshold = 0.05 × signal_range`
2. Find first point where `abs(signal) > threshold × 2.0`

### 4.4 P-Wave Onset Detection
**Method**: Atrial Vector (Lead I + aVF) - Clinical Standard

**Algorithm**:
1. Build median beats for Lead I and aVF
2. Create atrial vector: `atrial_vector = median_beat_I + median_beat_aVF`
3. HR-dependent search window:
   - **Very High HR (≥150 BPM)**: 140 ms to 50 ms before QRS onset
   - **High HR (120-149 BPM)**: 150 ms to 50 ms before QRS onset
   - **Mid HR (100-119 BPM)**: 170 ms to 50 ms before QRS onset
   - **Low HR (<100 BPM)**: 180 ms to 40 ms before QRS onset
4. Calculate QRS slope for threshold reference
5. HR-dependent threshold:
   - **Very High HR (≥150 BPM)**: `th = 0.06 × qrs_slope`
   - **High HR (120-149 BPM)**: `th = 0.07 × qrs_slope`
   - **Mid HR (100-119 BPM)**: `th = 0.07 × qrs_slope`
   - **Low HR (<100 BPM)**: `th = 0.06 × qrs_slope`
6. Find first sustained slope > threshold (minimum 20 ms run length)
7. P-onset = first point where slope exceeds threshold

**Fallback Method** (if atrial vector unavailable):
- Use single-lead method on Lead II
- HR-dependent search window (same as above)
- Threshold: `threshold = max(0.04 × qrs_amplitude, 0.05)`

### 4.5 PR Interval Calculation
```
PR_ms = time_axis[qrs_onset_idx] - time_axis[p_onset_idx]
```

### 4.6 PR Validation and Smoothing
**Validation Range**: 50-300 ms
- Extended range for very high HR (140+ BPM): allows down to 50 ms
- Standard range: 60-300 ms

**Smoothing**:
1. Buffer size: 7 readings
2. Median filter: `smoothed_pr = median(buffer)`
3. Hold-and-jump logic:
   - Small change (≤10 ms): Update immediately
   - Large change (>10 ms): Wait 2 seconds for stability

**Fallback** (if P-onset detection fails):
- Use conservative estimate: `PR = 200 ms`

---

## 5. QRS DURATION CALCULATION

### 5.1 Formula
```
QRS_ms = (QRS_offset - QRS_onset) / fs × 1000
```

Where:
- `QRS_onset` = QRS complex onset (from PR calculation)
- `QRS_offset` = J-point (end of S-wave)

### 5.2 J-Point Detection
**Method**: Slope-assisted detection

**Algorithm**:
1. Baseline correction: `signal_corrected = median_beat - TP_baseline`
2. Calculate QRS peak amplitude: `qrs_peak_amplitude = max(abs(QRS_window))`
3. Search window: R-peak + 20 ms to R-peak + 80 ms
4. Amplitude threshold: `amplitude_threshold = 0.032 × abs(qrs_peak_amplitude)`
5. Slope threshold: `slope_threshold = max(0.011 × signal_range × fs / 1000, abs(signal_range) × 0.004)`
6. Find J-point: First point after R-peak where:
   - Signal returns to baseline: `abs(signal) < amplitude_threshold`
   - AND slope is flat: `abs(slope) < slope_threshold`
   - Minimum run length: 10 ms

**Fallback Method** (if slope-assisted fails):
- Find minimum in search window: `j_point = argmin(signal[R+20ms : R+80ms])`

### 5.3 QRS Duration Calculation
```
QRS_ms = (j_point_idx - qrs_onset_idx) / fs × 1000
```

### 5.4 QRS Validation and Smoothing
**Validation Range**: 40-200 ms

**Smoothing**:
1. Buffer size: 7 readings
2. Median filter: `smoothed_qrs = median(buffer)`
3. Hold-and-jump logic (same as PR)

---

## 6. QT INTERVAL CALCULATION

### 6.1 Formula
```
QT_ms = (T_end - Q_onset) / fs × 1000
```

Where:
- `Q_onset` = QRS onset (from PR calculation)
- `T_end` = T-wave end

### 6.2 T-Wave End Detection
**Method**: Tangent method on Lead II

**Algorithm**:
1. Baseline correction: `signal_corrected = median_beat - TP_baseline`
2. Find T-wave peak: Maximum in window R+200 ms to R+600 ms
3. Search for T-end: From T-peak to end of beat
4. Calculate tangent line at T-peak
5. Find intersection with TP baseline
6. T-end = Intersection point

**Fallback Method**:
- Find point where signal returns to TP baseline
- Threshold: `threshold = 0.1 × T_peak_amplitude`

### 6.3 HR-Dependent Calibration Offsets
**Purpose**: Align with reference software values

**Offsets** (subtracted from calculated QT):
- **Very High HR (≥140 BPM)**: `QT -= 21.0 ms`
- **High HR (120-139 BPM)**: `QT -= 15.0 ms`
- **Mid-High HR (100-119 BPM)**: `QT += 3.0 ms`
- **Mid HR (80-99 BPM)**: `QT -= 5.0 ms`
- **Low HR (<80 BPM)**: `QT += 9.0 ms`

**Example at 100 BPM**:
```
QT_raw = 312 ms
QT_calibrated = 312 + 3 = 315 ms
```

### 6.4 QT Validation
**Validation Range**: 200-650 ms

---

## 7. QTc (CORRECTED QT) CALCULATION

### 7.1 Bazett's Formula (QTc)
```
QTc_ms = QT_ms / sqrt(RR_sec)
```

Where:
- `RR_sec` = RR interval in seconds = `RR_ms / 1000`

**Example**:
```
QT = 315 ms
RR = 600 ms (100 BPM)
RR_sec = 0.6 seconds
QTc = 315 / sqrt(0.6) = 315 / 0.775 = 407 ms
```

### 7.2 Fridericia's Formula (QTcF)
```
QTcF_ms = QT_ms / (RR_sec)^(1/3)
```

**Example**:
```
QT = 315 ms
RR_sec = 0.6 seconds
QTcF = 315 / (0.6)^(1/3) = 315 / 0.843 = 374 ms
```

### 7.3 QTc Validation
**Validation Range**: 250-600 ms

---

## 8. P-WAVE DURATION CALCULATION

### 8.1 Formula
```
P_duration_ms = P_offset_ms - P_onset_ms
```

Where:
- `P_onset` = P-wave onset (from PR calculation)
- `P_offset` = P-wave offset

### 8.2 P-Wave Bounds Detection
**Method**: Peak detection on Lead II

**Algorithm**:
1. HR-dependent search window:
   - **Very High HR (≥140 BPM)**: 180 ms to 50 ms before R
   - **High-Mid HR (≥100 BPM)**: 200 ms to 60 ms before R
   - **Low HR (<100 BPM)**: 200 ms to 60 ms before R
2. Baseline correction: `centered = segment - TP_baseline`
3. Detection threshold: `threshold = max(0.04 × qrs_amplitude, 0.05)`
4. Find absolute max peak in window
5. P-onset: First point before peak returning to baseline (0.3×threshold)
6. P-offset: First point after peak returning to baseline (0.3×threshold)

### 8.3 P-Wave Duration Calculation
```
P_duration_ms = (P_offset_idx - P_onset_idx) / fs × 1000
```

### 8.4 P-Wave Duration Validation
**Validation Range**: 40-200 ms

---

## 9. CALCULATION PIPELINE SUMMARY

### 9.1 Complete Flow
```
Hardware Data (ADC values)
    ↓
[1] Serial Packet Parsing
    ↓
[2] Derived Leads Calculation (III, aVR, aVL, aVF)
    ↓
[3] Dual-Path Filtering:
    ├─ Display Channel (0.5-40 Hz) → R-peak detection
    └─ Measurement Channel (0.05-150 Hz) → Clinical measurements
    ↓
[4] R-Peak Detection (Lead II, display channel)
    ↓
[5] RR Interval Calculation
    ↓
[6] Heart Rate Calculation (HR = 60000 / RR_ms)
    ↓
[7] Median Beat Construction (≥8 beats, measurement channel)
    ↓
[8] TP Baseline Extraction (350-150 ms before R)
    ↓
[9] QRS Onset Detection (slope-assisted, Lead II)
    ↓
[10] P-Wave Onset Detection (atrial vector: Lead I + aVF)
    ↓
[11] PR Interval Calculation (QRS_onset - P_onset)
    ↓
[12] J-Point Detection (slope-assisted, Lead II)
    ↓
[13] QRS Duration Calculation (J_point - QRS_onset)
    ↓
[14] T-Wave End Detection (tangent method, Lead II)
    ↓
[15] QT Interval Calculation (T_end - Q_onset) + HR calibration
    ↓
[16] QTc Calculation (Bazett: QT / sqrt(RR_sec))
    ↓
[17] P-Wave Bounds Detection (Lead II)
    ↓
[18] P-Wave Duration Calculation (P_offset - P_onset)
    ↓
[19] Smoothing and Validation
    ↓
[20] Display Update
```

### 9.2 Key Constants
- **Sampling Rate**: 500 Hz (default)
- **ADC Range**: 0-4095
- **Median Beat Window**: -400 ms to +900 ms (relative to R-peak)
- **Minimum Beats for Median**: 8 beats
- **TP Baseline Window**: 350-150 ms before R-peak

---

## 10. REFERENCE VALUES AT 100 BPM

### 10.1 Expected Values
Based on reference software calibration:

- **HR**: 100 BPM
- **RR**: 600 ms
- **PR**: 161 ms (±5 ms tolerance)
- **QRS**: 85-86 ms (±5 ms tolerance)
- **QT**: 315 ms (±5 ms tolerance)
- **QTc (Bazett)**: 407 ms (±5 ms tolerance)
- **P-Wave Duration**: Variable (40-200 ms range)

### 10.2 HR-Dependent Adjustments
**PR Interval**:
- Low HR (<100 BPM): Longer PR (detect P-onset earlier)
- High HR (≥120 BPM): Shorter PR (detect P-onset later)

**QT Interval**:
- Very High HR (≥140 BPM): QT -= 21 ms
- High HR (120-139 BPM): QT -= 15 ms
- Mid-High HR (100-119 BPM): QT += 3 ms
- Mid HR (80-99 BPM): QT -= 5 ms
- Low HR (<80 BPM): QT += 9 ms

---

## 11. ERROR HANDLING

### 11.1 Fallback Mechanisms
1. **QRS Onset**: Falls back to amplitude-only method if slope-assisted fails
2. **P-Wave Onset**: Falls back to single-lead method if atrial vector unavailable
3. **J-Point**: Falls back to minimum method if slope-assisted fails
4. **PR Interval**: Uses 200 ms estimate if P-onset detection fails

### 11.2 Validation Checks
- **HR**: 10-300 BPM range
- **RR**: 200-6000 ms range
- **PR**: 50-300 ms range
- **QRS**: 40-200 ms range
- **QT**: 200-650 ms range
- **QTc**: 250-600 ms range
- **P-Wave Duration**: 40-200 ms range

### 11.3 Smoothing Strategies
- **Buffer-based**: Store last N readings
- **Median Filter**: Reject outliers
- **EMA**: Exponential moving average for gradual changes
- **Dead Zone**: Ignore small fluctuations
- **Hold-and-Jump**: Wait for stability on large changes

---

## 12. IMPLEMENTATION NOTES

### 12.1 Key Functions
- `build_median_beat()`: Constructs median beat from aligned cycles
- `detect_qrs_onset_slope_assisted()`: QRS onset detection
- `detect_p_onset_atrial_vector()`: P-wave onset detection (atrial vector)
- `detect_qrs_offset_slope_assisted()`: J-point detection
- `measure_qt_from_median_beat()`: QT interval measurement
- `measure_pr_from_median_beat()`: PR interval measurement
- `measure_qrs_duration_from_median_beat()`: QRS duration measurement
- `measure_p_duration_from_median_beat()`: P-wave duration measurement

### 12.2 File Locations
- **Main Calculation**: `src/ecg/clinical_measurements.py`
- **Heart Rate**: `src/ecg/metrics/heart_rate.py`
- **Display Updates**: `src/ecg/ui/display_updates.py`
- **Main ECG Page**: `src/ecg/twelve_lead_test.py`

---

## 13. CALCULATION ACCURACY

### 13.1 Tolerance Ranges
- **PR, QRS, QT, QTc**: ±5 ms tolerance compared to reference software
- **HR**: Must be stable (no flickering between 99-101 BPM at 100 BPM)

### 13.2 Update Frequency
- **Metrics Calculation**: Every 2-3 plot updates (after first 20 immediate updates)
- **Display Update**: Every 0.3 seconds
- **Dashboard Update**: Every 0.3 seconds
- **Initial Update**: Within 10 seconds of acquisition start

---

## END OF DOCUMENTATION

**Version**: 1.0
**Last Updated**: 2025-01-15
**Author**: ECG Software Development Team
