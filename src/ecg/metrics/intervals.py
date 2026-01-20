"""ECG interval calculations (QTcF, RV5/SV1)"""
import numpy as np
from typing import Optional, Tuple
from scipy.signal import butter, filtfilt, find_peaks
from ..clinical_measurements import measure_rv5_sv1_from_median_beat, build_median_beat


def calculate_qtcf_interval(qt_ms: float, rr_ms: float) -> int:
    """Calculate QTcF using Fridericia formula: QTcF = QT / RR^(1/3)
    
    Args:
        qt_ms: QT interval in milliseconds
        rr_ms: RR interval in milliseconds
    
    Returns:
        QTcF in milliseconds (rounded to integer)
    """
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0
        
        # Convert to seconds
        qt_sec = qt_ms / 1000.0
        rr_sec = rr_ms / 1000.0
        
        # Fridericia formula: QTcF = QT / RR^(1/3)
        qtcf_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
        
        # Convert back to milliseconds
        qtcf_ms = int(round(qtcf_sec * 1000.0))
        
        return qtcf_ms
    except Exception as e:
        print(f" Error calculating QTcF: {e}")
        return 0


def calculate_rv5_sv1_from_median(data: list, r_peaks: np.ndarray, fs: float) -> Tuple[Optional[float], Optional[float]]:
    """Calculate RV5 and SV1 from median beats (GE/Philips standard).
    
    Args:
        data: List of ECG data arrays (12 leads)
        r_peaks: R-peak indices from Lead II
        fs: Sampling rate in Hz
    
    Returns:
        Tuple of (rv5_mv, sv1_mv) in millivolts, or (None, None) if calculation fails
    """
    try:
        if len(data) < 11:
            return None, None
        
        # CRITICAL: Correct lead indices for 12-lead ECG
        # LEADS_MAP: ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        # Index 6 = V1, Index 10 = V5
        lead_v5_raw = np.asarray(data[10], dtype=float) if len(data) > 10 else None
        lead_v1_raw = np.asarray(data[6], dtype=float) if len(data) > 6 else None
        lead_ii = np.asarray(data[1], dtype=float)
        
        if lead_v5_raw is None or lead_v1_raw is None:
            return None, None
        
        if len(r_peaks) < 8:
            return None, None
        
        # Call measurement function using RAW data and shared R-peaks
        # ADC factors for V5/V1 (Marquette standards)
        rv5_mv, sv1_mv = measure_rv5_sv1_from_median_beat(
            lead_v5_raw, lead_v1_raw, 
            r_peaks, r_peaks,  # Use shared R-peaks for alignment
            fs, 
            v5_adc_per_mv=2048.0, 
            v1_adc_per_mv=1441.0
        )
        
        return rv5_mv, sv1_mv
    except Exception as e:
        print(f" Error calculating RV5/SV1 from median: {e}")
        return None, None
