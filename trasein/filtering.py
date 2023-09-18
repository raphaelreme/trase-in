import numpy as np
import scipy.stats  # type: ignore


def is_noise(signals: np.ndarray, threshold: float) -> np.ndarray:
    """Check if the signal matches a gaussian distribution

    Args:
        signals (np.ndarray): Multiple signals to check
            Shape: (n_signals, n_frames)

    Returns:
        np.ndarray: True or false for each signal
            Shape: (n_signals, )
    """
    _, p_values = scipy.stats.normaltest(signals, axis=1)

    return p_values > threshold


# def has_peaks(signals, global_peak_quantile, local_quantile):
#     """check if there is peaks based on signals. Assume that there are some peaks"""
#     return np.quantile(signals, local_quantile, axis=1) <= np.quantile(signals, global_peak_quantile)


# def has_peaks(signals: np.ndarray, threshold: float) -> np.ndarray:
#     """Check if signals has peaks using using median averaged deviation"""
#     mean = np.median(signals, axis=1, keepdims=True)
#     std = np.median(np.abs(signals - mean), axis=1, keepdims=True)
#     return ((signals - mean) < threshold * std).all(axis=1)
