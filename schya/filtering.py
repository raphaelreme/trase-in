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
