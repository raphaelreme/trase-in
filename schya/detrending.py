import numpy as np
import scipy.signal  # type: ignore

from sklearn.decomposition import FastICA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def detrend(signal: np.ndarray, degree: int) -> np.ndarray:
    """Detrend a single signal"""
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, degree)
    polynomial = np.polyval(coeffs, x)
    new_sequence = signal - polynomial
    return new_sequence


def detrend_all(signals: np.ndarray, degree: int) -> np.ndarray:
    """Detrend multiple signals with independent polynomial fit"""
    detrended = []
    for signal in signals:
        detrended.append(detrend(signal, degree))
    return np.array(detrended)


def high_pass_filter(signals, f_critical, order=5):
    """Detrend signals using Butterworth filter (filtering low frequencies)"""
    b, a = scipy.signal.butter(order, f_critical, btype="high")  # pylint: disable=invalid-name
    return scipy.signal.filtfilt(b, a, signals, axis=1)


def ica_decorr(intensities: np.ndarray, ctrl_intensities: np.ndarray, tolerance: float, repeats: int) -> np.ndarray:
    """ICA decorrelation

    Args:
        intensities (np.ndarary): Extracted GCaMP intensities
            Shape: (n_tracks, n_frames), dtype: np.float64
        ctrl_intensities (np.ndarary): Extracted control (TdTomato) intensities
            Shape: (n_tracks, n_frames), dtype: np.float64
        tolerance (float): FastICA tolerance
        repeats (int): Repeat multiple times and keep the best outcome

    Returns:
        np.ndarray: Intensities
            Shape (n_tracks, n_frames), dtype: np.float64
    """
    # Kept noah's code but improved the interface

    G = intensities
    R = ctrl_intensities
    # edited function from Scholz et al to account for randomness of ICA by
    # repeating multiple times and selecting best outcome
    Ynew = []
    for li in range(len(R)):
        possible_outcomes = []
        for _ in range(repeats):
            ica = FastICA(n_components=2, tol=tolerance)
            Y = np.vstack([G[li], R[li]]).T
            sclar2 = StandardScaler(copy=True, with_mean=True, with_std=True)
            Y = sclar2.fit_transform(Y)
            S = ica.fit_transform(Y)
            # order components by max correlation with red signal
            v = [np.corrcoef(s, R[li])[0, 1] for s in S.T]
            idn = np.argmin(np.abs(v))
            # check if signal needs to be inverted
            sign = np.sign(np.corrcoef(S[:, idn], G[li])[0, 1])
            signal = sign * (S[:, idn])
            possible_outcomes.append(signal)
        # best_outcome = possible outcome least correlated with the red signal (including anticorrelation)
        correlations = []
        for j in range(len(possible_outcomes)):
            correlations.append(np.corrcoef(possible_outcomes[j], R[li])[0, 1])
        min_corr_index = np.argmin(np.abs(correlations))
        best_outcome = possible_outcomes[min_corr_index]
        Ynew.append(best_outcome)
    return np.array(Ynew)
