from typing import Tuple

import numpy as np
import scipy.signal  # type: ignore

from caiman.source_extraction.cnmf import deconvolution  # type: ignore


def foopsi_all(intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Uses FOOPSI algrorithm from CAIMAN package to decompose calcium signal and estimate neural spikes

    Args:
        intensities (np.ndarray): Calcium signal
            Shape: (n_tracks, n_frames)

        Returns:
            np.ndarray: Reconstructed calcium signal from spikes
                Shape: (n_tracks, n_frames)
            np.ndarray: Spikes
                Shape: (n_tracks, n_frames)
    """
    spikes = np.zeros_like(intensities)
    reconstruction = np.zeros_like(intensities)
    for i, intensity in enumerate(intensities):
        ca_foopsi, _, _, _, _, spikes_foopsi, _ = deconvolution.constrained_foopsi(intensity, p=2)
        spikes[i] = spikes_foopsi
        reconstruction[i] = ca_foopsi
    return reconstruction, spikes


def clusterize_spikes(spikes: np.ndarray, std=5.0) -> np.ndarray:
    """Clusterize spikes using a gaussian kernel

    We convolve the spikes with a gaussian kernel and extract the location of the maximums.
    """
    clustered = np.zeros_like(spikes)

    half_kernel_size = int(5 * std)
    kernel = np.exp(-0.5 * np.linspace(-half_kernel_size, half_kernel_size, 2 * half_kernel_size + 1) ** 2 / std**2)

    for i, line in enumerate(spikes):
        smoothed = np.convolve(line, kernel, mode="full")
        peaks, _ = scipy.signal.find_peaks(smoothed)  # Does not handle well borders but with full conv its fine
        clustered[i, peaks - half_kernel_size] = smoothed[peaks]

    return clustered


def binarize_std(spikes: np.ndarray, k: float) -> np.ndarray:
    """Keep spikes that deviate more than k std (on each line)"""
    spikes = spikes.copy()

    for line in spikes:
        thresh = line[line > 0].mean() + k * line[line > 0].std()
        line[line <= thresh] = 0
        line[line > thresh] = 1

    return spikes


def binarize_global_std(spikes: np.ndarray, k: float) -> np.ndarray:
    """Keep spikes that deviate more than k std (globally computed)"""
    spikes = spikes.copy()

    thresh = spikes[spikes > 0].mean() + k * spikes[spikes > 0].std()

    spikes[spikes <= thresh] = 0
    spikes[spikes > thresh] = 1

    return spikes


def binarize_ratio(spikes: np.ndarray, ratio: float) -> np.ndarray:
    """Keep spikes larger than ratio * max (for each line)"""
    spikes = spikes.copy()

    thresh = spikes.max(axis=1, keepdims=True) * ratio

    spikes[spikes <= thresh] = 0
    spikes[spikes > thresh] = 1

    return spikes


def to_raster_pos(spikes: np.ndarray) -> list:
    """Convert to raster pos (pos of spikes)

    Can be plotted with plt.eventplot
    """
    indices = np.arange(spikes.shape[1])
    pos = []
    for line in spikes:
        pos.append(indices[line > 0.0])

    return pos
