from typing import List, Sequence

import cv2  # type: ignore
import numpy as np
import scipy.stats  # type: ignore
import torch

import byotrack


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


class InteractiveTracksFiltering:
    """Allows manual tracks filtering by looking at the video

    Double click on a track to accept or reject it. (Green tracks are accepted, red are rejected)

    Keys:
        * w/x: Move backward/forward in the video (when paused)
        * t: Switch tracks display mode (Not displayed, Both, Valid, Invalid)

    Attributes:
        video (Sequence[np.ndarray]): Video to display tracks with
        valid_tracks (List[byotrack.Track]): Current list of valid tracks
        invalid_tracks (List[byotrack.Track]): Current list of invalid tracks

    """

    window_name = "Tracks Filtering"
    scale = 1
    valid_color = (255, 255, 255)
    invalid_color = (100, 100, 255)

    def __init__(self, video: Sequence[np.ndarray], tracks: List[byotrack.Track], is_valid: np.ndarray) -> None:
        self.video = video
        self.tracks = tracks
        self.is_valid = is_valid
        self._tracks_tensor = byotrack.Track.tensorize(self.tracks)
        self._frame_id = 0
        self._display_tracks = 1

    @property
    def valid_tracks(self) -> List[byotrack.Track]:
        return [track for i, track in enumerate(self.tracks) if self.is_valid[i]]

    @property
    def invalid_tracks(self) -> List[byotrack.Track]:
        return [track for i, track in enumerate(self.tracks) if not self.is_valid[i]]

    def run(self, frame_id=0) -> None:
        """Run the visualization

        Args:
            frame_id (int): Starting frame_id
        """
        try:
            self._frame_id = frame_id
            self._run()
        finally:
            cv2.destroyWindow(self.window_name)

    def _run(self) -> None:
        while True:
            frame = (self.video[self._frame_id] * 255).astype(np.uint8)
            frame = np.concatenate([np.zeros_like(frame), frame, np.zeros_like(frame)], axis=2)

            if self._display_tracks in (1, 2):
                for track in self.valid_tracks:
                    point = track[self._frame_id] * self.scale
                    if torch.isnan(point).any():
                        continue

                    i, j = point.round().to(torch.int).tolist()

                    cv2.circle(frame, (j, i), 5, self.valid_color)

            if self._display_tracks in (1, 3):
                for track in self.invalid_tracks:
                    point = track[self._frame_id] * self.scale
                    if torch.isnan(point).any():
                        continue

                    i, j = point.round().to(torch.int).tolist()

                    cv2.circle(frame, (j, i), 5, self.invalid_color)

            # Display the resulting frame
            cv2.imshow(self.window_name, frame)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            cv2.setWindowTitle(self.window_name, f"Frame {self._frame_id} / {len(self.video)}")

            # Handle user actions
            key = cv2.waitKey(1000 // 20) & 0xFF

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord("q"):
                break

            if key == ord("w"):  # Prev
                self._frame_id = (self._frame_id - 1) % len(self.video)

            if key == ord("x"):  # Next
                self._frame_id = (self._frame_id + 1) % len(self.video)

            if key == ord("t"):
                self._display_tracks = (self._display_tracks + 1) % 4

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _) -> None:
        """Handle mouse clicks

        Switch the selected track to accepted or rejected.

        Args:
            event (int): Opencv event type
            x (int), y (int): position of the click
            _flags (int): Opencv modifiers
            _ (Any): Additional data given by opencv
        """
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return

        # Find the closest track if any
        dists = (self._tracks_tensor[self._frame_id] - torch.tensor([[y, x]])).pow(2).sum(dim=1).sqrt()
        dists[torch.isnan(dists)] = torch.inf
        dists[dists > 5] = torch.inf
        target_id = int(dists.argmin())

        if dists[target_id] == torch.inf:  # No track close enough
            return

        # Switch track state
        self.is_valid[target_id] = 1 - self.is_valid[target_id]
        print(f"Manual switch of track {self.tracks[target_id].identifier}")
