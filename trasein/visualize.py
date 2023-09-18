from typing import Sequence, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import torch

import byotrack


class TwoColorInteractiveVisualizer:
    """Interactive visualization with opencv

    Keys:
        * `space`: Pause/Unpause the video
        * w/x: Move backward/forward in the video (when paused)
        * d: Switch detections display mode (Not displayed, Mask, Segmentation) if available
        * t: Switch tracks display mode (Not displayed, Both, TdTomato, GCaMP) if available
        * v: Switch video display mode (Not displayed, Both, TdTomato, GCaMP) if available

    Attributes:
        videos (Optional[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]]): Optional videos (TdTomato, GCaMP)
            to display. Should be normalized in [0, 1]
            Default: None
        detections_sequence (Collection[byotrack.Detections]): Optional detections to display
            Default: () (no detections)
        tracks (Collection[byotrack.Tracks]): Optional tracks to display
            Default: () (no tracks)
        frame_shape (Tuple[int, int]): Shape of frames
        n_frames (int): Number of frames
    """

    window_name = "ByoTrack ViZ"
    scale = 1

    def __init__(
        self,
        videos: Optional[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = None,
        detections_sequence=(),
        tracks=(),
        calcium_tracks=(),
    ) -> None:
        assert videos is not None or detections_sequence or tracks or calcium_tracks, "No data to display"

        self.videos = videos
        self.detections_sequence = detections_sequence
        self.tracks = tracks
        self.calcium_tracks = calcium_tracks

        self.frame_shape = self._get_frame_shape(
            self.videos[0] if self.videos else None, self.detections_sequence, self.tracks
        )
        self.n_frames = self._get_n_frames(
            self.videos[0] if self.videos else None, self.detections_sequence, self.tracks
        )

        self._frame_id = 0
        self._display_video = int(videos is not None)
        self._display_detections = int(bool(detections_sequence))
        self._display_tracks = int(bool(tracks))
        self._running = False

    def run(self, frame_id=0, fps=20) -> None:
        """Run the visualization

        Args:
            frame_id (int): Starting frame_id
            fps (int): Frame rate
        """
        try:
            self._frame_id = frame_id
            self._run(fps)
        finally:
            cv2.destroyWindow(self.window_name)

    def _run(self, fps=20) -> None:
        frame_to_detections = {detections.frame_id: detections for detections in self.detections_sequence}

        while True:
            self._frame_id += self._running
            frame = np.zeros((*self.frame_shape, 3), dtype=np.uint8)

            if self._display_video and self.videos is not None:
                red_frame = np.zeros(self.frame_shape, dtype=np.uint8)[..., None]
                if self._display_video in (1, 2):
                    red_frame = (self.videos[0][self._frame_id] * 255).astype(np.uint8)
                    assert red_frame.shape[2] == 1

                green_frame = np.zeros(self.frame_shape, dtype=np.uint8)[..., None]
                if self._display_video in (1, 3):
                    green_frame = (self.videos[1][self._frame_id] * 255).astype(np.uint8)
                    assert green_frame.shape[2] == 1

                frame = np.concatenate([np.zeros_like(red_frame), green_frame, red_frame], axis=2)

            if self._display_detections and self._frame_id in frame_to_detections:
                segmentation: np.ndarray = frame_to_detections[self._frame_id].segmentation.clone().numpy()
                if self._display_detections == 1:  # Display segmentation as mask
                    segmentation = segmentation != 0
                    segmentation = segmentation.astype(np.uint8)[..., None] * 255
                else:
                    segmentation = (segmentation % 206) + 50
                    segmentation[frame_to_detections[self._frame_id].segmentation == 0] = 0
                    segmentation = segmentation.astype(np.uint8)[..., None]

                segmentation = np.concatenate(
                    [segmentation, np.zeros_like(segmentation), np.zeros_like(segmentation)], axis=2
                )

                if self._display_video:
                    frame = frame / 2 + segmentation / 2
                    frame = frame.astype(np.uint8)
                else:
                    frame = segmentation

            # frame = cv2.GaussianBlur(frame, (33, 33), 1.0)
            frame = cv2.resize(frame, dsize=(frame.shape[1] * self.scale, frame.shape[0] * self.scale))

            if self._display_tracks in (1, 2):
                for track in self.tracks:
                    point = track[self._frame_id] * self.scale
                    if torch.isnan(point).any():
                        continue

                    i, j = point.round().to(torch.int).tolist()

                    color = (100, 100, 255)

                    cv2.circle(frame, (j, i), 5, color)
                    cv2.putText(
                        frame, str(track.identifier % 100), (j + 4, i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color
                    )

            if self._display_tracks in (1, 3):
                for track in self.calcium_tracks:
                    point = track[self._frame_id] * self.scale
                    if torch.isnan(point).any():
                        continue

                    i, j = point.round().to(torch.int).tolist()

                    color = (100, 255, 100)

                    cv2.circle(frame, (j, i), 5, color)
                    cv2.putText(
                        frame, str(track.identifier % 100), (j + 4, i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color
                    )

            # Display the resulting frame
            cv2.imshow(self.window_name, frame)
            cv2.setWindowTitle(self.window_name, f"Frame {self._frame_id} / {self.n_frames}")

            # Handle user actions
            key = cv2.waitKey(1000 // fps) & 0xFF
            if self.handle_actions(key):
                break

    def handle_actions(self, key: int) -> bool:
        """Handle inputs from user

        Return True to quit

        Args:
            key (int): Key input from user

        Returns:
            bool: True to quit visualization
        """
        if key == ord("q"):
            return True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        if key == ord(" "):
            self._running = not self._running

        if not self._running and key == ord("w"):  # Prev
            self._frame_id = (self._frame_id - 1) % self.n_frames

        if not self._running and key == ord("x"):  # Next
            self._frame_id = (self._frame_id + 1) % self.n_frames

        if key == ord("d"):
            self._display_detections = (self._display_detections + 1) % 3

        if key == ord("t"):
            self._display_tracks = (self._display_tracks + 1) % 4

        if key == ord("v"):
            self._display_video = (self._display_video + 1) % 4

        return False

    @staticmethod
    def _get_frame_shape(
        video=None,
        detections_sequence=(),
        tracks=(),
    ):
        frame_shape = (1, 1)
        if video is not None:
            frame_shape = np.broadcast_shapes(frame_shape, video[0].shape[:2])  # type: ignore
        if detections_sequence:
            detections = next(iter(detections_sequence))
            frame_shape = np.broadcast_shapes(frame_shape, detections.shape)  # type: ignore
        if tracks:
            track_tensor = byotrack.Track.tensorize(tracks)
            positions = track_tensor[~torch.isnan(track_tensor).any(dim=2)]
            minimum: torch.Tensor = positions.min(dim=0).values.round()
            maximum: torch.Tensor = positions.max(dim=0).values.round()

            if frame_shape == (1, 1):
                size = (maximum + 20).to(torch.int).tolist()  # Add some margin
                frame_shape = (size[0], size[1])

            if (minimum < 0).any() or maximum[0] >= frame_shape[0] or maximum[1] >= frame_shape[1]:
                print("Some tracks are out of frame")

        return frame_shape

    @staticmethod
    def _get_n_frames(
        video=None,
        detections_sequence=(),
        tracks=(),
    ) -> int:
        if video is not None:
            return len(video)

        n_frames = 1
        if tracks:
            n_frames = max(track.start + len(track) for track in tracks)

        if detections_sequence:
            n_frames = max(
                n_frames,
                1 + max(detections.frame_id if detections.frame_id else 0 for detections in detections_sequence),
            )

        return n_frames
