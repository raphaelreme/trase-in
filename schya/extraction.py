"""Singal Extraction from videos and tracks"""

from typing import Collection, Sequence, Tuple

import cv2  # type: ignore
import numpy as np
import tqdm  # type: ignore

import byotrack


def extract_intensities_from_roi(
    video: Sequence[np.ndarray], tracks: Collection[byotrack.Track], roi_size: int
) -> np.ndarray:
    """Extract intensities at the track location from the given video

    Args:
        video (Sequence[np.ndarray]): Video
        tracks (Collection[byotrack.Track]): Tracks
        roi_size (int): Size of the square roi

    Returns:
        np.ndarray: Intensities for each track and time frame
            Shape (n_tracks, n_frames), dtype: np.float64
    """
    intensities = np.full((len(tracks), len(video)), np.nan)

    for frame_id, frame in enumerate(tqdm.tqdm(video)):
        for track_id, track in enumerate(tracks):
            point = track[frame_id]
            if point.isnan().any():
                continue

            i = int(point[0] - roi_size / 2)
            j = int(point[1] - roi_size / 2)

            intensities[track_id, frame_id] = frame[max(0, i) : i + roi_size, max(0, j) : j + roi_size].mean()
    return intensities


# Extraction of a 2d+t roi for a given track.
# def extract_temporal_roi(video, track: byotrack.Track, patch_size: int) -> np.ndarray:
#     rois = np.zeros((len(video), patch_size, patch_size))

#     for frame_id, frame in enumerate(tqdm.tqdm(video)):
#         point = track[frame_id]
#         if point.isnan().any():
#             continue

#         i = int(point[0] - patch_size / 2)
#         j = int(point[1] - patch_size / 2)

#         # Complexe slice:
#         # if x < 0 then we will only copy starting from -x in the patch
#         # if x + patch_size > W then we not copy till the end
#         i_patch_slice = slice(max(0, -i), frame.shape[0] - i)
#         j_patch_slice = slice(max(0, -j), frame.shape[1] - j)

#         #patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
#         patch = rois[frame_id]

#         if patch[i_patch_slice, j_patch_slice].shape != (patch_size, patch_size):
#             tqdm.tqdm.write(f"Partial patch for tracklet {track.identifier} at {point} ({frame_id})")

#         patch[i_patch_slice, j_patch_slice] = frame[max(0, i) : i + patch_size, max(0, j) : j + patch_size, 0]

#     return rois


class SubRoiExtractor:
    """Extract intensities of TdTomato tracks in the GCaMP video.

    Solve the alignement problem between TdTomato and GCaMP videos with single object tracking (SOT) for
    each 2d+t roi defined by a nucleus track.

    This is a very naive and simple tracking algorithm: For each nucleus, first select the closest maxima in the
    nucleus roi. Then for the following frames, we assume a small relative motion with the nucleus and
    associate with a very close maximum (if any < max_motion else do not associate and keep it at the same position)

    We look for circle of `spot_radius` pixels and we apply some gaussian blurring (with `blur_std`)
    to improve maxima localization.

    Note: Sometimes, two close nuclei can result in the same local maximum tracked.
          (This could be improved in a future version)

    Attributes:
        video (Sequence[np.ndarray]): GCaMP video
        nuclei_positions (np.ndarray): Positions of each nucleus at each frame
            Shape: (n_frames, n_tracks, 2)
        roi_size (int): Size of the squared roi (roi_size, roi_size) where to search for maxima
        max_motion (int): Max relative motion between two consecutive frames for the calcium maxima

    """

    n_local_maxima = 6
    spot_radius = 4
    blur_std = 1.0

    def __init__(
        self, video: Sequence[np.ndarray], nuclei_positions: np.ndarray, roi_size: int, max_motion: int
    ) -> None:
        assert roi_size % 2 == 1, "Please provide an odd roi size"
        assert (
            len(video) == nuclei_positions.shape[0]
        ), "The video and positions does not have the same number of frames"

        self.video = video
        self.nuclei_positions = nuclei_positions.round().astype(np.int32)
        self.roi_size = roi_size
        self.max_motion = max_motion

    def local_prior(self) -> np.ndarray:
        """Return a gaussian window for the roi, with a std of 5"""
        gauss_1d = np.exp(-0.5 * np.linspace(-(self.roi_size // 2), self.roi_size // 2, self.roi_size) ** 2 / 5**2)
        return gauss_1d[None] * gauss_1d[:, None]

    def read_frame(self, frame_id: int) -> np.ndarray:
        frame = self.video[frame_id].sum(axis=-1)  # Aggregate channels
        frame = cv2.GaussianBlur(frame, (33, 33), self.blur_std)
        return frame

    def compute(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the algorithm

        Returns:
            np.ndarray: Extracted sub roi intensities
                Shape: (n_tracks, n_frames)
            np.ndarray: Positions of the extracted intensities
                Shape: (n_frames, n_tracks, 2), dtype: np.int32
        """
        intensities = np.zeros((self.nuclei_positions.shape[1], self.nuclei_positions.shape[0]), dtype=np.float64)
        signal_positions = np.zeros_like(self.nuclei_positions, dtype=np.int32)
        previous_relative_positions = np.zeros((self.nuclei_positions.shape[1], 2), np.float32)

        prior = self.local_prior()

        for frame_id in tqdm.trange(len(self.video)):
            frame = self.read_frame(frame_id)

            for track_id, (i, j) in enumerate(self.nuclei_positions[frame_id]):
                roi = self.extract_roi(frame, i, j)

                probabilities = prior * roi  # Bias toward the center

                maxima = self.find_local_maxima(probabilities)

                if frame_id == 0:
                    # Take the first one (more intense in probability)
                    maximum = maxima[0]
                    previous_relative_positions[track_id] = maximum  # Relative pixels
                else:
                    # Let's compare with the previous relative position
                    distances = np.sum((maxima - previous_relative_positions[track_id]) ** 2, axis=1)

                    if distances.min() > self.max_motion:  # To much motion, let's stay where we are
                        maximum = previous_relative_positions[track_id]
                    else:  # Let's take the closest one
                        maximum = maxima[np.argmin(distances)]

                    previous_relative_positions[track_id] = (  # Stable update
                        0.9 * previous_relative_positions[track_id] + 0.1 * maximum
                    )  # Relative pixels

                signal_positions[frame_id, track_id] = previous_relative_positions[track_id].round().astype(
                    np.int32
                ) + (
                    i - self.roi_size // 2,
                    j - self.roi_size // 2,
                )  # In real pixels

                # Extract intensity
                # Square extraction
                # i, j = signal_positions[frame_id, track_id] - [self.spot_radius] * 2
                # intensities[track_id, frame_id] = frame[
                #     max(0, i) : i + 2 * self.spot_radius + 1, max(0, j) : j + 2 * self.spot_radius + 1
                # ].mean()

                # Circle extraction
                i = int(round(previous_relative_positions[track_id][0]))
                j = int(round(previous_relative_positions[track_id][1]))

                mask = np.zeros_like(roi)
                cv2.circle(mask, (j, i), self.spot_radius, 255, -1)
                mask[roi == 0] = 0  # When edge of video, roi is padded with 0 -> Drop the circle part out of video

                intensities[track_id, frame_id] = roi[mask > 0].mean()

        return intensities, signal_positions

    def extract_roi(self, frame: np.ndarray, i: int, j: int) -> np.ndarray:
        """Extract a roi centered on (i, j) of size self.roi_size

        Pad the roi with 0 if we are on the edge of the frame.
        """

        i -= self.roi_size // 2
        j -= self.roi_size // 2

        # Complexe slice:
        # if i < 0 then we will only copy starting from -i in the patch
        # if i + patch_size > W then we not copy till the end

        i_patch_slice = slice(max(0, -i), frame.shape[0] - i)
        j_patch_slice = slice(max(0, -j), frame.shape[1] - j)

        roi = np.zeros((self.roi_size, self.roi_size))

        if roi[i_patch_slice, j_patch_slice].shape != (self.roi_size, self.roi_size):
            tqdm.tqdm.write("Partial patch found")

        roi[i_patch_slice, j_patch_slice] = frame[max(0, i) : i + self.roi_size, max(0, j) : j + self.roi_size]

        return roi

    def find_local_maxima(self, image: np.ndarray) -> np.ndarray:
        """Find the n biggest local maxima in image"""
        image = image.copy()
        local_maxima = np.zeros((self.n_local_maxima, 2), dtype=np.int32)
        for k in range(self.n_local_maxima):
            argmax = np.argmax(image)
            local_maxima[k, 0] = argmax // image.shape[1]
            local_maxima[k, 1] = argmax % image.shape[1]

            # Delete the local maxima and its neighborhood (could be square rather than circle ?)
            cv2.circle(image, (local_maxima[k, 1], local_maxima[k, 0]), self.spot_radius, 0, -1)

        return local_maxima


def smooth(intensities: np.ndarray, window_size: int) -> np.ndarray:
    """Running averaged of the signal

    Args:
        intensities (np.ndarary): Extracted intensities
            Shape: (n_tracks, n_frames), dtype: np.float64

    Returns:
        np.ndarray: Smoothed intensities
            Shape: (n_tracks, n_frames), dtype: np.float64
    """
    smoothed = []
    n = np.convolve(np.ones(intensities.shape[1]), np.ones(window_size), mode="same")

    for intensity in intensities:
        smoothed.append(np.convolve(intensity, np.ones(window_size), mode="same") / n)

    return np.array(smoothed)


# Old Single cell extraction from Noah. Rewritten as SubRoiExtractor
# #extraction of intensity from single neuron within ROI
# def SingleCellIntensity(video, track, dimentionROI, Circle_radius, distance_threshold, display_on = False):
#     display = []
#     mask_circle_radius = Circle_radius - 0
#     neuron_points = []
#     intensities = []
#     positions_corrected = []
#     backframes = 10
#     past_thresh = 3
#     for frame in range(len(video)):
#         point = track[frame].round()

#         i = int(point[0])
#         j = int(point[1])

#         raw_image = video[frame][i - dimentionROI: i + dimentionROI + 1, j - dimentionROI: j + dimentionROI + 1, 0]
#         image = raw_image.copy()

#         #correction for ROI leaving field
#         dim_image = np.min(image.shape[0:1])
#         if dim_image < 5:
#             dim_image = 5
#         image = image[0:dim_image, 0:dim_image]

#         #Convert to grayscale
#         #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         #Copy Image pre processing to use for finding final intensity value from
#         image_out = image.copy()
#         image_display = image.copy()
#         #Create Gaussian Kernel to make points close to the centre appear brighter than points on the edges
#         kernel_gauss = gkern(image.shape[0], 0.7)
#         image = kernel_gauss*image
#         #find Centre Point of the image to use as starting point for finding neuron
#         centrept = [int(len(image[0])/2), int(len(image[1])/2)]

#         if frame == 0:
#             #find first 3 highest pixels highest pixel
#             highpt_prev = centrept
#             highpts = []
#             distances = []
#             for k in range(4):
#                 image = cv2.GaussianBlur(image, (5,5), 0)
#                 low,high,lowpt,highpt = cv2.minMaxLoc(image)
#                 highpts.append(highpt)

#                 print(high, highpt)

#                 #add brighness coefficient to couteract possibility of darker point near centre
#                 bright_coeff = (1/high)*0.5

#                 cv2.circle(image, highpt, Circle_radius, 0, -1)

#                 distances.append(bright_coeff*euclidean(highpts[k], highpt_prev))

#             print(distances)

#             #take point closest to centre
#             mindist = np.argmin(distances)
#             neuronpt = highpts[mindist]

#         elif frame > 0:
#             image = cv2.GaussianBlur(image, (5,5), 0)
#             low,high,lowpt,highpt = cv2.minMaxLoc(image)
#             distance = euclidean(highpt, highpt_prev)

#             if frame < backframes:
#                 if distance <= distance_threshold:
#                     neuronpt = highpt

#                 elif distance > distance_threshold:
#                     for k in range(3):
#                         cv2.circle(image, highpt, Circle_radius, 0, -1)
#                         image = cv2.GaussianBlur(image, (5,5), 0)
#                         low,high,lowpt,highpt = cv2.minMaxLoc(image)
#                         distance = euclidean(highpt, highpt_prev)
#                         if distance <= distance_threshold:
#                             neuronpt = highpt
#                             break

#             else:
#                 distance_past = euclidean(highpt, neuron_points[frame-backframes])

#                 if distance <= distance_threshold and distance_past <= past_thresh:
#                     neuronpt = highpt

#                 if distance > distance_threshold:
# #                     print('thresh1')
#                     for k in range(3):
#                         cv2.circle(image, highpt, Circle_radius-1, 0, -1)
#                         #image = cv2.GaussianBlur(image, (5,5), 0)
#                         low,high,lowpt,highpt = cv2.minMaxLoc(image)
#                         distance = euclidean(highpt, highpt_prev)
#                         if distance <= distance_threshold:
#                             neuronpt = highpt
#                             distance_past = euclidean(highpt, neuron_points[frame-backframes])
#                             break

#                 if distance_past > past_thresh: # and distance <= distance_threshold:
# #                     print('past')
#                     neuronpt = neuron_points[frame-2]

#         #save_values for next iteration
#         highpt_prev = neuronpt
#         neuron_points.append(neuronpt)
# #         print('npt ',neuronpt)

#         #extract fluorescence within selected circle
#         circle_mask = np.zeros((image_out.shape[0],image_out.shape[1]),dtype = np.uint8)
#         cv2.circle(circle_mask, neuronpt, mask_circle_radius, 255, -1)

#         if display_on == True:
#             image_display = cv2.GaussianBlur(image_display, (5,5), 0)
#             cv2.circle(image_display, neuronpt, Circle_radius, 255)
#             plt.imshow(image_display)
#             plt.show()
#             display.append(image_display)

#         image_out = cv2.GaussianBlur(image_out, (5,5), 0)
#         mean = np.max(cv2.mean(image_out, mask=circle_mask))
# #         print('mean: ', mean)
# #         print('max: ', np.max(image_out))
#         intensities.append(mean)
#         #positions_corrected.append(positions[neuron][frame])
#     if display_on == True:
#         return intensities, positions_corrected, neuron_points, display
#     else:
#         return  intensities, positions_corrected, neuron_points


# # intensities from all individual neurons
# def SingleCellIntensities(video, positions, dimentionROI, Circle_radius, distance_threshold):
#     """Must Run the Extract_Fluorescence function before the SingleCellIntensities function as you need
#     to use the updated 'posit_corrected' output from Extract_Fluorescence as the positions for
#     SingleCellIntensities as it has no feature to correct this itself"""

#     intensities = []
#     position_updated = []
#     all_points = []
#     for track in range(len(positions)):
#         try:
#             intensity, posits_corr, points = SingleCellIntensity(
#                 track, video, positions, dimentionROI, Circle_radius, distance_threshold
#             )
#             intensities.append(intensity)
#             position_updated.append(positions[track])
#             all_points.append(points)
#         except Exception as e:
#             print(e)
#             print("Issue in track ID: ", track)
#     return intensities, position_updated, all_points
