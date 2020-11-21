from typing import List

import numpy as np

from scipy.optimize import linear_sum_assignment

from py_tracker.deepsort.deepsort_tracker import DeepSortKalmanTracker
from py_tracker.detection import Detection

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T).min(axis=0)


class DeepSortDetection(Detection):
    def __init__(self, detection, feature, confidence):
        super().__init__()
        # (x, y, w, h)
        self.detection = detection
        self.feature = feature
        self.confidence = confidence


class DeepSort:
    def __init__(self, max_cosine_distance, max_iou_distance, max_age, cascade_depth):
        self._distance_features_targets_meta = dict()
        self._max_cosine_distance = max_cosine_distance
        self._max_age = max_age
        self._cascade_depth = cascade_depth
        self._max_iou_distance = max_iou_distance

        self.tracks = list()

    def track(self, detections: List[DeepSortDetection]):

        for track in self.tracks:
            track.predict()

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]
        (
            matches,
            unmatched_tracks,
            unmatched_detections,
        ) = self.associate_detections_to_trackers(
            detections, confirmed_tracks, unconfirmed_tracks
        )

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.from_x_y_width_height_to_x_y_aspect_height(
                    detections[detection_idx].detection
                )
            )
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self.tracks.append(
                DeepSortKalmanTracker(
                    self.from_x_y_width_height_to_x_y_aspect_height(
                        detections[detection_idx].detection
                    ),
                    detections[detection_idx].feature,
                )
            )
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self._update_distance_meta(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def associate_detections_to_trackers(
        self, detections, confirmed_tracks, unconfirmed_tracks
    ):

        (
            matches_a,
            unmatched_tracks_a,
            unmatched_detections,
        ) = self.matching_cascade(detections, self.tracks, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        if len(unmatched_detections) == 0 or len(iou_track_candidates) == 0:
            matches_b = list()
            unmatched_tracks_b = iou_track_candidates
            unmatched_detections = unmatched_detections

        else:
            (
                matches_b,
                unmatched_tracks_b,
                unmatched_detections,
            ) = self.min_cost_matching(
                self.iou_cost(
                    detections, self.tracks, iou_track_candidates, unmatched_detections
                ),
                detections,
                self.tracks,
                iou_track_candidates,
                unmatched_detections,
                self._max_iou_distance,
            )
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def matching_cascade(
        self,
        detections: List[DeepSortDetection],
        tracks,
        track_indices=None,
        detection_indices=None,
    ):
        track_indices = (
            list(range(len(tracks))) if track_indices is None else track_indices
        )
        detection_indices = (
            list(range(len(detections)))
            if detection_indices is None
            else detection_indices
        )

        unmatched_detections = detection_indices
        matches = []

        for level in range(self._cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break
            track_indices_l = [
                k for k in track_indices if tracks[k].time_since_update == 1 + level
            ]

            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue

            if len(unmatched_detections) == 0 or len(track_indices_l) == 0:
                matches_l = list()
                unmatched_detections = unmatched_detections
            else:
                matches_l, _, unmatched_detections = self.min_cost_matching(
                    self.distance_gate_cost_matrix(
                        detections, tracks, track_indices_l, unmatched_detections
                    ),
                    detections,
                    tracks,
                    track_indices_l,
                    unmatched_detections,
                    self._max_cosine_distance,
                )
            matches += matches_l

        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections

    def min_cost_matching(
        self,
        cost_matrix,
        detections: List[DeepSortDetection],
        tracks,
        track_indices=None,
        detection_indices=None,
        threshold_distance=None,
    ):
        track_indices = (
            np.arange(len(tracks)) if track_indices is None else track_indices
        )
        detection_indices = (
            np.arange(len(detections))
            if detection_indices is None
            else detection_indices
        )

        matches, unmatched_tracks, unmatched_detections = list(), list(), list()

        cost_matrix[cost_matrix > self._max_cosine_distance] = threshold_distance + 1e-5
        association_x, association_y = linear_sum_assignment(cost_matrix)
        association_matrix = np.array(list(zip(association_x, association_y)))

        for col, detection_idx in enumerate(detection_indices):
            if col not in association_matrix[:, 1]:
                unmatched_detections.append(detection_idx)

        for row, track_idx in enumerate(track_indices):
            if row not in association_matrix[:, 0]:
                unmatched_tracks.append(track_idx)

        for row, col in association_matrix:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]

            if cost_matrix[row, col] > self._max_cosine_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))

        return matches, unmatched_tracks, unmatched_detections

    def distance_gate_cost_matrix(
        self,
        detections: List[DeepSortDetection],
        tracks,
        track_indices,
        detection_indices,
    ) -> np.ndarray:

        cost_matrix = self.distance_cost_matrix(
            np.array([detections[i].feature for i in detection_indices]),
            np.array([tracks[i].track_id for i in track_indices]),
        )
        cost_matrix = self.gate_cost_matrix(
            detections,
            tracks,
            track_indices,
            detection_indices,
            cost_matrix,
        )

        return cost_matrix

    def gate_cost_matrix(
        self, detections, tracks, track_indices, detection_indices, cost_matrix
    ):
        gating_threshold = chi2inv95[4]
        gated_cost = 1e5

        measurements = np.asarray(
            [
                self.from_x_y_width_height_to_x_y_aspect_height(
                    detections[i].detection
                ).reshape(4, 1)
                for i in detection_indices
            ]
        )

        for row, track_idx in enumerate(track_indices):
            track = tracks[track_idx]
            gating_distance = track.gating_distance(measurements[:, :, 0].T)
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost

        return cost_matrix

    def distance_cost_matrix(self, features, tracks):

        cost_matrix = np.zeros((len(tracks), len(features)))
        for i, target in enumerate(tracks):
            cost_matrix[i, :] = _cosine_distance(
                self._distance_features_targets_meta[target], features
            )
        return cost_matrix

    def _update_distance_meta(self, features, tracks, active_tracks):
        for feature, target in zip(features, tracks):
            self._distance_features_targets_meta.setdefault(target, []).append(feature)
        self._distance_features_targets_meta = {
            k: self._distance_features_targets_meta[k] for k in active_tracks
        }

    @staticmethod
    def iou(bbox, candidates):
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]

        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[
            np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
            np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
        ]
        br = np.c_[
            np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
            np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
        ]
        wh = np.maximum(0.0, br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        return area_intersection / (area_bbox + area_candidates - area_intersection)

    def iou_cost(self, detections, tracks, track_indices, detection_indices):
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for row, track_idx in enumerate(track_indices):
            if tracks[track_idx].time_since_update > 1:
                cost_matrix[row, :] = 1e5
                continue
            bbox = self.from_x_y_aspect_height_to_x_y_width_height(
                tracks[track_idx].extract_position_from_state()
            )
            candidates = np.asarray(
                [detections[i].detection for i in detection_indices]
            )
            cost_matrix[row, :] = 1.0 - self.iou(bbox, candidates)
        return cost_matrix

    @staticmethod
    def from_x_y_width_height_to_x_min_y_min_x_max_y_max(bbox):
        """

        :return: (min x, min y, max x, max y)
        """
        copy_bbox = bbox.copy()
        copy_bbox[2:] += copy_bbox[:2]
        return copy_bbox.reshape(1, 4)

    @staticmethod
    def from_x_y_width_height_to_x_y_aspect_height(bbox):
        copy_bbox = bbox.copy()
        copy_bbox[:2] += copy_bbox[2:] / 2
        copy_bbox[2] /= copy_bbox[3]
        return copy_bbox.reshape(4, 1)

    @staticmethod
    def from_x_y_aspect_height_to_x_y_width_height(bbox):
        copy_bbox = bbox.copy()

        copy_bbox[2] *= copy_bbox[3]
        copy_bbox[:2] -= copy_bbox[2:] / 2
        return copy_bbox
