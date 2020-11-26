from typing import List, Tuple

import numpy as np

from scipy.optimize import linear_sum_assignment

from py_detect_track.track.deepsort.deepsort_tracker import DeepSortKalmanTracker
from py_detect_track.detect.detection import (
    from_x_y_width_height_to_x_y_aspect_height,
    from_x_y_aspect_height_to_x_y_width_height,
)

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


class DeepSortDetection:
    def __init__(self, detection, appearance_feature):
        super().__init__()
        # (x, y, w, h)
        self.detection = detection
        self.appearance_feature = appearance_feature


class DeepSort:
    def __init__(self, cascade_matching_threshold, iou_matching_threshold, max_age):
        self._appearance_features_targets_meta = dict()

        self._iou_matching_threshold = iou_matching_threshold
        self._cascade_matching_threshold = cascade_matching_threshold
        self._max_age = max_age

        self.trackers = list()

    def _track(self, detections: List[DeepSortDetection]):

        for track in self.trackers:
            track.predict()

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_trackers_indices = [
            i for i, t in enumerate(self.trackers) if t.is_confirmed()
        ]
        unconfirmed_trackers_indices = [
            i for i, t in enumerate(self.trackers) if not t.is_confirmed()
        ]

        (
            matches,
            unmatched_tracks,
            unmatched_detections,
        ) = self._associate_detections_to_trackers(
            detections,
            self.trackers,
            confirmed_trackers_indices,
            unconfirmed_trackers_indices,
        )

        # Update track set.
        for track_idx, detection_idx in matches:
            self.trackers[track_idx].update(
                from_x_y_width_height_to_x_y_aspect_height(
                    detections[detection_idx].detection
                ).reshape(4, 1)
            )

        # If no association is found for tracks within self._init number of frames then such tracks are deleted

        # 1. During this time, we expect a successful measurement association at each time step.
        #    Tracks that are not successfully associated to a measurement within their first three frames are deleted.

        # 2. Tracks that exceed a predefined maximum age Amax are considered to have left the scene and are deleted
        for track_idx in unmatched_tracks:
            self.trackers[track_idx].mark_missed()

        for detection_idx in unmatched_detections:
            self.trackers.append(
                DeepSortKalmanTracker(
                    from_x_y_width_height_to_x_y_aspect_height(
                        detections[detection_idx].detection
                    ).reshape(4, 1),
                    detections[detection_idx].appearance_feature,
                    self._max_age,
                )
            )
        self.trackers = [t for t in self.trackers if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.trackers if t.is_confirmed()]
        appearance_features, targets = [], []
        for track in self.trackers:
            if not track.is_confirmed():
                continue
            appearance_features += track.appearance_features
            targets += [track.track_id for _ in track.appearance_features]

        # Further, we keep a gallery Rk of the last Lk = 100 associated appearance descriptors for each track k.
        self._update_appearance_meta(
            np.asarray(appearance_features), np.asarray(targets), active_targets
        )

        return self.trackers

    def _associate_detections_to_trackers(
        self,
        detections,
        trackers,
        confirmed_trackers_indices,
        unconfirmed_trackers_indices,
    ):
        # Note that this matching cascade gives priority to tracks of smaller age, i.e., tracks that have been
        # seen more recently.
        (
            cascade_matches_detection_trackers_indices,
            cascade_unmatched_trackers_indices,
            cascade_unmatched_detections_indices,
        ) = self._matching_cascade(
            detections,
            trackers,
            confirmed_trackers_indices,
            cascade_depth=self._max_age,
            cascade_matching_threshold=self._cascade_matching_threshold,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates_indices = unconfirmed_trackers_indices + [
            k
            for k in cascade_unmatched_trackers_indices
            if trackers[k].time_since_update == 1
        ]
        cascade_unmatched_trackers_indices = [
            k
            for k in cascade_unmatched_trackers_indices
            if trackers[k].time_since_update != 1
        ]

        # In a final matching stage, we run intersection over union association as proposed in the original
        # SORT algorithm [12]
        # on the set of unconfirmed and unmatched tracks of age n = 1.
        # This helps to to account for sudden appearance changes, e.g.,
        # due to partial occlusion with static scene geometry, and to
        # increase robustness against erroneous initialization.
        (
            iou_matches_detection_trackers_indices,
            iou_unmatched_tracks_indices,
            unmatched_detections_indices,
        ) = self._matching_iou(
            detections,
            trackers,
            cascade_unmatched_detections_indices,
            iou_track_candidates_indices,
            self._iou_matching_threshold,
        )

        matches_detection_trackers = (
            cascade_matches_detection_trackers_indices
            + iou_matches_detection_trackers_indices
        )
        unmatched_tracks = list(
            set(cascade_unmatched_trackers_indices + iou_unmatched_tracks_indices)
        )
        return (
            matches_detection_trackers,
            unmatched_tracks,
            unmatched_detections_indices,
        )

    def _matching_iou(
        self,
        detections: List[DeepSortDetection],
        trackers: List[DeepSortKalmanTracker],
        unmatched_detections_indices: List,
        iou_track_candidates_indices: List,
        iou_matching_threshold: float = 0.5,
    ) -> Tuple[List, List, List]:
        """

        :param detections:
        :param trackers:
        :param unmatched_detections_indices:
        :param iou_track_candidates_indices:
        :param iou_matching_threshold:
        :return:
        """

        if (
            len(unmatched_detections_indices) == 0
            or len(iou_track_candidates_indices) == 0
        ):
            iou_matches_detection_tracker_indices = list()
            unmatched_tracker_indices = iou_track_candidates_indices
            unmatched_detections_indices = unmatched_detections_indices

        else:
            cost = self._iou_cost(
                detections,
                trackers,
                iou_track_candidates_indices,
                unmatched_detections_indices,
            )
            (
                iou_matches_detection_tracker_indices,
                unmatched_tracker_indices,
                unmatched_detections_indices,
            ) = self._min_cost_matching(
                cost,
                detections,
                trackers,
                iou_track_candidates_indices,
                unmatched_detections_indices,
                iou_matching_threshold,
            )
        return (
            iou_matches_detection_tracker_indices,
            unmatched_tracker_indices,
            unmatched_detections_indices,
        )

    def _matching_cascade(
        self,
        detections: List[DeepSortDetection],
        trackers: List[DeepSortKalmanTracker],
        tracker_indices: List = None,
        cascade_depth: int = 30,
        cascade_matching_threshold: float = 0.50,
    ) -> Tuple[List, List, List]:

        """
            Input: Track indices T = {1, . . . , N}, Detection indices D =
            {1, . . . , M}, Maximum age Amax
            1: Compute cost matrix C = [ci,j ] using Eq. 5
            2: Compute gate matrix B = [bi,j ] using Eq. 6
            3: Initialize set of matches M ← ∅
            4: Initialize set of unmatched detections U ← D
            5: for n ∈ {1, . . . , Amax} do
            6: Select tracks by age Tn ← {i ∈ T | ai = n}
            7: [xi,j ] ← min cost matching(C, Tn, U)
            8: M ← M ∪ {(i, j) | bi,j · xi,j > 0}
            9: U ← U \ {j | SUMi bi,j · xi,j > 0}
            10: end for
            11: return M, U

        :param cascade_matching_threshold:
        :param cascade_depth:
        :param detections:
        :param trackers:
        :param tracker_indices:
        :return:
        """
        tracker_indices = (
            list(range(len(trackers))) if tracker_indices is None else tracker_indices
        )
        unmatched_detections_indices = list(range(len(detections)))

        cascade_matches_detection_tracker_indices = []

        for level in range(cascade_depth):
            if len(unmatched_detections_indices) == 0:
                break

            # Select tracks by age Tn ← {i ∈ T | ai = n}
            # Therefore, we introduce a matching cascade that gives priority to more frequently seen objects to
            # encode our notion of probability spread in the association likelihood

            # Note
            # that this matching cascade gives priority to tracks of smaller
            # age, i.e., tracks that have been seen more recently.
            tracker_indices_with_small_age = [
                k for k in tracker_indices if trackers[k].time_since_update == 1 + level
            ]

            if len(tracker_indices_with_small_age) == 0:
                continue

            if (
                len(unmatched_detections_indices) == 0
                or len(tracker_indices_with_small_age) == 0
            ):
                matches = list()
                unmatched_detections_indices = unmatched_detections_indices
            else:
                cascade_matching_cost = self._compute_gate_cost(
                    detections,
                    trackers,
                    tracker_indices_with_small_age,
                    unmatched_detections_indices,
                )
                matches, _, unmatched_detections_indices = self._min_cost_matching(
                    cascade_matching_cost,
                    detections,
                    trackers,
                    tracker_indices_with_small_age,
                    unmatched_detections_indices,
                    cascade_matching_threshold,
                )
            cascade_matches_detection_tracker_indices += matches

        unmatched_trackers_indices = list(
            set(tracker_indices)
            - set(k for k, _ in cascade_matches_detection_tracker_indices)
        )
        return (
            cascade_matches_detection_tracker_indices,
            unmatched_trackers_indices,
            unmatched_detections_indices,
        )

    @staticmethod
    def _min_cost_matching(
        cost_matrix: np.ndarray,
        detections: List[DeepSortDetection],
        trackers: List[DeepSortKalmanTracker],
        track_indices: List = None,
        detection_indices: List = None,
        threshold_distance: float = None,
    ) -> Tuple[List, List, List]:
        """

        :param cost_matrix:
        :param detections:
        :param trackers:
        :param track_indices:
        :param detection_indices:
        :param threshold_distance:
        :return:
        """

        track_indices = (
            np.arange(len(trackers)) if track_indices is None else track_indices
        )
        detection_indices = (
            np.arange(len(detections))
            if detection_indices is None
            else detection_indices
        )

        (
            matches_detection_tracker,
            unmatched_trackers_indices,
            unmatched_detections_indices,
        ) = (list(), list(), list())

        cost_matrix[cost_matrix > threshold_distance] = threshold_distance + 1e-5
        association_x, association_y = linear_sum_assignment(cost_matrix)
        association_matrix = np.array(list(zip(association_x, association_y)))

        for col, detection_idx in enumerate(detection_indices):
            if col not in association_matrix[:, 1]:
                unmatched_detections_indices.append(detection_idx)

        for row, track_idx in enumerate(track_indices):
            if row not in association_matrix[:, 0]:
                unmatched_trackers_indices.append(track_idx)

        for row, col in association_matrix:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]

            if cost_matrix[row, col] > threshold_distance:
                unmatched_trackers_indices.append(track_idx)
                unmatched_detections_indices.append(detection_idx)
            else:
                matches_detection_tracker.append((track_idx, detection_idx))

        return (
            matches_detection_tracker,
            unmatched_trackers_indices,
            unmatched_detections_indices,
        )

    def _compute_gate_cost(
        self,
        detections: List[DeepSortDetection],
        trackers: List[DeepSortKalmanTracker],
        track_indices: List,
        detection_indices: List,
    ) -> np.ndarray:

        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

        # In particular, unaccounted camera motion can introduce rapid displacements in the image plane,
        # making the Mahalanobis distance a rather uninformed metric for tracking through occlusions. Therefore, we
        # integrate a second metric into the assignment problem
        # we compute an appearance descriptor
        cost_matrix = self._appearance_cost(
            np.array([detections[i].appearance_feature for i in detection_indices]),
            np.array([trackers[i].track_id for i in track_indices]),
            cost_matrix,
        )

        # To incorporate motion information we use the (squared) Mahalanobis distance between
        # predicted Kalman states and newly arrived measurements
        cost_matrix = self._mahalanobis_cost(
            detections,
            trackers,
            track_indices,
            detection_indices,
            cost_matrix,
        )

        return cost_matrix

    @staticmethod
    def _mahalanobis_cost(
        detections: List[DeepSortDetection],
        trackers: List[DeepSortKalmanTracker],
        tracker_indices: List,
        detection_indices: List,
        cost_matrix: np.ndarray,
    ) -> np.ndarray:
        """

        :param detections:
        :param trackers:
        :param tracker_indices:
        :param detection_indices:
        :param cost_matrix:
        :return:
        """

        # For our four dimensional measurement space the corresponding Mahalanobis threshold is = 9.4877
        gating_threshold = chi2inv95[4]
        gated_cost = 1e5

        measurements = np.asarray(
            [
                from_x_y_width_height_to_x_y_aspect_height(
                    detections[i].detection
                ).reshape(4, 1)
                for i in detection_indices
            ]
        )

        for row, track_idx in enumerate(tracker_indices):
            track = trackers[track_idx]

            # To incorporate motion information we use the (squared) Mahalanobis distance between
            # predicted Kalman states and newly arrived measurements
            gating_distance = track.gating_distance(measurements[:, :, 0].T)
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost

        return cost_matrix

    def _appearance_cost(
        self,
        features: np.ndarray,
        trackers_indices: np.ndarray,
        cost_matrix: np.ndarray,
    ) -> np.ndarray:
        """

        :param features:
        :param trackers_indices:
        :param cost_matrix:
        :return:
        """

        for i, target in enumerate(trackers_indices):
            # Then, our second metric measures
            # the smallest cosine distance between the i-th track and j-th
            # detection in appearance space:
            cost_matrix[i, :] = _cosine_distance(
                self._appearance_features_targets_meta[target], features
            )
        return cost_matrix

    def _update_appearance_meta(self, features, tracks, active_tracks):
        """

        :param features:
        :param tracks:
        :param active_tracks:
        :return:
        """
        # Further, we keep a gallery Rk of the last Lk = 100 associated appearance descriptors for each track k.
        for feature, target in zip(features, tracks):
            self._appearance_features_targets_meta.setdefault(target, []).append(
                feature
            )
        self._appearance_features_targets_meta = {
            k: self._appearance_features_targets_meta[k] for k in active_tracks
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

        area_intersection = np.maximum(0.0, br - tl).prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        return area_intersection / (area_bbox + area_candidates - area_intersection)

    def _iou_cost(
        self,
        detections: List[DeepSortDetection],
        trackers: List[DeepSortKalmanTracker],
        tracker_indices: List,
        detection_indices: List,
    ) -> np.ndarray:
        """

        :param detections:
        :param trackers:
        :param tracker_indices:
        :param detection_indices:
        :return:
        """

        cost_matrix = np.zeros((len(tracker_indices), len(detection_indices)))
        for row, track_idx in enumerate(tracker_indices):
            if trackers[track_idx].time_since_update > 1:
                cost_matrix[row, :] = 1e5
                continue
            bbox = from_x_y_aspect_height_to_x_y_width_height(
                trackers[track_idx].extract_position_from_state()
            )
            candidates = np.asarray(
                [detections[i].detection for i in detection_indices]
            )
            cost_matrix[row, :] = 1.0 - self.iou(bbox, candidates)
        return cost_matrix

    def track_with_detections_and_appearance_features(
        self, detections: List, appearance_features: List
    ):
        """

        :param detections: (x, y, w, h)
        :param appearance_features:
        :return:
        """
        detections = [
            DeepSortDetection(
                np.asarray(detection, dtype=np.float),
                np.asarray(appearance_feature, dtype=np.float32),
            )
            for detection, appearance_feature in zip(detections, appearance_features)
        ]
        return self._track(detections)

    def track_with_detection_object(self, detections: List[DeepSortDetection]):
        """

        :param detections:
        :return:
        """
        return self._track(detections)
