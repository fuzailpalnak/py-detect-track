from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from py_tracker.sort.tracker import KalmanTracker


class Sort:
    def __init__(self, max_age: int = 1, iou_threshold: float = 0.40):
        self.max_age: int = max_age
        self.iou_threshold: float = iou_threshold

        self.trackers: List[KalmanTracker] = list()
        self.frame_count: int = 0

    def track(self, detections: np.ndarray) -> np.ndarray:
        """

        :param detections: of format [x1, y1, x2, y2] or [x1, y1, x2, y2, score]
        :return:
        """
        self.frame_count += 1

        detections = detections[:, 0:4]

        predicted_tracks = np.zeros((len(self.trackers), 4))
        matched_detection_tracker_indices = np.empty((0, 2), dtype=int)
        unmatched_detections_ids = np.arange(len(detections))

        detections_with_active_tracks = list()

        to_del = list()
        for track_iterator, empty_state in enumerate(predicted_tracks):
            track = self.trackers[track_iterator]
            track.predict()
            state = self.scale_aspect_ratio_fmt_to_bbox_fmt(
                track.extract_position_from_state()
            )[0]
            empty_state[:] = [state[0], state[1], state[2], state[3]]
            if np.any(np.isnan(state)):
                to_del.append(track_iterator)

        predicted_tracks = np.ma.compress_rows(np.ma.masked_invalid(predicted_tracks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        if len(predicted_tracks) > 0:
            (
                matched_detection_tracker_indices,
                unmatched_detections_ids,
            ) = self.associate_detections_to_trackers(detections, predicted_tracks)

        # Update the matched tracking
        for detection_index, tracker_index in matched_detection_tracker_indices:
            self.trackers[tracker_index].update(
                self.bbox_fmt_to_scale_aspect_ratio_fmt(
                    detections[detection_index, :].reshape(4, 1)
                )
            )

        # Create and Initialize new tracker for unmatched detections with scale aspect format
        for unmatched_detection_id in unmatched_detections_ids:
            self.trackers.append(
                KalmanTracker(
                    self.bbox_fmt_to_scale_aspect_ratio_fmt(
                        detections[unmatched_detection_id, :].reshape(4, 1)
                    )
                )
            )

        # Creation and Deletion of Track Identities
        for individual_track in reversed(self.trackers):
            state = self.scale_aspect_ratio_fmt_to_bbox_fmt(
                individual_track.extract_position_from_state()
            )[0]

            # If the tracker is being updated every self.max_age frame
            if individual_track.time_since_update < self.max_age:
                detections_with_active_tracks.append(
                    np.concatenate((state, [individual_track.id + 1])).reshape(1, -1)
                )
            # Remove tracker if not updated for self.max_age frame
            elif individual_track.time_since_update > self.max_age:
                self.trackers.remove(individual_track)

        return (
            np.concatenate(detections_with_active_tracks)
            if len(detections_with_active_tracks) > 0
            else np.empty((0, 4))
        )

    def associate_detections_to_trackers(
        self, detections: np.ndarray, trackers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        """

        :param detections: machine predicted output
        :param trackers: output estimated from kalman filter
        :return:
        """
        matches = list()
        unmatched_detections = list()

        iou_matrix = self.iou_between_detection_and_tracker(detections, trackers)

        # Hungarian Algorithm
        (
            matched_detection_tracker_x,
            matched_detection_tracker_y,
        ) = linear_sum_assignment(-iou_matrix)
        matched_detection_tracker_indices = np.array(
            list(zip(matched_detection_tracker_x, matched_detection_tracker_y))
        )

        for detection_iterator, detections in enumerate(detections):
            if detection_iterator not in matched_detection_tracker_indices[:, 0]:
                unmatched_detections.append(detection_iterator)

        # For creating tracking, we consider any detection with an overlap less than min IOU to signify the existence
        # of an un tracked object. The tracker is initialised using the geometry of the bounding box with the velocity
        # set to zero
        for detection_index, tracker_index in matched_detection_tracker_indices:
            if iou_matrix[detection_index, tracker_index] < self.iou_threshold:
                unmatched_detections.append(detection_index)
            else:
                matches.append(np.array([detection_index, tracker_index]).reshape(1, 2))
        return (
            np.concatenate(matches, axis=0)
            if len(matches) > 0
            else np.empty((0, 2), dtype=int),
            np.array(unmatched_detections),
        )

    @staticmethod
    def iou_between_detection_and_tracker(
        detections: np.ndarray, trackers: np.ndarray
    ) -> np.ndarray:
        """

        :param detections:
        :param trackers:
        :return:
        """
        trackers = np.expand_dims(trackers, 0)
        detections = np.expand_dims(detections, 1)

        xx1 = np.maximum(detections[..., 0], trackers[..., 0])
        yy1 = np.maximum(detections[..., 1], trackers[..., 1])
        xx2 = np.minimum(detections[..., 2], trackers[..., 2])
        yy2 = np.minimum(detections[..., 3], trackers[..., 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (
            (detections[..., 2] - detections[..., 0])
            * (detections[..., 3] - detections[..., 1])
            + (trackers[..., 2] - trackers[..., 0])
            * (trackers[..., 3] - trackers[..., 1])
            - wh
        )
        return o

    @staticmethod
    def bbox_fmt_to_scale_aspect_ratio_fmt(bbox: np.ndarray) -> np.ndarray:
        """

        :param bbox:
        :return:
        """

        assert bbox.ndim == 2, (
            "Expected bbox to be one dimensional" "got %s",
            (bbox.ndim,),
        )

        assert bbox.shape == (4, 1), (
            "Expected bbox to have shape '[4, 1]'" "got %s",
            (bbox.shape,),
        )

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def scale_aspect_ratio_fmt_to_bbox_fmt(scale_aspect: np.ndarray) -> np.ndarray:
        """

        :param scale_aspect:
        :return:
        """
        assert scale_aspect.ndim == 2, (
            "Expected scale_aspect to be one dimensional" "got %s",
            (scale_aspect.ndim,),
        )

        assert scale_aspect.shape == (4, 1), (
            "Expected scale_aspect to have shape '[4, 1]'" "got %s",
            (scale_aspect.shape,),
        )

        w = np.sqrt(scale_aspect[2] * scale_aspect[3])
        h = scale_aspect[2] / w
        return np.array(
            [
                scale_aspect[0] - w / 2.0,
                scale_aspect[1] - h / 2.0,
                scale_aspect[0] + w / 2.0,
                scale_aspect[1] + h / 2.0,
            ]
        ).reshape((1, 4))
