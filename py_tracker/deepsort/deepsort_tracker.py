import numpy as np
import scipy

from py_tracker.detection import (
    from_x_y_aspect_height_to_x_y_width_height,
    from_x_y_width_height_to_x_min_y_min_x_max_y_max,
)
from py_tracker.filters import KalmanFilter
from py_tracker.tracker import Tracker, TrackerId


class DeepSortTrackerState:
    """
    Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    def __init__(
        self, tentative: bool = False, confirmed: bool = False, deleted: bool = False
    ):
        self._tentative = tentative
        self._confirmed = confirmed
        self._deleted = deleted

    @property
    def tentative(self):
        return self._tentative

    @tentative.setter
    def tentative(self, value):
        self._tentative = value

    @property
    def confirmed(self):
        return self._confirmed

    @confirmed.setter
    def confirmed(self, value):
        self._confirmed = value

    @property
    def deleted(self):
        return self._deleted

    @deleted.setter
    def deleted(self, value):
        self._deleted = value

    @classmethod
    def tentative_tracker_state(cls):
        return DeepSortTrackerState(tentative=True)

    @classmethod
    def confirmed_tracker_state(cls):
        return DeepSortTrackerState(confirmed=True)

    @classmethod
    def deleted_tracker_state(cls):
        return DeepSortTrackerState(deleted=True)


class DeepSortKalmanTracker(Tracker):
    def __init__(self, measurement, feature, max_age: int = 30):
        super().__init__()

        self._hits = 1

        # These new tracks are classified as tentative during their first three frames, _n_init = number of frames
        # until which tracks will be considered tentative
        self._n_init = 3

        # Tracks that exceed a predefined maximum age Amax are considered to have left the scene and are
        # deleted from the track set
        self._max_age = max_age

        self.features = list()
        if feature is not None:
            self.features.append(feature)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        self._filter = KalmanFilter(dim_x=8, dim_z=4)

        # Filter state
        self._filter.x[:4] = measurement

        # The state of each target is modelled as: x = (u, v, γ, h, x˙, y˙, γ ˙, h˙)
        self._filter.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self._filter.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        std = [
            2 * self._std_weight_position * float(measurement[3]),
            2 * self._std_weight_position * float(measurement[3]),
            1e-2,
            2 * self._std_weight_position * float(measurement[3]),
            10 * self._std_weight_velocity * float(measurement[3]),
            10 * self._std_weight_velocity * float(measurement[3]),
            1e-5,
            10 * self._std_weight_velocity * float(measurement[3]),
        ]

        # initial position, velocity error covariance
        self._filter.P = np.diag(np.square(std))

        self.track_id = TrackerId.tracker_id()
        self.time_since_update = 0

        # New track hypotheses are initiated for each detection that cannot be associated to an existing track
        # These new tracks are classified as tentative during their first three frames
        self._tracker_state = None
        self.mark_state_tentative()

    def extract_position_from_state(self):
        """
        Extract position from the state
        :return:
        """
        return self.state()[0:4, :]

    def state(self):
        """

        :return: state
        """
        return self._filter.x

    def bbox(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return from_x_y_width_height_to_x_min_y_min_x_max_y_max(
            from_x_y_aspect_height_to_x_y_width_height(
                self.extract_position_from_state()
            )
        ).reshape(4, 1)

    def update(self, measurement: np.ndarray):
        """

        :param measurement:
        :return:
        """
        self.time_since_update = 0

        std = [
            self._std_weight_position * float(self._filter.x[3]),
            self._std_weight_position * float(self._filter.x[3]),
            1e-1,
            self._std_weight_position * float(self._filter.x[3]),
        ]

        # observation error covariance
        self._filter.R = np.diag(np.square(std))

        self._filter.correction(measurement)

        self._hits += 1

        # New track hypotheses are initiated for each detection that cannot be associated to an existing track
        # These new tracks are classified as tentative during their first three frames
        if self.is_tentative() and self._hits >= self._n_init:
            self.mark_state_confirmed()

    def predict(self):
        """

        :return:
        """

        std_pos = [
            self._std_weight_position * float(self._filter.x[3]),
            self._std_weight_position * float(self._filter.x[3]),
            1e-2,
            self._std_weight_position * float(self._filter.x[3]),
        ]
        std_vel = [
            self._std_weight_velocity * float(self._filter.x[3]),
            self._std_weight_velocity * float(self._filter.x[3]),
            1e-5,
            self._std_weight_velocity * float(self._filter.x[3]),
        ]

        # process noise
        self._filter.Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        self._filter.predict()
        self.time_since_update += 1

    def gating_distance(self, measurements):
        """
        https://stackoverflow.com/questions/31807843/vectorizing-code-to-calculate-squared-mahalanobis-distiance
        https://stats.stackexchange.com/questions/147210/efficient-fast-mahalanobis-distance-computation

        To incorporate motion information we use the (squared) Mahalanobis distance between predicted Kalman states and
        newly arrived measurements:

        :param measurements:
        :return:
        """
        std = [
            self._std_weight_position * float(self._filter.x[3]),
            self._std_weight_position * float(self._filter.x[3]),
            1e-1,
            self._std_weight_position * float(self._filter.x[3]),
        ]

        # we denote the projection of the i-th track distribution into measurement space by (yi,Si)
        # Just consider position
        mean = measurements - np.dot(self._filter.H, self._filter.x)
        covariance = np.dot(
            self._filter.H, np.dot(self._filter.P, self._filter.H.T)
        ) + np.diag(np.square(std))

        distance = scipy.linalg.solve_triangular(
            np.linalg.cholesky(covariance),
            mean,
            lower=True,
            check_finite=False,
            overwrite_b=True,
        )

        return np.sum(distance * distance, axis=0)

    def mark_state_tentative(self):
        self._tracker_state = DeepSortTrackerState.tentative_tracker_state()

    def mark_state_confirmed(self):
        self._tracker_state = DeepSortTrackerState.confirmed_tracker_state()

    def mark_state_deleted(self):
        self._tracker_state = DeepSortTrackerState.deleted_tracker_state()

    def mark_missed(self):
        # Tracks that are not successfully associated to a measurement within their first three frames are deleted
        if self.is_tentative():
            self.mark_state_deleted()
        # Tracks that exceed a predefined maximum age Amax are considered to have left the scene and
        # are deleted from the track set
        elif self.time_since_update > self._max_age:
            self.mark_state_deleted()

    def is_tentative(self):
        return self._tracker_state.tentative

    def is_confirmed(self):
        return self._tracker_state.confirmed

    def is_deleted(self):
        return self._tracker_state.deleted
