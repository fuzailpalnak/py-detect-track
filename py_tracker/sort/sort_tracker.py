import numpy as np

from py_tracker.detection import from_scale_aspect_ratio_to_x_min_y_min_x_max_y_max
from py_tracker.filters import KalmanFilter
from py_tracker.tracker import Tracker, TrackerId


class KalmanTracker(Tracker):
    def __init__(self, measurement):
        super().__init__()

        self._filter = KalmanFilter(dim_x=7, dim_z=4)

        # Filter state
        self._filter.x[:4] = measurement

        # The state of each target is modelled as: x = [u, v, s, r, u˙, v˙, s˙]
        self._filter.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self._filter.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # observation error covariance
        self._filter.R[2:, 2:] *= 10.0

        # initial velocity error covariance
        # Since the velocity is unobserved at this point the covariance of the velocity component is initialised
        # with large values, reflecting this uncertainty
        self._filter.P[4:, 4:] *= 1000.0

        # initial location error covariance
        self._filter.P *= 10.0

        # process noise
        self._filter.Q[-1, -1] *= 0.01
        self._filter.Q[4:, 4:] *= 0.01

        self.id = TrackerId.tracker_id()
        self.time_since_update = 0

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
        return from_scale_aspect_ratio_to_x_min_y_min_x_max_y_max(
            self.extract_position_from_state()
        ).reshape(4, 1)

    def update(self, measurement: np.ndarray):
        """

        :param measurement:
        :return:
        """
        self.time_since_update = 0
        self._filter.correction(measurement)

    def predict(self):
        """

        :return:
        """

        # Don't quite know the reason for this particular condition
        if (self._filter.x[6] + self._filter.x[2]) <= 0:
            self._filter.x[6] *= 0.0

        self._filter.predict()
        self.time_since_update += 1
