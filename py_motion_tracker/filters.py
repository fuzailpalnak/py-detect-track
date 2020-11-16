import numpy as np


class Filter:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError


class KalmanFilter(Filter):

    """
    Implementation here does not consider estimate affected by external influence, i.e [control matrix,
    control vector]

    # https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/#mjx-eqn-kalpredictfull
    # https://arxiv.org/ftp/arxiv/papers/1204/1204.0375.pdf

    """

    def __init__(self, dim_x, dim_z):
        super().__init__(dim_x, dim_z)
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State Model
        self.x = np.zeros((self.dim_x, 1))

        # Covariance Matrix of state model
        self.P = np.eye(self.dim_x)

        # Additional Uncertainty from the environment / Motion Noise
        # process uncertainty
        self.Q = np.eye(self.dim_x)

        # Prediction Step / state transition matrix
        # Describes How the state evolves from t-1 to t without controls or noise
        self.F = np.eye(self.dim_x)

        # Map state to an observation
        # Model sensor with matrix
        # Describes how to map state 'x_t' to an observation 'z_t'
        self.H = np.zeros((self.dim_z, self.dim_x))

        # Measurement Noise
        # State Uncertainty
        self.R = np.eye(self.dim_z)

        # kalman gain - How certain am i about the observation?
        self.K = np.zeros((self.dim_x, self.dim_z))
        self.y = np.zeros((self.dim_z, 1))

        # system uncertainty
        self.S = np.zeros((self.dim_z, self.dim_z))

        self._I = np.eye(self.dim_x)

    def predict(self):

        """
        X_next(new_state) = F . X_previous
        P_next(new_uncertainty) = F.P_previous.F' + Q

        F = prediction step
        Q = additional uncertainty from environment

        :return:
        """

        self.x = np.dot(self.F, self.x)

        # motion always adds uncertainty, its makes the system uncertain
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z):

        """
        1. Calculate error between measurement and prediction
        y = z - (H.X_next)

        2. Calculate system uncertainty
        s = H.P_next.H' + R

        3. Calculate kalman gain
        k = P_next.H'.S_inv

        4. Calculate new estimate
        new_mean_estimate = X_next + k.y
        new_covariance_estimate = I-(k.H).P_next

        :param z: new measurement
        :return:

        """

        # Residual between measurement and prediction on the time step k
        self.y = z - np.dot(self.H, self.x)

        pht = np.dot(self.P, self.H.T)

        # measurement prediction covariance on the time step k
        self.S = np.dot(self.H, pht) + self.R

        # Kalman gain
        self.K = np.dot(pht, np.linalg.inv(self.S))

        # new estimated mean
        self.x = self.x + np.dot(self.K, self.y)
        # new estimated uncertainty
        self.P = np.dot(self._I - np.dot(self.K, self.H), self.P)

