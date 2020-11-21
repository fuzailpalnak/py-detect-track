class TrackerId:
    _tracker_id = -1

    @staticmethod
    def tracker_id():
        TrackerId._tracker_id += 1
        return TrackerId._tracker_id


class Tracker:
    def __init__(self, *args, **kwargs):
        pass

    def state(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def extract_position_from_state(self):
        raise NotImplementedError
