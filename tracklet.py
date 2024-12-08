# tracklet means one object's trajectory, i use a class to store it
from collections import deque
from filterpy.kalman import KalmanFilter # it is not accelerated, but it needs more work to do
import numpy as np
from utils import convert_bbox_to_z, convert_x_to_bbox


class Tracklet:
    def __init__(self, id, start_frame, bbox ):
        self.id = id
        self.start_frame = start_frame
        self.bbox = bbox
        self.end_frame = -1
        self.state = 0
        # state: 0 means not born, 1 means borning, 2 means alive, 3 means dying, 4 means dead
        self.bornnum = 0
        self.deadnum = 0
        self.frames = deque(maxlen=3) # store the frames of the object, default is 3

        # kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # define constant velocity model
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(self.bbox)

    def kalman_update(self, bbox):
        self.kf.update(convert_bbox_to_z(bbox))

    def kalman_predict(self):
        """
            Advances the state vector and returns the predicted bounding box estimate.
            """
        # x[2] is the square of the bbox, so it must be positive
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.bbox = convert_x_to_bbox(self.kf.x)
