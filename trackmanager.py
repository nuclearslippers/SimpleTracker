# this file manage the life of the tracklets
from numpy.array_api import float32

from utils import iou
import lap
from tracklet import Tracklet
import numpy as np
import os

class LifeManager(object):
    def __init__(self, max_age, min_hits, iou_threshold, name='mot15'):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.tracker_standby = []
        self.frame_count = 1
        self.id_count = 1 # give id to every tracklet
        self.name = name
        self.initialize()

    def initialize(self):
        if os.path.exists('./output/{}.txt'.format(self.name)):
            os.remove('./output/{}.txt'.format(self.name))

    # this method must be called every frame, even there's no object detected
    def update(self, dets):
        # predict the tracklets, using kalman filter
        for t in self.trackers:
            t.kalman_predict()
        for t in self.tracker_standby:
            t.kalman_predict()

        # data association
        self.data_association(dets)

        # remove tracklets meet the deadnum
        self.trackers = list(filter(lambda t: t.deadnum < self.max_age, self.trackers))

        # create new trackers from standby
        for t in self.tracker_standby:
            if t.bornnum > self.min_hits:
                self.trackers.append(t)
                t.state = 2 # alive
        self.tracker_standby = list(filter(lambda t: t.bornnum <= self.min_hits, self.tracker_standby))

        # log the tracklets
        self.log_trackers()

        # move a frame on
        self.frame_count += 1

    def log_trackers(self):
        # log format: frame_id, tracklet_id, x1, y1, x2, y2
        with open('./output/{}.txt'.format(self.name), 'a') as f:
            for t in self.trackers:
                f.write(str(self.frame_count) + ',' + str(t.id) + ',' + str(t.bbox[0]) + ',' + str(t.bbox[1]) + ',' + str(t.bbox[2]) + ',' + str(t.bbox[3]) + '\n')
        f.close()


    def data_association(self,detection):
        # data association
        # return 3 list: tracklet, new_tracklet, unmatched_tracklet
        # the column is detection, the row is tracklet

        tracklets = []
        matched_indices = []
        for t in self.trackers:
            tracklets.append(t.bbox)
        for t in self.tracker_standby:
            tracklets.append(t.bbox)
        detection = np.array(detection)
        if len(tracklets) == 0: # no tracklet
            for i in detection:
                temp_track = Tracklet(self.id_count, self.frame_count, i)
                temp_track.state = 1  # borning
                temp_track.bornnum += 1
                self.tracker_standby.append(temp_track)
                self.id_count += 1
        else:
            tracklets = np.vstack(tracklets)
            # len(detection)==0, it can be dealt with numpy itself
            iou_batch = iou(detection, tracklets)
            if min(iou_batch.shape) == 0: # the detection is not empty
                matched_indices = np.empty((0,2))
            else:
                _, x, y = lap.lapjv(1 - iou_batch, extend_cost=True)
                matched_indices = np.array([[y[i],i] for i in x if i >= 0])

            # filter out matched with low IOU
            matches = [] # matches will be returned
            for m in matched_indices:
                if iou(detection[m[0]], tracklets[m[1]]) < self.iou_threshold:
                    pass
                else:
                    matches.append(m)
            if len(matches) == 0:
                matches = np.empty((0, 2),dtype=int)

            matches = np.array(matches)
            unmatched_detections = []
            unmatched_tracklets = []
            for i, t in enumerate(tracklets):
                if i not in matches[:, 1]:
                    unmatched_tracklets.append(i)
            for i, d in enumerate(detection):
                if i not in matches[:, 0]:
                    unmatched_detections.append(d)

            # update matched tracklets
            num_t = len(self.trackers)
            for m in matches:
                if m[1] > num_t-1:
                    self.tracker_standby[m[1] - num_t].kalman_update( detection[m[0]] )
                    self.tracker_standby[m[1] - num_t].bornnum += 1
                else:
                    self.trackers[m[1]].kalman_update( detection[m[0]] )
                    if self.trackers[m[1]].state == 3:
                        self.trackers[m[1]].deadnum = 0 # reset the deadnum
                        self.trackers[m[1]].state = 2 # alive

            # deal with unmatched tracklets
            for um in unmatched_tracklets:
                if um > num_t-1:
                    self.tracker_standby[um-num_t].state = 4 # dead
                else:
                    self.trackers[um].kalman_update(self.trackers[um].bbox) # using itself as measurement
                    self.trackers[um].state = 3 # dying
                    self.trackers[um].deadnum += 1

            # for unmatched detection, create new tracklet_standby
            for i in unmatched_detections:
                temp_track = Tracklet(self.id_count, self.frame_count, i)
                temp_track.state = 1 # borning
                temp_track.bornnum += 1
                self.tracker_standby.append(temp_track)
                self.id_count += 1

