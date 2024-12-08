# this file a basic tracking unit
# the tracker directly use the detection results, but Tracking is done on a per-frame basis
# you can consider it as an offline tracker. Hopefully, i will add a real-time tracker in the future.
import glob
import os.path
import numpy as np
from trackmanager import LifeManager
import cv2
from visualize import vis_track


def tracker():
    print('Reading mot15 dataset')
    if not os.path.exists('./output'):
        os.makedirs('./output')
    print('The result will be saved in ./output')

    result_list = 'data/train/*/det/det.txt'
    frames = 0 # total frames of all sequences
    num_frame = 0
    for det_file in glob.glob(result_list):
        # read detection sequence and its name
        det_seq = np.loadtxt(det_file, delimiter=',')
        seq_name = det_file.split(os.sep)[-3]

        # count frames
        num_frame = int( det_seq[:, 0].max() )
        frames = frames + num_frame

        print('tracking {}'.format(seq_name))

        # build the system
        track_sys = LifeManager(3,1,0.3,seq_name)

        # creating visual window
        cv2.namedWindow(seq_name, cv2.WINDOW_NORMAL)
        pic_path = './pic/{}/img1/'.format(seq_name) # modify the pic path here

        # tracking in one frame
        for i in range(num_frame):
            dets = det_seq[det_seq[:, 0] == i + 1, 2:7] # read detection in one frame
            dets[:, 2:4] += dets[:, 0:2] # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
            dets = dets[:,:-1]
            track_sys.update(dets) # update tracker

            # visualize
            display = []
            for dis in track_sys.trackers:
                temp = dis.bbox.squeeze()
                temp = np.concatenate((temp,np.array([dis.id])))
                display.append(temp)

            img_path = pic_path + '{:06d}.jpg'.format(i+1)
            vis_track(seq_name, img_path, display)
        cv2.destroyAllWindows()

    print('total frames: {} is done!'.format(frames))




if __name__ == '__main__':
    tracker()