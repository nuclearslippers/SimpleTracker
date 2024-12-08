# this is the main file for the project
import argparse
import glob
import numpy as np
import os

# Tracking arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Tracking arguments')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument('-d',"--detection", help="Path to detections.", type=str, default='mot15')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("-iou","--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.detection == 'mot15':
        print('Reading mot15 dataset')
        result_list = 'data/train/*/det/det.txt'
        frames = 0
        num_frame = 0
        for det_file in glob.glob(result_list):
            det_seq = np.loadtxt(det_file, delimiter=',')
            seq_name = det_file.split(os.sep)[-3]
            num_frame = det_seq[:, 0].max()
            frames = frames + num_frame

            det_seq[:, 4:6] += det_seq[:, 2:4]

    else:
        print('Reading {} dataset'.format(args.detection))
        print('this work is not done yet')

