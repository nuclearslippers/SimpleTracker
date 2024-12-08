# this file is for visualizing the results of the tracking
import cv2
import numpy as np
from utils import convert_bbox_to_vis

def color_map(num):
    num = int(num)
    colors = [
        (0, 0, 0),  # 黑色
        (255, 255, 255),  # 白色
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (0, 255, 255),  # 青色
        (255, 0, 255),  # 品红色
        (128, 0, 0),  # 深红色
        (0, 128, 0),  # 深绿色
        (0, 0, 128),  # 深蓝色
        (128, 128, 0),  # 橄榄色
        (128, 0, 128),  # 紫罗兰色
        (0, 128, 128),  # 海蓝色
        (192, 192, 192),  # 银白色
        (128, 128, 128),  # 灰色
        (64, 0, 0),  # 马鞍棕色
        (0, 64, 0),  # 深橄榄绿色
        (64, 0, 64),  # 靛蓝色
        (64, 64, 64)  # 暗灰色
    ]
    while num>=len(colors):
        num = num % 20
    return colors[num]

def vis_track(window_name, img_path, tracks):
    img = cv2.imread(img_path)
    for i in range(len(tracks)):
        cv2.rectangle(img, (int(tracks[i][0]), int(tracks[i][1])), (int(tracks[i][2]), int(tracks[i][3])), color_map(tracks[i][4]), 3)
    cv2.imshow(window_name, img)
    cv2.waitKey(10)


if __name__ == '__main__':
    vis_path = './output/ADL-Rundle-6.txt'
    dets = []
    with open(vis_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            det = line.split(',')
            raw = [float(x) for x in det[:6]]
            # dets.append( [int(raw[0]), raw[2], raw[3], raw[2]+raw[4], raw[3]+raw[5], raw[1]] ) # x1,y1,w,h
            dets.append([int(raw[0]), raw[2], raw[3], raw[4], raw[5], raw[1]])  # x1,y1,x2,y2


    dets = np.array(dets)
    # temp = dets[:,0].astype(int)
    # dets[:,0] = temp
    cv2.namedWindow('test', cv2.WINDOW_AUTOSIZE)
    for i in range(179):
        track = dets[dets[:,0]==i+1, 1:]
        vis_track('test', './pic/ADL-Rundle-6/img1/{:06d}.jpg'.format(i+1), track)
    cv2.destroyAllWindows()
