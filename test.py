import cv2

from visualize import vis_track

if __name__ == '__main__':
    root_path = './pic/ADL-Rundle-6/img1/'
    window_name = 'Image Display'
    cv2.namedWindow(window_name)
    a = [[50, 50, 200, 200, 19], [100, 100, 250, 250, 20], [150, 150, 350, 350, 21]]
    for i in range(1, 100):
        img_path = root_path + '{:06d}.jpg'.format(i)
        vis_track(window_name, img_path, a)

    cv2.destroyAllWindows()