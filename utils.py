# this file contains the small fuctions used in the tracker
import numpy as np

def convert_bbox_to_vis(bbox):
  """
  the raw bbox [x1,y1,x2,y2] is left bottom and right top, the opencv need to be left top and right bottom
  """
  return np.array([bbox[0], bbox[3], bbox[2], bbox[1]]).reshape((1,4))


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the bottom left and x2,y2 is the top right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    res = np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    return res[0]
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


def iou(bbox1,bbox2):

  # cacluate the iou between two bounding boxes
  # bbox:[x1,y1,x2,y2], it can be a numpy array or a list
  # return the iou value
  if bbox1.ndim==1:
    bbox1 = np.expand_dims(bbox1, 0)
  if bbox2.ndim==1:
    bbox2 = np.expand_dims(bbox2, 0)
  bbox1 = np.expand_dims(bbox1, 1)
  bbox2 = np.expand_dims(bbox2, 0)

  xx1 = np.maximum(bbox2[..., 0], bbox1[..., 0])
  yy1 = np.maximum(bbox2[..., 1], bbox1[..., 1])
  xx2 = np.minimum(bbox2[..., 2], bbox1[..., 2])
  yy2 = np.minimum(bbox2[..., 3], bbox1[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])
            + (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1]) - wh)
  return (o)