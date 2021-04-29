import os
import os.path as osp
import glob
import cv2


def get_last_chkpt_path(logdir):
    """Returns the last checkpoint file name in the given log dir path."""
    checkpoints = glob.glob(osp.join(logdir, '*.pth'))
    checkpoints.sort()
    if len(checkpoints) == 0:
        return None
    return checkpoints[-1]


def spring2autumn(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img[:,:,1] = 127
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img
