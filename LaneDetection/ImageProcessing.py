import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

from CameraCalibration import get_camera_calibration, CameraCalibrator
from LaneDetector import LaneDetector

import cProfile

HIST_STEPS = 10
OFFSET = 200
FRAME_MEMORY = 5
SRC = np.float32([
    (300, 720),
    (580, 470),
    (730, 470),
    (1100, 720)])

DST = np.float32([
    (SRC[0][0] - 50 + OFFSET, SRC[0][1]),
    (SRC[0][0] - 50 + OFFSET, 0),
    (SRC[-1][0] - OFFSET, 0),
    (SRC[-1][0] - OFFSET, SRC[0][1])])


def process():
    cam_calibration = get_camera_calibration()
    img = imread('../test_images/test15.jpg')
    cam_calibrator = CameraCalibrator(img[:, :, 0].shape[::-1], cam_calibration)
    ld = LaneDetector(SRC, DST, n_frames=FRAME_MEMORY, cam_calibration=cam_calibrator, transform_offset=OFFSET)
    img = ld.process_frame(img)
#    plt.imshow(img, cmap='gray')
#    plt.show()


if __name__ == '__main__':
    cProfile.run('process()', sort=1)



