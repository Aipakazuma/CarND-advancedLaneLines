import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.misc import imread

from CameraCalibration import get_camera_calibration, CameraCalibrator
from LaneDetector import LaneDetector

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


if __name__ == '__main__':
    cam_calibration = get_camera_calibration()
    cam_calibrator = CameraCalibrator((1280, 720), cam_calibration)

    ld = LaneDetector(SRC, DST, n_frames=FRAME_MEMORY, cam_calibration=cam_calibrator, transform_offset=OFFSET)

    project_output = '../project_video_ann.mp4'
    clip1 = VideoFileClip("../project_video.mp4")
    project_clip = clip1.fl_image(ld.process_frame)
    project_clip.write_videofile(project_output, audio=False)