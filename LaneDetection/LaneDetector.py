from ImageUtils import *
from Line import Line
from PerspectiveTransformer import PerspectiveTransformer


class LaneDetector:
    def __init__(self, perspective_src, perspective_dst, n_frames=1, cam_calibration=None, line_segments=10,
                 transform_offset=0):
        """
        Tracks lane lines on images or a video stream using techniques like Sobel operation, color thresholding and
        sliding histogram.

        :param perspective_src: Source coordinates for perspective transformation
        :param perspective_dst: Destination coordinates for perspective transformation
        :param n_frames: Number of frames which will be taken into account for smoothing
        :param cam_calibration: calibration object for distortion removal
        :param line_segments: Number of steps for sliding histogram and when drawing lines
        :param transform_offset: Pixel offset for perspective transformation
        """
        self.n_frames = n_frames
        self.cam_calibration = cam_calibration
        self.line_segments = line_segments
        self.image_offset = transform_offset

        self.left_line = None
        self.right_line = None
        self.center_poly = None
        self.curvature = 0.0
        self.offset = 0.0

        self.perspective_src = perspective_src
        self.perspective_dst = perspective_dst
        self.perspective_transformer = PerspectiveTransformer(perspective_src, perspective_dst)

        self.dists = []

    def __line_found(self, left, right):
        """
        Determines if pixels describing two line are plausible lane lines based on curvature and distance.
        :param left: Tuple of arrays containing the coordinates of detected pixels
        :param right: Tuple of arrays containing the coordinates of detected pixels
        :return:
        """
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            new_left = Line(y=left[0], x=left[1])
            new_right = Line(y=right[0], x=right[1])
            return are_lanes_plausible(new_left, new_right)

    def __draw_info_panel(self, img):
        """
        Draws information about the center offset and the current lane curvature onto the given image.
        :param img:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Radius of Curvature = %d(m)' % self.curvature, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1,
                    (255, 255, 255), 2)

    def __draw_lane_overlay(self, img):
        """
        Draws the predicted lane onto the image. Containing the lane area, center line and the lane lines.
        :param img:
        """
        mask = np.zeros([img.shape[0], img.shape[1]])

        # center line
        mask[:] = 0
        mask = draw_poly_arr(mask, self.center_poly, 20, 255, 5, True, tip_length=0.5)
        img[mask == 255] = (255, 75, 2)

        # lines best
        mask[:] = 0
        mask = draw_poly(mask, self.left_line.best_fit_poly, 5, 255)
        mask = draw_poly(mask, self.right_line.best_fit_poly, 5, 255)
        img[mask == 255] = (255, 200, 2)

        # lines curret
        mask[:] = 0
        mask = draw_poly(mask, self.left_line.current_fit_poly, 5, 255)
        mask = draw_poly(mask, self.right_line.current_fit_poly, 5, 255)
        img[mask == 255] = (0, 200, 0)

    def process_frame(self, frame):
        """
        Apply lane detection on a single image.
        :param frame:
        :return: annotated frame
        """
        orig_frame = np.copy(frame)

        # Apply the distortion correction to the raw image.
        if self.cam_calibration is not None:
            frame = self.cam_calibration.undistort(frame)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        frame = generate_lane_mask(frame, 400)

        # Apply a perspective transform to rectify binary image ("birds-eye view").
        frame = self.perspective_transformer.transform(frame)

        left_detected = False
        right_detected = False

        # If there have been lanes detected in the past, the algorithm will first try to
        # find new lanes along the old one. This will improve performance
        if self.left_line is not None and self.right_line is not None:
            left_x, left_y = detect_lane_along_poly(frame, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = detect_lane_along_poly(frame, self.right_line.best_fit_poly, self.line_segments)

            if self.__line_found((left_x, left_y), (right_x, right_y)):
                left_detected = True
                right_detected = True
            elif self.left_line is not None and self.right_line is not None:
                if self.__line_found((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                    left_detected = True
                if self.__line_found((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                    right_detected = True

        # If no lanes are found a histogram search will be performed
        if not left_detected:
            left_x, left_y = histogram_lane_detection(
                frame, self.line_segments, (self.image_offset, frame.shape[1] // 2), h_window=7)
            left_x, left_y = outlier_removal(left_x, left_y)
        if not right_detected:
            right_x, right_y = histogram_lane_detection(
                frame, self.line_segments, (frame.shape[1] // 2, frame.shape[1] - self.image_offset), h_window=7)
            right_x, right_y = outlier_removal(right_x, right_y)

        # Validate if detected lanes are plausible
        if self.__line_found((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.__line_found((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.__line_found((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        # Updated left lane information.
        if left_detected:
            # switch x and y since lines are almost vertical
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.n_frames, left_y, left_x)

        # Updated right lane information.
        if right_detected:
            # switch x and y since lines are almost vertical
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.n_frames, right_y, right_x)

        frame = np.stack([frame, frame, frame], 2) * 255
        if len(left_y) > 0:
            frame[left_y, left_x] = (255, 0, 0)
        if len(right_y) > 0:
            frame[right_y, right_x] = (0, 0, 255)

        return frame
