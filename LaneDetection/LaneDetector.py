from ImageUtils import *
from Line import Line, calc_curvature
from PerspectiveTransformer import PerspectiveTransformer


class LaneDetector:
    def __init__(self, perspective_src, perspective_dst, n_frames=1, cam_calibration=None, line_segments=10, transform_offset=0):
        # Frame memory
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

        self.n_frames_processed = 0
        self.dists = []

    def __line_found(self, left, right):
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            new_left = Line(y=left[0], x=left[1])
            new_right = Line(y=right[0], x=right[1])
            return are_lanes_plausible(new_left, new_right)

    def __draw_info_panel(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Radius of Curvature = %d(m)' % self.curvature, (50, 50), font, 1, (1., 1., 1.), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1,
                    (1., 1., 1.), 2)

    def __draw_lane_overlay(self, img):
        overlay = np.zeros([*img.shape])
        mask = np.zeros([img.shape[0], img.shape[1]])

        lane_area = calculate_lane_area((self.left_line, self.right_line), img.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        mask = self.perspective_transformer.inverse_transform(mask)

        overlay[mask == 1] = (1., 0.5, 0)
        selection = (overlay != 0)
        img[selection] = img[selection] * 0.3 + overlay[selection] * 0.7

        center_line = np.zeros([img.shape[0], img.shape[1]])
        center_line = draw_poly_arr(center_line, self.center_poly, 20, 1., 5, True, tip_length=0.5)
        center_line = self.perspective_transformer.inverse_transform(center_line)
        img[center_line == 1.] = (1., 0.3, 0.01)

        lines_best = np.zeros([img.shape[0], img.shape[1]])
        lines_best = draw_poly(lines_best, self.left_line.best_fit_poly, 5, 1.)
        lines_best = draw_poly(lines_best, self.right_line.best_fit_poly, 5, 1.)
        lines_best = self.perspective_transformer.inverse_transform(lines_best)
        img[lines_best == 1.] = (1., 0.8, 0.01)

    def __detect_line(self, img, line, window):
        line_detected = False
        if self.n_frames_processed != 0:
            x, y = detect_lane_along_poly(img, line.best_fit_poly, self.line_segments)

            if self.__line_found((x, y), (line.ally, line.allx)):
                line_detected = True

        if not line_detected:
            x, y = histogram_lane_detection(img, self.line_segments, window, h_window=21, v_window=3)

        if self.__line_found((x, y), (line.ally, line.allx)):
            line_detected = True

        if line_detected:
            return x, y
        else:
            return None

    def process_frame(self, frame):
        frame = img_to_float(frame)
        orig_frame = np.copy(frame)

        # Apply the distortion correction to the raw image.
        if self.cam_calibration is not None:
            frame = self.cam_calibration.undistort(frame)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        frame = gaussian_blur(frame, kernel_size=5)
        frame = generate_lane_mask(frame)

        # Apply a perspective transform to rectify binary image ("birds-eye view").
        frame = self.perspective_transformer.transform(frame)

        # mask outside are of persp trans
        frame[:, frame.shape[1] - self.image_offset:] = 0
        frame[:, :self.image_offset] = 0

        left_detected = False
        right_detected = False
        if self.n_frames_processed != 0 and self.left_line is not None and self.right_line is not None:
            left_x, left_y = detect_lane_along_poly(frame, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = detect_lane_along_poly(frame, self.right_line.best_fit_poly, self.line_segments)
            left_x, left_y = outlier_removal(left_x, left_y)
            right_x, right_y = outlier_removal(right_x, right_y)

            if self.__line_found((left_x, left_y), (right_x, right_y)):
                left_detected = True
                right_detected = True
            elif self.left_line is not None and self.right_line is not None:
                if self.__line_found((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                    left_detected = True
                if self.__line_found((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                    right_detected = True

        if not left_detected:
            left_x, left_y = histogram_lane_detection(
                frame, self.line_segments, (self.image_offset, frame.shape[1] // 2), h_window=21, v_window=3)
            left_x, left_y = outlier_removal(left_x, left_y)
        if not right_detected:
            right_x, right_y = histogram_lane_detection(
                frame, self.line_segments, (frame.shape[1] // 2, frame.shape[1] - self.image_offset), h_window=21, v_window=3)
            right_x, right_y = outlier_removal(right_x, right_y)

        if self.__line_found((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.__line_found((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.__line_found((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        if left_detected:
            # switch x and y since lines are almost vertical
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.n_frames, left_y, left_x)

        if right_detected:
            # switch x and y since lines are almost vertical
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.n_frames, right_y, right_x)

        if self.left_line is not None and self.right_line is not None:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = calc_curvature(self.center_poly)
            self.offset = (frame.shape[1] / 2 - self.center_poly(719)) * 3.7 / 700

            self.__draw_lane_overlay(orig_frame)
            self.__draw_info_panel(orig_frame)

        self.n_frames_processed += 1
        return img_to_int(orig_frame)

        # plt.scatter(left_x, left_y, color='b')
        # plt.scatter(right_x, right_y, color='r')
        #
        # return frame
