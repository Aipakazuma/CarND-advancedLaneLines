import cv2
import numpy as np
from scipy import signal


def img_to_float(img):
    img = img.astype(np.float32)
    img /= 255
    return img


def img_to_int(img):
    return (img * 255).astype(np.uint8)


def abs_sobel_thresh(img_ch, orient='x', sobel_kernel=3, thresh=(0., 1.)):
    if orient == 'x':
        axis = (1, 0)
    elif orient == 'y':
        axis = (0, 1)
    else:
        raise ValueError('orient has to be "x" or "y" not "%s"' % orient)

    sobel = cv2.Sobel(img_ch, cv2.CV_64F, *axis, ksize=sobel_kernel)
    abs_s = np.absolute(sobel)

    binary_output = np.zeros_like(abs_s)
    binary_output[(abs_s > thresh[0]) & (abs_s < thresh[1])] = 1

    return binary_output


def mag_thresh(img_ch, sobel_kernel=3, thresh=(0., 1.)):
    sobel_x = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    mag_binary = np.zeros_like(abs_grad_mag)
    mag_binary[(abs_grad_mag > thresh[0]) & (abs_grad_mag < thresh[1])] = 1.

    return mag_binary


def dir_threshold(img_ch, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    with np.errstate(divide='ignore', invalid='ignore'):
        abs_grad_dir = np.absolute(np.arctan(sobely / sobelx))
        max_val = np.nanmax(abs_grad_dir)
        abs_grad_dir[np.isnan(abs_grad_dir)] = max_val
        dir_binary = np.zeros_like(abs_grad_dir)
        abs_grad_dir = abs_grad_dir / max_val
        dir_binary[(abs_grad_dir > thresh[0]) & (abs_grad_dir < thresh[1])] = 1

    return dir_binary


def equalize_luminance(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    l = img[:, :, 2]
    blur = gaussian_blur(l, 35)
    blur = 1. - blur
    img[:, :, 2] = (l + blur) / 2
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img


def colorspace_channel_select(img, cs, ch, thresh=(0, 255)):
    cs_img = cv2.cvtColor(img, cs)
    s_ch = cs_img[:, :, ch]
    retval, ch_binary = cv2.threshold(s_ch.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)

    return ch_binary


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def adjust_gamma(image, gamma=1.0):
    was_float = False
    if image.dtype == np.float32:
        was_float = True
        image = img_to_int(image)

    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype(np.uint8)

    # apply gamma correction using the lookup table
    image = cv2.LUT(image, table)

    if was_float:
        image = img_to_float(image)

    return image


def extract_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (30., 0.3, 0.3), (70., 1., 1.))

    return mask


def extract_white(img):
    thresh = 0.80 - (1. - np.percentile(img, 99.9))
    mask = cv2.inRange(img, (thresh, thresh, thresh), (1., 1., 1.))

    return mask


def extract_dark(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (0, 0, 0.), (360., 0.6, 0.5))

    return mask


def generate_lane_mask(img):
    # img = equalize_luminance(img)

    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    gray = np.mean(yuv[:, :, 1:2], 2)

    x = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0.05, .4))
    y = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(0.05, .2))

    x_mul_y = x * y
    x_mul_y = gaussian_blur(x_mul_y, 11)

    dir = dir_threshold(gray, sobel_kernel=3, thresh=(0.4, 0.8))
    dir = gaussian_blur(dir, 11)
    mag = mag_thresh(gray, 3, thresh=(0.05, 0.5))
    mag = gaussian_blur(mag, 11)

    dir_mul_mag = dir * mag

    comb = x_mul_y + dir_mul_mag
    comb = comb / comb.max()
    comb = cv2.inRange(comb, 0.3, 1.)

    wht = extract_white(img)
    ylw = extract_yellow(img)
    dark = extract_dark(img)

    img = np.dstack([comb, wht, ylw])
    img = np.max(img, axis=2)
    img[dark == 255] = 0

    return img


def histogram_lane_detection(img, steps, search_window, h_window, v_window):
    # TODO add dynamic search window. Grows if detected peaks get close to border.
    all_x = []
    all_y = []
    masked_img = img[:, search_window[0]:search_window[1]]
    histograms = np.zeros((steps, masked_img.shape[1]))
    pixels_per_step = img.shape[0] // steps

    for i in range(steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histograms[i] = histogram

    histograms = histogram_smoothing(histograms, window=v_window)

    for i, histogram in enumerate(histograms):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 50)))

        highest_peak = detect_highest_peak_in_area(histogram_smooth, peaks, threshold=1000)
        if highest_peak is not None:
            center = (start + end) // 2
            x, y = get_pixel_in_window(masked_img, highest_peak, center, pixels_per_step)

            all_x.extend(x)
            all_y.extend(y)

    all_x = np.array(all_x) + search_window[0]
    all_y = np.array(all_y)

    return all_x, all_y


def highest_n_peaks(histogram, peaks, n=2, threshold=0):
    if len(peaks) == 0:
        return []

    peak_list = []
    for peak in peaks:
        y = histogram[peak]
        if y > threshold:
            peak_list.append((peak, histogram[peak]))
    peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

    if len(peak_list) == 0:
        return []
    else:
        x, y = zip(*peak_list)
        return x[:n]


def histogram_smoothing(histograms, window=3):
    smoothed = np.zeros_like(histograms)
    for h_i, hist in enumerate(histograms):
        window_sum = np.zeros_like(hist)
        for w_i in range(window):
            index = w_i + h_i - window // 2
            if index < 0:
                index = 0
            elif index > len(histograms) - 1:
                index = len(histograms) - 1

            window_sum += histograms[index]

        smoothed[h_i] = window_sum / window

    return smoothed


def detect_highest_peak_in_area(histogram, peaks, threshold=0):
    peak = highest_n_peaks(histogram, peaks, n=1, threshold=threshold)
    if len(peak) == 1:
        return peak[0]
    else:
        return None


def detect_lane_along_poly(img, poly, steps):
    pixels_per_step = img.shape[0] // steps
    all_x = []
    all_y = []

    for i in range(steps):
        start = img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        x = poly(center)

        x, y = get_pixel_in_window(img, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y


def get_pixel_in_window(img, x_center, y_center, size):
    half_size = size // 2
    window = img[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]

    x, y = (window.T == 255).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y


def calculate_lane_area(lanes, img_height, steps):
    """
    Expects the line polynom to be a function of y.
    """
    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))

    for i in range(steps + 1):
        pixels_per_step = img_height // steps
        start = img_height - i * pixels_per_step

        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]

    return np.concatenate((points_left, points_right[::-1]), axis=0)


def are_lanes_plausible(lane_one, lane_two, parallel_thresh=(0.0003, 0.55), dist_thresh=(400, 600)):
    is_parall = lane_one.is_current_fit_parallel(lane_two, threshold=parallel_thresh)
    dist = lane_one.get_current_fit_distance(lane_two)
    is_plausible_dist = dist_thresh[0] < dist < dist_thresh[1]

    return is_parall & is_plausible_dist


def draw_poly(img, poly, steps, color, thickness=10, dashed=False):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img


def draw_poly_arr(img, poly, steps, color, thickness=10, dashed=False, tip_length=1):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed is False or i % 2 == 1:
            img = cv2.arrowedLine(img, end_point, start_point, color, thickness, tipLength=tip_length)

    return img


def outlier_removal(x, y, q=10):
    if len(x) == 0 or len(y) == 0:
        return x, y

    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]
