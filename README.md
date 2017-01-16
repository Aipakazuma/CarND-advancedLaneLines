# Advanced Lane Finding

This project was created as an assessment for the [Self-Driving Car Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013) Program by Udacity. The goal is to detect lane lines on videos, calculate the curvature of the and the offset of the car.

## Result
### Images
![sample detection](sample_output.jpg)

### Videos
Click on the image to play the video.

Track 1                       |  Track 1 + Debugging
:----------------------------:|:------------------------------:
[![Track 1](http://img.youtube.com/vi/IrW4YQ9iKRY/0.jpg)](http://www.youtube.com/watch?v=IrW4YQ9iKRY) | [![Track 1 D](http://img.youtube.com/vi/sU_SbmkqLMs/0.jpg)](http://www.youtube.com/watch?v=sU_SbmkqLMs) 


Track 2                      |  Track 2 + Debugging
:----------------------------:|:------------------------------:
[![Track 2](http://img.youtube.com/vi/gPTx25qRpOY/0.jpg)](http://www.youtube.com/watch?v=gPTx25qRpOY) | [![Track 2 D](http://img.youtube.com/vi/5oFHE803Hyw/0.jpg)](http://www.youtube.com/watch?v=5oFHE803Hyw) 

## Camera Calibration

Original                     |  Undistorted
:----------------------------:|:------------------------------:
![Original](camera_cal/calibration1.jpg)| ![Undistorted](output_images/calibration1_undist.jpg)

## Pipeline

### Undistortion
Original                     |  Undistorted
:----------------------------:|:------------------------------:
![Original](test_images/test1.jpg)| ![Undistorted](output_images/test1_undist.jpg)

### Lane Masking
Sobel X & Y                   |  Magnitude & Direction of Gradient  | Yellow | Highlights | Combined
:----------------------------:|:-----------------------------------:|:------:|:----------:|:---------:
![Sobel](output_images/test1_mask_sobelxy.jpg)| ![Gradient](output_images/test1_mask_gradient_mag_dir.jpg) | ![Yellow](output_images/test1_mask_ylw.jpg) | ![Highlights](output_images/test1_mask_highlights.jpg) | ![Combined](output_images/test1_mask.jpg)

### Birdseye View
Source                        |  Transformed
:----------------------------:|:-----------------------------------------------------------:
![Mask](test_images/ground_plane.jpg)| ![Birdseye](output_images/ground_plane_birdseye.jpg)

Mask                          |  Birdseye View
:----------------------------:|:-----------------------------------------------------------:
![Mask](output_images/test1_mask.jpg)| ![Birdseye](output_images/test1_birdseye.jpg)

### Identify Pixels
Left Lane Histogram           | Right Lane Histogram           | Assigned Pixels
:------------------------------:|:------------------------------:|:------------------------------:
![Left Lane Histogram](output_images/test1_histogram_left.jpg) | ![Right Lane Histogram](output_images/test1_histogram_right.jpg) | ![Assigned Pixels](output_images/test1_pixel.jpg)

### Fit Polynomial
