Camera calibration tool powered by OpenCV, the result is in [OpenCV camera
calibration format](http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html).

# Dependencies

- OpenCV >= 2.4
- Python 2.7
- numpy

# Example Usage

Calculate camera calibration parameters from JPG files.
```
calibrate2.py --board_width 9 -board_height 6 -square_size 0.020588 *.jpg
```

# Help
```
calibrate2.py -h
```
