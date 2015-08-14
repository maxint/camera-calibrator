# coding: utf-8

import os
import numpy as np
import cv2

def calibrate(board_width, board_height, square_size, rectified_dir, debug_dir, save_file, image_patterns):
    import glob
    from common import splitfn

    img_names = glob.glob(image_patterns)

    pattern_size = (board_width, board_height)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    if debug_dir:
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

    if rectified_dir:
        if not os.path.exists(rectified_dir):
            os.makedirs(rectified_dir)

    obj_points = []
    img_points = []
    good_images = []
    h, w = 0, 0
    for fn in img_names:
        print 'processing %s...' % fn,
        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            cv2.imwrite('%s/%s_chess.png' % (debug_dir, name), vis)

        if not found:
            print 'chessboard not found in %s' % (fn, )
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)
        good_images.append(fn)

        print 'ok'

    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h))
    print "RMS:", rms
    print "camera matrix:\n", camera_matrix
    print "distortion coefficients: ", dist_coefs.ravel()

    if save_file:
        with open(save_file, 'wt') as f:
            m = camera_matrix
            v = [m[0][0], m[1][1], m[0][2], m[1][2]] + dist_coefs[0].tolist()
            f.write(' '.join(map(str, v)))
            f.write('\n')
            f.write(' '.join(map(str, [w, h])))

    if rectified_dir:
        # new_camera_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h),
        #                                                              alpha=0.55, centerPrincipalPoint=False)
        # print "new camera matrix:\n", new_camera_matrix
        # if new_camera_matrix[0, 0] < 0:
        #     new_camera_matrix[0, 0] *= -1
        # map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None,
        #                                          new_camera_matrix, (w, h),
        #                                          m1type=cv2.CV_16SC2)
        # for i in xrange(len(good_images)):
        #     fn = good_images[i]
        #     img = cv2.imread(fn, 0)
        #     r_img = cv2.remap(img, map1,map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        #     _, name, _ = splitfn(fn)
        #     cv2.imwrite('%s/%s_rectified.png' % (rectified_dir, name), r_img)

        for i in xrange(len(good_images)):
            fn = good_images[i]
            img = cv2.imread(fn, 0)
            _, name, _ = splitfn(fn)
            r_img2 = cv2.undistort(img, camera_matrix, dist_coefs)
            cv2.imwrite('%s/%s_rectified2.png' % (rectified_dir, name), r_img2)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Camera calibration tool powered by OpenCV')
    parser.add_argument('--board_width', type=int, default=9, help='number of board columns')
    parser.add_argument('--board_height', type=int, default=6, help='number of board rows')
    parser.add_argument('--square_size', type=float, default=1.0, help='square size (m)')
    parser.add_argument('--rectified_dir', default='debug', help='rectified image directory')
    parser.add_argument('--debug_dir', default='debug', help='debug output directory')
    parser.add_argument('-o', '--save_file', default='calib.cfg', help='final calibration file')
    parser.add_argument('image_patterns', nargs='?', default='*.jpg', help='input image patterns')

    args = parser.parse_args()
    calibrate(**vars(args))

