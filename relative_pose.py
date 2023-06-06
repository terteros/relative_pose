import cv2
import numpy as np


# from 2D-3D correspondences we obtain a good focal length. We will use this focal length for decomposing the essential
# matrix and in COLMAP.
vr2d = np.load('vr2d.npy')
vr3d = np.load('vr3d.npy')
cx,cy, f_init = 960, 540, 100
K_init = np.eye(3)
K_init[0, 0], K_init[1, 1], K_init[0,2], K_init[1,2] = f_init, f_init, cx, cy
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(vr3d.transpose((1, 0, 2)), vr2d.transpose((1, 0, 2)), (cx * 2, cy * 2), K_init, None,
                                                 flags=cv2.CALIB_USE_INTRINSIC_GUESS|cv2.CALIB_FIX_ASPECT_RATIO| cv2.CALIB_FIX_PRINCIPAL_POINT|cv2.CALIB_ZERO_TANGENT_DIST|cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3)
print("Estimated Focal Length: ", K[0,0])


def relativeCameraPose(im1, im2):
    '''

    :param im1: image with the identity pose.
    :param im2: image captured from the pose we want to estimate.
    :return: rotation R and translation T (up to a scale).
    Mostly from openCV tutorials.
    '''
    # extract sift features from images
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # match features using KNN. Flann may work better but didn't test it.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    matched_kp1, matched_kp2 = [], []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])
            matched_kp1.append(kp1[m.queryIdx].pt)
            matched_kp2.append(kp2[m.trainIdx].pt)
    matched_kp1, matched_kp2 = np.array(matched_kp1), np.array(matched_kp2)

    # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(im1, kp1, im2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imwrite('matches.png', img3)
    # find the essential matrix and recover pose
    E, mask = cv2.findEssentialMat(matched_kp1, matched_kp2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    res, R, T, mask = cv2.recoverPose(E, matched_kp1, matched_kp2, K, mask)
    return R, T




img1 = cv2.cvtColor(cv2.imread('images/img1.png'), cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('images/img2.png'), cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(cv2.imread('images/img3.png'), cv2.COLOR_BGR2GRAY)

print( 'Pose of cam2 w.r.t cam1: ', relativeCameraPose(img1, img2))
print( 'Pose of cam3 w.r.t cam1: ', relativeCameraPose(img1, img3))


# 1 0.99999999973382792 -1.2565863114999059e-05 -7.8946044732357913e-06 1.7666875452767466e-05 3.2058883538270511 -0.18268941773970113 4.2649464458441679 1 img1.png
# 2 0.9999999975276973 2.118927515641026e-05 6.6000432961189041e-05 -1.1813673683345705e-05 -2.3961411969631636 -0.17961230417713589 -0.26391571985269946 1 img2.png
# 3 0.99297053760841747 0.0072895122797531969 -0.11783670846450343 -0.0084192987015187758 0.15491180586905226 0.42740984366188028 -4.0725892136181141 1 img3.png

