# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:33:44 2024

@author: emmay
"""
'''
import cv2
import numpy as np

# Load your original stereo images
img_left = cv2.imread("Image1.png")
img_right = cv2.imread("Image2.png")

# Convert images to grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Detect feature points (e.g., using SIFT)
sift = cv2.SIFT_create()
keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

# Match feature points
matcher = cv2.BFMatcher()
matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

# Filter matches using ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Get corresponding points
pts_left = np.float32([keypoints_left[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts_right = np.float32([keypoints_right[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute the fundamental matrix
F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_LMEDS)

# Draw epipolar lines on the left image
lines_left = cv2.computeCorrespondEpilines(pts_right, 2, F)
lines_left = lines_left.reshape(-1, 3)

img_left_epilines = img_left.copy()
for line, pt_right in zip(lines_left, pts_left):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [img_left.shape[1], -(line[2] + line[0] * img_left.shape[1]) / line[1]])
    cv2.line(img_left_epilines, (x0, y0), (x1, y1), color, 1)
    cv2.circle(img_left_epilines, (int(pt_right[0][0]), int(pt_right[0][1])), 5, color, -1)


# Draw epipolar lines on the right image
lines_right = cv2.computeCorrespondEpilines(pts_left, 1, F)
lines_right = lines_right.reshape(-1, 3)

img_right_epilines = img_right.copy()
for line, pt_left in zip(lines_right, pts_right):
    color = tuple(np.random.randint(0, 255, 3).tolist())
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [img_right.shape[1], -(line[2] + line[0] * img_right.shape[1]) / line[1]])
    cv2.line(img_right_epilines, (x0, y0), (x1, y1), color, 1)
    cv2.circle(img_left_epilines, (int(pt_right[0][0]), int(pt_right[0][1])), 5, color, -1)

# Display the images with epipolar lines and corresponding points
cv2.imshow('Left Image with Epipolar Lines', img_left_epilines)
cv2.imshow('Right Image with Epipolar Lines', img_right_epilines)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#########################################################################
# https://www.andreasjakl.com/how-to-apply-stereo-matching-to-generate-depth-maps-part-3/

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load your original stereo images
#img_left = cv.imread("Daphne1.png")
#img_right = cv.imread("Daphne2.png")

img_left = cv.imread("Image1.png")
img_right = cv.imread("Image2.png")


img1 = img_left
img2 = img_right

# detect keypoints and their descriptors
sift = cv.SIFT_create()
#find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Visualize keypoints
imgSift = cv.drawKeypoints(
    img1, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("SIFT Keypoints", imgSift)
cv.waitKey(0)

# Match keypoints in both images
# Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # Keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
        
# Draw the keypoint matches between both pictures
# Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask[300:500],
                   flags=cv.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv.drawMatchesKnn(
    img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
cv.imshow("Keypoint matches", keypoint_matches)
cv.waitKey(0)

# STEREO RECTIFICATION

# Calculate the fundamental matrix for the cameras
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

# We select only inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]


# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1, d1 = img1.shape
h2, w2, d2 = img2.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
)

# Undistort (rectify) the images and save them
# Adapted from: https://stackoverflow.com/a/62607343
img1_rectified = cv.warpPerspective(img1, H1, (w1, h1))
img2_rectified = cv.warpPerspective(img2, H2, (w2, h2))
cv.imshow("rectified_1.png", img1_rectified)
cv.waitKey(0)
cv.imshow("rectified_2.png", img2_rectified)
cv.waitKey(0)

# Draw the rectified images
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(img1_rectified, cmap="gray")
axes[1].imshow(img2_rectified, cmap="gray")
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
plt.suptitle("Rectified images")
plt.savefig("rectified_images.png")
plt.show()

# ------------------------------------------------------------
# CALCULATE DISPARITY (DEPTH MAP)
# Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
# and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

# StereoSGBM Parameter explanations:
# https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 11
min_disp = -128
max_disp = 128
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 200
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 0

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
)
disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
cv.imshow("Disparity", disparity_SGBM, )
cv.waitKey(0)

plt.imshow(disparity_SGBM, cmap='plasma')
plt.colorbar()
plt.show()


#Generate point cloud
#https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
from TakePic_GetCamIntr import CamIntr
h, w = img1_rectified.shape[:2]
#get focal length
ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = CamIntr(["Image1.png"], [6,9], 0.025)
f = mtx[0,0]
Q = np.float32([[1, 0, 0, -0.5*w],
                [0, -1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])
points = cv.reprojectImageTo3D(disparity_SGBM, Q)
#colors = cv.cvtColor(img1_rectified, cv.COLOR_BGR2RGB)
mask = disparity_SGBM > disparity_SGBM.min() #+ 150
out_points = points[mask]

import open3d as o3d

vis = o3d.geometry.PointCloud()
vis.points = o3d.utility.Vector3dVector(out_points)
o3d.visualization.draw_geometries([vis])

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d') 
#ax.scatter(xs, ys, zs, marker=m)

'''
#perform filtering
img1g = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2g = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

wsize=11
max_disp = 128
sigma = 1#1.5
lmbda = 1.0#8000.0
left_matcher = cv.StereoBM_create(max_disp, wsize);
right_matcher = cv.ximgproc.createRightMatcher(left_matcher);
left_disp = left_matcher.compute(img1g, img2g);
right_disp = right_matcher.compute(img1g,img2g);

# Now create DisparityWLSFilter
wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher);
wls_filter.setLambda(lmbda);
wls_filter.setSigmaColor(sigma);
filtered_disp = wls_filter.filter(left_disp, img1g, disparity_map_right=right_disp);

cv.imshow("Disparity filtered", filtered_disp, )
cv.waitKey(0)
'''




