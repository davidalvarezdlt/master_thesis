from __future__ import print_function
import cv2
import numpy as np
import torch


def alignment(x_target, x_ref):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    im1 = (x_ref.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im2 = (x_target.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return torch.from_numpy(im1Reg.astype(np.float) / 255).permute(2, 0, 1)


def inpainting(x_target, m_target, x_refs, m_refs):
    a = 1
