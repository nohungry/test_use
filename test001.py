import cv2
import numpy as np
import matplotlib.pyplot as plt

temp1 = r"C:\Users\norman_cheng\Desktop\airtest_process\deepseaadventure\deepsea_target\subtraction03.jpg"
temp2 = r"C:\Users\norman_cheng\Desktop\snapshot_log\screen_20220901_175301.jpg"

img1 = cv2.imread(temp1) # 目標圖
img2 = cv2.imread(temp2) # 資源圖

# 灰階
# img1 = cv2.imread(temp1, cv2.IMREAD_GRAYSCALE) # 目標圖
# img2 = cv2.imread(temp2, cv2.IMREAD_GRAYSCALE) # 資源圖

# 高斯模糊測試
img1 = cv2.GaussianBlur(img1, (15, 15), 0)
img2 = cv2.GaussianBlur(img2, (53, 53), 0)

# HSV (色相飽和度值)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# ----- 前處理測試區域 ------
# 需要搭配使用HSV
# minRed = np.array([0, 50, 50])
# maxRed = np.array([30, 255, 255])
# img1 = cv2.inRange(img1, minRed, maxRed)
# img2 = cv2.inRange(img2, minRed, maxRed)
# ----- 前處理測試區域 ------


# KAZE 暴力match
def kaze_test_match(img1, img2):
    # init kaze detector
    kaze = cv2.KAZE_create()

    # find the keypoints and descriptors with kaze
    kp1, des1 = kaze.detectAndCompute(img1, None)
    kp2, des2 = kaze.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # match descriptors
    matches = bf.match(des1, des2)

    # sort them in the order og their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # position
    position = []
    for i in matches:
        pos1 = kp1[i.queryIdx]
        pos2 = kp2[i.trainIdx]
        position.append((pos1, pos2))

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img3


# KAZE knn match
def kaze_test_match_knn(img1, img2):
    # init kaze detector
    kaze = cv2.KAZE_create()

    # find the keypoints and descriptors with kaze
    kp1, des1 = kaze.detectAndCompute(img1, None)
    kp2, des2 = kaze.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # match descriptors
    matches = bf.knnMatch(des1, des2, k=2)

    # choose good match points with equal distance
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    # alert
    if len(good) == 0:
        raise EOFError('somethings wrong')

    # position
    position = []
    for i in good:
        pos1 = kp1[i[0].queryIdx].pt
        pos2 = kp2[i[0].trainIdx].pt
        position.append((pos1, pos2))

    # Draw first 10 matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, matchColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img3


# KAZE FLANN match
def kaze_test_match_flann(img1, img2):
    # init kaze detector
    kaze = cv2.KAZE_create()

    # find the keypoints and descriptors with kaze
    kp1, des1 = kaze.detectAndCompute(img1, None)
    kp2, des2 = kaze.detectAndCompute(img2, None)

    # create FLANN match
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)

    # match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]

    drawPrams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask)
    # drawPrams = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0))

    # Draw first 10 matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, **drawPrams)

    return img3


# TODO 圖片分割成 3 * 3 區塊, 1 | 2 | 3
#                          4 | 5 | 6
#                          7 | 8 | 9
# return x_start, x_end, y_start, y_end 限制範圍
def img_crop_confirm(img, area=0):
    x_slicelength = int(abs(img.shape[1] / 3))
    y_slicelength = int(abs(img.shape[0] / 3))
    slicePoint = {
        "x0": 0,
        "x1": int(abs((0 + x_slicelength))),
        "x2": int(abs((0 + (2 * x_slicelength)))),
        "x3": int(abs(img.shape[1])),
        "y0": 0,
        "y1": int(abs((0 + y_slicelength))),
        "y2": int(abs((0 + (2 * y_slicelength)))),
        "y3": int(abs(img.shape[0])),
    }
    if area == 1:
        x_start = slicePoint["x0"]
        x_end = slicePoint["x1"]
        y_start = slicePoint["y0"]
        y_end = slicePoint["y1"]
        return x_start, x_end, y_start, y_end
    elif area == 2:
        x_start = slicePoint["x1"]
        x_end = slicePoint["x2"]
        y_start = slicePoint["y0"]
        y_end = slicePoint["y1"]
        return x_start, x_end, y_start, y_end
    elif area == 3:
        x_start = slicePoint["x2"]
        x_end = slicePoint["x3"]
        y_start = slicePoint["y0"]
        y_end = slicePoint["y1"]
        return x_start, x_end, y_start, y_end
    elif area == 4:
        x_start = slicePoint["x0"]
        x_end = slicePoint["x1"]
        y_start = slicePoint["y1"]
        y_end = slicePoint["y2"]
        return x_start, x_end, y_start, y_end
    elif area == 5:
        x_start = slicePoint["x1"]
        x_end = slicePoint["x2"]
        y_start = slicePoint["y1"]
        y_end = slicePoint["y2"]
        return x_start, x_end, y_start, y_end

# main
if __name__ == '__main__':
    img3 = kaze_test_match(img1, img2)
    # img3 = img3[:, :, ::-1]
    # plt.subplot(1, 2, 1)
    # plt.imshow(img1)
    # plt.subplot(1, 2, 2)
    plt.imshow(img3)
    plt.show()