import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def get_Middle_Str(content, startStr, endStr):
    """
    根据字符串首尾字符来获取指定字符
    :param content: 字符内容
    :param startStr: 开始字符
    :param endStr: 结束字符
    :return:
    """
    startIndex = content.index(startStr)
    if startIndex >= 0:
        startIndex += len(startStr)
    endIndex = content.index(endStr)
    return content[startIndex:endIndex]


def get_image_element_point(src_path, dst_path):
    """
    获取图像目标的坐标点
    :param src_path: 原图像
    :param dst_path: 目标识别图像
    :return: 目标元素的中心坐标点
    """

    print('src_path:%s,dst_path:%s' % (src_path, dst_path))

    # 以灰度模式读取图像
    src_img = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
    dst_img = cv.imread(dst_path, cv.IMREAD_GRAYSCALE)

    # 创建SITF对象
    sift = cv.SIFT_create()

    # 使用SITF找到关键点和特征描述
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(dst_img, None)

    # FLANN 匹配算法参数
    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 第一个参数指定算法
    search_params = dict(checks=50)  # 指定应递归遍历索引中的树的次数

    # flann特征匹配
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 初始化匹配模板表
    matchesMask = [[0, 0] for i in range(len(matches))]

    good = []

    # 匹配阈值
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            good.append(m)
            matchesMask[i] = [1, 0]

    MIN_MATCH_COUNT = 10

    # 获取转换矩阵
    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 获取变换矩阵，M就是变化矩阵
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # 获得原图像高和宽
        h, w = src_img.shape

        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        # 提取坐标点
        cordinate_x1 = get_Middle_Str(str(dst[2]), '[[', ']]').split()[0].split('.')[0]
        cordinate_y1 = get_Middle_Str(str(dst[2]), '[[', ']]').split()[1].split('.')[0]

        cordinate_x2 = get_Middle_Str(str(dst[0]), '[[', ']]').split()[0].split('.')[0]
        cordinate_y2 = get_Middle_Str(str(dst[0]), '[[', ']]').split()[1].split('.')[0]

        # 提取目标元素中心坐标点
        mid_cordinate_x = (int(cordinate_x1) - int(cordinate_x2)) / 2 + int(cordinate_x2)
        mid_cordinate_y = (int(cordinate_y1) - int(cordinate_y2)) / 2 + int(cordinate_y2)

        # print(mid_cordinate_x/2,mid_cordinate_y/2)

        # 原图像还原为灰度
        img2 = cv.polylines(dst_img, [np.int32(dst)], True, 255, 10, cv.LINE_AA)

        ############打印图像轮廓#################
        draw_params = dict(matchColor=(0, 255, 0),
                           # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv.drawMatches(src_img, kp1, dst_img, kp2, good, None, **draw_params)
        # plt.imshow(img3, 'gray'), plt.show()

        return int(mid_cordinate_x / 2), int(mid_cordinate_y / 2)

    else:
        print("------------------------------------------------------------------")
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        print("------------------------------------------------------------------")
        matchesMask = None


if __name__ == '__main__':
    img1 = r'C:/Users/brent_yang/Desktop/Selenium/pic/background.png'
    img2 = r'C:/Users/brent_yang/Desktop/Selenium/pic/spin.png'
    get_image_element_point(img1, img2)