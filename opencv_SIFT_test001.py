import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- opencv中sift特徵點檢測 & 繪製

# img1 = cv2.imread(r"C:\Users\norman_cheng\Desktop\airtest_process\deepseaadventure\deepsea_target2\spin.png")
# img2 = cv2.imread(r"C:\Users\norman_cheng\Desktop\snapshot_log\screen_20220901_175354.jpg")
# img3 = cv2.imread(r"C:\Users\norman_cheng\Desktop\snapshot_log\screen_20220901_175354.jpg")
#
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
# keypoints, descriptors = sift.detectAndCompute(img2_gray, None)
#
# for keypoint, descriptor in zip(keypoints, descriptors):
#     print("keypoint:", keypoint.angle, keypoint.class_id, keypoint.octave, keypoint.pt, keypoint.response, keypoint.size)
#     print("descriptor: ", descriptor.shape)
#
# img = cv2.drawKeypoints(image=img2_gray, outImage=img2, keypoints=keypoints,
#                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#                         color=(220, 10, 10))
#
# # cv2.imshow("img_gray", img3)
# plt.imshow(img2)
# plt.show()
# # cv2.imshow("new_img", img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# ---- opencv中sift特徵點檢測 & 繪製

tarimg1 = cv2.imread(r"C:\Users\norman_cheng\Desktop\airtest_process\deepseaadventure\deepsea_target\spin.jpg")
tarimg2 = cv2.imread(r"C:\Users\norman_cheng\Desktop\airtest_process\deepseaadventure\deepsea_target2\spin.png")
oriimg1 = cv2.imread(r"C:\Users\norman_cheng\Desktop\snapshot_log\screen_20220901_175354.jpg")
oriimg2 = cv2.imread(r"C:\Users\norman_cheng\Desktop\snapshot_log\screen_20220901_175354.jpg")

sift = cv2.SIFT_create()

okp1, odes1 = sift.detectAndCompute(oriimg1, None)
tkp1, tdes1 = sift.detectAndCompute(tarimg1, None)

# --------- 匹配方法 BFMatcher，FlannBasedMatcher:

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(odes1, tdes1, k=2)
good_match = []
test = matches[0]
good_match.append(test[0])

h1, w1 = oriimg1.shape[:2]
h2, w2 = tarimg1.shape[:2]

outtest001 = np.zeros

pass
# for m in matches:
#     print(m[0].queryIdx, m[0].queryIdx, m[0].distance)
#     print(m[1].queryIdx, m[1].queryIdx, m[1].distance)



# -------- 繪製圖項
# output = cv2.drawMatches()

# plt.plot(69, 75, marker=".", color="red")
# plt.plot(69, 84, marker=".", color="r")
# plt.plot(74, 84, marker=".", color="r")
# plt.plot(75, 75, marker=".", color="r")
# plt.plot(71, 79, marker=".", color="g")

plt.plot(451, 1960, marker=".", color="red")
plt.plot(451, 1961, marker=".", color="r")
plt.plot(453, 1961, marker=".", color="r")
plt.plot(453, 1960, marker=".", color="r")
plt.plot(452, 1960, marker=".", color="g")


plt.imshow(oriimg1)
plt.show()