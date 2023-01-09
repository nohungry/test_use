# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import pandas
# import matplotlib.patches as patches
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# rect = patches.Rectangle(
#     (1, 1),  # (x,y)
#     1,  # width
#     1,  # height
#     color='g',  # 邊的顏色
#     fill=False  # 是否填充，默認填充
#
# )
#
# ax.add_patch(rect)
# ax.axis("equal")
# plt.show()

# from PIL import Image

# im = Image.open(r"C:\Users\norman_cheng\Desktop\snapshot_log\screen_20220824_112047.jpg")
# pass

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

test = mpimg.imread(r"C:\Users\norman_cheng\Desktop\norman_test003\screen_20220829_125221.jpg")
plt.imshow(test)
# plt.plot(771, 1820, marker=".", color="red")
# plt.plot(699, 1751, marker=".", color="b")
# plt.plot(699, 1889, marker=".", color="g")
# plt.plot(844, 1889, marker=".", color="y")
# plt.plot(844, 1751, marker=".", color="k")
plt.show()

