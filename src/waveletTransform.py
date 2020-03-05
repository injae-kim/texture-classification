import numpy as np
import matplotlib.pyplot as plt
import cv2
import pywt
from skimage.feature import greycomatrix, greycoprops

oil_locations = [(230, 290), (257, 303), (273, 336), (311, 336)]
ocean_locations = [(344, 234), (397, 118), (43, 305), (427, 302)]

rgb_img = cv2.imread("./image/article.jpg")
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

ocean_crops = []
oil_crops = []
rgb_crops = []
crop_offset = 21

for loc in oil_locations:
  crop = gray_img[loc[1]:loc[1] + crop_offset, loc[0]:loc[0] + crop_offset]
  oil_crops.append(crop)
  cv2.rectangle(rgb_img, (loc[0], loc[1]), (loc[0] + crop_offset,loc[1] + crop_offset), color = (0, 0, 255), thickness = 3)

for loc in ocean_locations:
  crop = gray_img[loc[1]:loc[1] + crop_offset, loc[0]:loc[0] + crop_offset]
  ocean_crops.append(crop)
  cv2.rectangle(rgb_img, (loc[0], loc[1]), (loc[0] + crop_offset,loc[1] + crop_offset), color = (255, 0, 0), thickness = 3)

# display original image with locations of patches
# ax = fig.add_subplot(9, 1, 1)
# ax.imshow(rgb_img)
# ax.set_xlabel('Original Image')
# ax.set_xticks([])
# ax.set_yticks([])
# ax.axis('image')

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']


coeffs2 = pywt.dwt2(gray_img, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))

for i, a in enumerate([LL, LH, HL, HH]):
  ax = fig.add_subplot(1, 4, i + 1)
  ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
  ax.set_title(titles[i], fontsize=10)
  ax.set_xticks([])
  ax.set_yticks([])

fig.tight_layout()
plt.show()