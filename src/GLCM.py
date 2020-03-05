import numpy as np
import matplotlib.pyplot as plt
import cv2


from skimage.feature import greycomatrix, greycoprops

oil_locations = [(230, 290), (257, 303), (273, 336), (311, 336)]
ocean_locations = [(344, 234), (397, 118), (43, 305), (427, 302)]

rgb_img = cv2.imread("./image/unnamed.jpg")
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

print(len(ocean_crops))

xs = []
ys = []



for patch in (ocean_crops + oil_crops):

    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'contrast')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])

fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(rgb_img)
ax.set_xlabel('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(xs[:len(ocean_crops)], ys[:len(ocean_crops)], 'ro',
        label='Ocean')
ax.plot(xs[len(oil_crops):], ys[len(oil_crops):], 'bo',
        label='Oil')
ax.set_xlabel('GLCM contrast')
ax.set_ylabel('GLCM correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(ocean_crops):
    ax = fig.add_subplot(3, len(ocean_crops), len(ocean_crops)*1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Ocean %d' % (i + 1))

for i, patch in enumerate(oil_crops):
    ax = fig.add_subplot(3, len(oil_crops), len(oil_crops)*2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_xlabel('Oil %d' % (i + 1))


fig.tight_layout()
plt.show()