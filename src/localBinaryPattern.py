import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops

oil_locations = [(230, 290), (257, 303), (273, 336), (311, 336)]
ocean_locations = [(344, 234), (397, 118), (43, 305), (427, 302)]

rgb_img = cv2.imread("./image/unnamed.jpg")
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

ocean_crops = []
oil_crops = []
rgb_crops = []
crop_offset = 50

for loc in oil_locations:
	crop = gray_img[loc[1]:loc[1] + crop_offset, loc[0]:loc[0] + crop_offset]
	oil_crops.append(crop)
	cv2.rectangle(rgb_img, (loc[0], loc[1]), (loc[0] + crop_offset,loc[1] + crop_offset), color = (0, 0, 255), thickness = 3)

for loc in ocean_locations:
	crop = gray_img[loc[1]:loc[1] + crop_offset, loc[0]:loc[0] + crop_offset]
	ocean_crops.append(crop)
	cv2.rectangle(rgb_img, (loc[0], loc[1]), (loc[0] + crop_offset,loc[1] + crop_offset), color = (255, 0, 0), thickness = 3)

# settings for LBP
radius = 2
n_points = 8 * radius
METHOD = 'uniform'

def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins), facecolor='0.5')

def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, density=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

refs = []
for image in ocean_crops:
	ref = local_binary_pattern(image, n_points, radius, METHOD)
	refs.append(ref)


# plot histograms of LBP of textures
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(9, 6))
plt.gray()

ax1.imshow(ocean_crops[0])
ax1.axis('off')
hist(ax4, refs[0])
ax4.set_ylabel('Percentage')

ax2.imshow(ocean_crops[1])
ax2.axis('off')
hist(ax5, refs[1])
ax5.set_xlabel('Uniform LBP values')

ax3.imshow(ocean_crops[2])
ax3.axis('off')
hist(ax6, refs[2])

plt.show()
