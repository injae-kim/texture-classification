import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage as ndi

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


oil_locations = [(230, 290), (257, 303), (273, 336), (311, 336)]
ocean_locations = [(344, 234), (397, 118), (43, 305), (427, 302)]

rgb_img = cv2.imread("./image/unnamed.jpg")
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

ocean_crops = []
oil_crops = []
rgb_crops = []
crop_offset = 30

for loc in oil_locations:
    crop = gray_img[loc[1]:loc[1] + crop_offset, loc[0]:loc[0] + crop_offset]
    oil_crops.append(crop)
    cv2.rectangle(rgb_img, (loc[0], loc[1]), (loc[0] + crop_offset,loc[1] + crop_offset), color = (0, 0, 255), thickness = 3)

for loc in ocean_locations:
    crop = gray_img[loc[1]:loc[1] + crop_offset, loc[0]:loc[0] + crop_offset]
    ocean_crops.append(crop)
    cv2.rectangle(rgb_img, (loc[0], loc[1]), (loc[0] + crop_offset,loc[1] + crop_offset), color = (255, 0, 0), thickness = 3)

# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


shrink = (slice(0, None, 3), slice(0, None, 3))

images = ocean_crops[:]
image_names = ['ocean1', 'ocean2', 'ocean3', 'ocean4']

for image in images:
	image = img_as_float(image)[shrink]

# prepare reference features
ref_feats = np.zeros((len(images), len(kernels), 2), dtype=np.double)

for image, ref_feat in zip(images, ref_feats):
	ref_feats = compute_feats(image, kernels)


def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# Plot a selection of the filter bank kernels and their responses.
results = []
kernel_params = []
for theta in (2, 3):
    theta = theta / 4. * np.pi
    for frequency in (0.1, 0.4):
        kernel = gabor_kernel(frequency, theta=theta)
        params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)
        # Save kernel and the power image for each image
        results.append((kernel, [power(img, kernel) for img in images]))

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(5, 7))
plt.gray()

fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

axes[0][0].axis('off')

# Plot original images
for label, img, ax in zip(image_names, images, axes[0][1:]):
    ax.imshow(img)
    ax.set_title(label, fontsize=9)
    ax.axis('off')

for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
    # Plot Gabor kernel
    ax = ax_row[0]
    ax.imshow(np.real(kernel))
    ax.set_ylabel(label, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot Gabor responses with the contrast normalized for each filter
    vmin = np.min(powers)
    vmax = np.max(powers)
    for patch, ax in zip(powers, ax_row[1:]):
        ax.imshow(patch, vmin=vmin, vmax=vmax)
        ax.axis('off')

plt.show()
