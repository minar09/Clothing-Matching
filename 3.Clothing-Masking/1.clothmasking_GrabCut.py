### Author: Matiur Rahman Minar ###
### EMCOM Lab, SeoulTech, 2021 ###
### Task: Generating binary mask/silhouette/segmentation ###
### especially for clothing image ###
### Focused method: GrabCut ###


import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def cloth_masking_with_grabcut(im_path, mask_path, viz=False):
    lo = 250
    hi = 255

    img = cv2.imread(im_path, 0)
    img1 = Image.open(im_path).convert('RGB')
    img2 = cv2.imread(im_path)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # 1. binary thresholding
    ret, th_bin = cv2.threshold(img, lo, hi, cv2.THRESH_BINARY_INV)

    # 2. Filling operation:

    # 2.1 Copy the thresholded image.
    im_floodfill = th_bin.copy()
    # 2.2 Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = th_bin.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # 2.3 Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    # 2.4 Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # 2.5 Combine the two images to get the foreground.
    th_filled = th_bin | im_floodfill_inv

    # 3. Morphology operation:
    kernel = np.ones((2, 2), np.uint8)

    # 3.1 opening for salt noise removal
    th_opened = cv2.morphologyEx(th_filled, cv2.MORPH_OPEN, kernel)

    # 3.2 closing for pepper noise removal (not needed it seems)
    # th_closed = cv2.morphologyEx(th_opened, cv2.MORPH_CLOSE, kernel)

    # 3.3 erosion for thinning out boundary
    # kernel = np.ones((3, 3), np.uint8)
    # th_eroded = cv2.erode(th_opened, kernel, iterations=1)

    # 4. GrabCut

    # 4.1 make mask
    # wherever it is marked white (sure foreground), change mask=1
    # wherever it is marked black (sure background), change mask=0
    gc_mask = np.zeros(img2.shape[:2], np.uint8)
    newmask = th_opened.copy()
    newmask_segm = cv2.bitwise_and(img2, img2, mask=newmask)

    # 4.2 define GrabCut priors
    absolute_foreground = cv2.erode(newmask, kernel, iterations=2)
    probable_foreground = newmask - absolute_foreground
    dilated_newmask = cv2.dilate(newmask, kernel, iterations=2)
    absolute_background = cv2.bitwise_not(dilated_newmask)
    probable_background = dilated_newmask - newmask

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 4.3 change mask based on priors
    # any mask values greater than zero should be set to probable
    # foreground
    gc_mask[absolute_foreground > 0] = cv2.GC_FGD
    gc_mask[probable_foreground > 0] = cv2.GC_PR_FGD
    gc_mask[absolute_background > 0] = cv2.GC_BGD
    gc_mask[probable_background > 0] = cv2.GC_PR_BGD
    gc_prior = gc_mask.copy()

    # 4.4 apply GrabCut masking/segmentation
    gc_mask, bgdModel, fgdModel = cv2.grabCut(img2, gc_mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)
    gc_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype('uint8')
    img2 = img2 * gc_mask[:, :, np.newaxis]

    # 5. plot/visualize results
    if viz:
        # plot figures:
        titles = ['Original Image', 'Binary thresholding', 'Filling', 'Initial mask', 'Initial segm', 'GrabCut Priors',
                  'GrabCut Mask', 'GrabCut segm']
        images = [img1, th_bin, th_filled, th_opened, newmask_segm, gc_prior, gc_mask, img2]

        for i in range(8):
            plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

    # 6. save result
    gc_mask[gc_mask > 0] = 255    # make visible white
    print("Saving ", save_path)
    cv2.imwrite(mask_path, gc_mask)


def main():
    # Get paths
    cloth_dir = "data/cloth/"
    res_dir = "results/masks/"
    image_list = os.listdir(cloth_dir)

    # iterate images in the path
    for each in image_list:
        image_path = os.path.join(cloth_dir, each)
        res_path = os.path.join(res_dir, each.replace(".jpg", ".png"))
        cloth_masking_with_grabcut(image_path, res_path, viz=True)


if __name__ == "__main__":
    main()
