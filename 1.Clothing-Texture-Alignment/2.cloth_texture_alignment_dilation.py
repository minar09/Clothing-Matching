### Author: Matiur Rahman Minar ###
### EMCOM Lab, SeoulTech, 2021 ###
### Task: Extrapolating clothing boundary for aligning/matching texture to extended silhouette/mask ###
### helpful for reducing artifacts in 3D clothing reconstruction ###
### Focused method: Gray dilation ###


import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def mask_segm(immask):
    """
    :param immask: binary mask/silhouette/segmentation in Umat/OpenCV format
    :return: binary mask/silhouette/segmentation in pillow format
    """
    mask = np.array(immask)
    mask = np.reshape(mask, mask.shape + (1,))
    mask = (mask == 255)

    return mask


def apply_gray_dilation(img):
    """
    :param img: input image
    :return: dilated image (gray dilated each channel separately)
    """
    # split channels
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b, g, r = cv2.split(im_rgb)

    # dilation for extrapolating out of boundary
    kernel = np.ones((3, 3), np.uint8)
    iter = 5
    b = cv2.dilate(b, kernel, iterations=iter)
    g = cv2.dilate(g, kernel, iterations=iter)
    r = cv2.dilate(r, kernel, iterations=iter)

    # merge channels
    exp_img = cv2.merge((r, g, b))    # Merge dilated channels

    return exp_img


def cloth_boundary_extrapolation(im_path, mask_path, save_path, viz=False, save=True):
    # read images
    img = Image.open(im_path)
    immask = Image.open(mask_path)
    c_mask = cv2.imread(mask_path, 0)
    mask = mask_segm(immask)

    # dilation of mask for extrapolating out of boundary
    kernel = np.ones((3, 3), np.uint8)
    iter = 2
    ce_mask = cv2.erode(c_mask, kernel, iterations=iter)
    cemask = mask_segm(ce_mask)
    cd_mask = cv2.dilate(c_mask, kernel, iterations=iter)
    dilated_boundary_mask = cd_mask - ce_mask
    eroded_cloth = img * cemask

    # get extrapolated/extended cloth
    dilated_cloth = apply_gray_dilation(eroded_cloth)
    exp_img = eroded_cloth * cemask + dilated_cloth * (1 - cemask)

    # visualize result
    if viz:
        # plot figures:
        titles = ['Original Image', 'Eroded cloth', 'extra boundary', 'segmented (original)', 'Dilated cloth',
                  'Extrapolated']
        images = [img, eroded_cloth, dilated_boundary_mask, img * mask, dilated_cloth, exp_img]

        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

    # save result
    if save:
        # save result
        img_float32 = np.float32(exp_img)
        bgr_img = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_img)


def main():
    # Get paths
    cloth_dir = "data/cloth/"
    mask_dir = "data/masks/"
    res_dir = "results/clothes/"
    image_list = os.listdir(cloth_dir)

    # iterate images in the path
    for each in image_list:
        image_path = os.path.join(cloth_dir, each)
        mask_path = os.path.join(mask_dir, each.replace(".jpg", ".png"))
        res_path = os.path.join(res_dir, each.replace(".jpg", ".png"))
        cloth_boundary_extrapolation(image_path, mask_path, res_path, viz=True, save=True)


if __name__ == "__main__":
    main()
