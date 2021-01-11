### Author: Matiur Rahman Minar ###
### EMCOM Lab, SeoulTech, 2021 ###
### Task: Extrapolating clothing boundary for aligning/matching texture to extended silhouette/mask ###
### helpful for reducing artifacts in 3D clothing reconstruction ###
### Focused method: TPS transformation ###


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


def apply_tps(body_shape, cloth_shape, matches, img):
    """
    :param body_shape: target points for TPS
    :param cloth_shape: source points for TPS
    :param matches: point matches
    :param img: clothing image
    :return: matched/warped image
    """
    # Forward TPS
    tps = cv2.createThinPlateSplineShapeTransformer(
        regularizationParameter=0)
    tps.estimateTransformation(body_shape, cloth_shape, matches)
    tps.applyTransformation(body_shape)

    # forward warping
    warped_img = tps.warpImage(img)

    return warped_img


def cloth_boundary_extrapolation(im_path, mask_path, save_path, viz=False, save=True):
    """
    :param im_path: clothing image location
    :param mask_path: location of the binary mask of clothing
    :param save_path: location for saving aligned clothing texture image
    :param viz: whether to visualize
    :param save: whether to save result
    :return: None
    """

    # read from paths
    img_rgb = Image.open(im_path)
    img_gray = cv2.imread(im_path, 0)
    immask = Image.open(mask_path)
    c_mask = cv2.imread(mask_path, 0)
    mask = mask_segm(immask)

    # dilation for extrapolating out of boundary
    kernel = np.ones((3, 3), np.uint8)
    iter = 2
    ce_mask = cv2.erode(c_mask, kernel, iterations=iter)
    cemask = mask_segm(ce_mask)
    cd_mask = cv2.dilate(c_mask, kernel, iterations=iter)
    dilated_boundary_mask = cd_mask - ce_mask
    eroded_cloth = img_rgb * cemask

    emask = mask_segm(ce_mask)
    dbmask = mask_segm(dilated_boundary_mask)

    # apply tps
    x_cloth = []
    y_cloth = []
    x_smpl = []
    y_smpl = []
    x_smpl_2nd = []
    y_smpl_2nd = []

    # get contours
    ret, th_bin = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
    cloth_contours, hierarchy = cv2.findContours(th_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_contours, hierarchy = cv2.findContours(c_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # get matching points
    pad = 4
    for each in cloth_contours:
        i, j = each[0][0]
        x_cloth.append(i)
        y_cloth.append(j)

        if cv2.pointPolygonTest(mask_contours[0], (i + pad, j), True) > 0:  # inside and not too close to the contours
            x_smpl.append(i - pad)
            x_smpl_2nd.append(i + pad)
        else:
            x_smpl.append(i + pad)
            x_smpl_2nd.append(i - pad)

        if cv2.pointPolygonTest(mask_contours[0], (i, j + pad), True) > 0:  # inside and not too close to the contours
            y_smpl.append(j - pad)
            y_smpl_2nd.append(j + pad)
        else:
            y_smpl.append(j + pad)
            y_smpl_2nd.append(j - pad)

    # prepare source and target points
    control_points_cloth = np.vstack([x_cloth, y_cloth]).T
    control_points_smpl = np.vstack([x_smpl, y_smpl]).T
    control_points_smpl_2nd = np.vstack([x_smpl_2nd, y_smpl_2nd]).T
    cloth_shape = np.array(control_points_cloth, np.float32)
    body_shape = np.array(control_points_smpl, np.float32)
    body_shape_2nd = np.array(control_points_smpl_2nd, np.float32)
    cloth_shape = cloth_shape.reshape(1, -1, 2)
    body_shape = body_shape.reshape(1, -1, 2)
    body_shape_2nd = body_shape_2nd.reshape(1, -1, 2)

    matches = list()
    for i in range(len(x_cloth)):
        matches.append(cv2.DMatch(i, i, 0))

    # eroded clothing texture
    t1 = img_rgb * emask
    t1[t1 == 255] = 0

    # extended clothing textures
    warped_img = apply_tps(body_shape, cloth_shape, matches, t1)
    warped_img_2nd = apply_tps(body_shape_2nd, cloth_shape, matches, t1)

    # merge textures (eroded + extended boundary)
    t2 = warped_img * dbmask
    t2[t2 == 255] = 0
    t2[t1 > 0] = 0
    t3 = warped_img_2nd * dbmask
    t3[t3 == 255] = 0
    t3[t1 > 0] = 0

    dilated_cloth = cv2.add(t2, t3)
    exp_img = cv2.add(t1, dilated_cloth)

    # visualize result
    if viz:
        # plot figures:
        titles = ['Original Image', 'Eroded cloth', 'extra boundary', 'segmented (original)', 'Dilated cloth',
                  'Extrapolated']
        images = [img_rgb, eroded_cloth, dilated_boundary_mask, img_rgb * mask, dilated_cloth, exp_img]

        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i])
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()

    # save result
    if save:
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
