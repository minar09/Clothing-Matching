# Clothing Masking (Binary Silhouettes)
Tools to generate better fitting and proper binary masks/silhouettes from clothing images.

## 1) Masking with GrabCut using priors

### Steps of implementation:
- Binary thresholding (to get the primary silhouette/mask)
- Filling operation (Floodfilling, to fill the big holes in the mask)
- Opening operation (for salt noise removal)
- GrabCut priors defining (absolute_foreground, probable_foreground, absolute_background, probable_background)
- GrabCut masking

## 2) Masking with binary thresholding

### Steps of implementation:
- Binary thresholding (to get the primary silhouette/mask)
- Filling operation (Floodfilling, to fill the big holes in the mask)
- Opening operation (for salt noise removal)
- Erosion operation (for thinning out boundary)
