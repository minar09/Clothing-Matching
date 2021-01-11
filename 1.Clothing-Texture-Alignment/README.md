# Clothing Texture Alignment (Boundary Extrapolation)
Clothing texture alignment or boundary extrapolation for matching the extended binary masks/silhouettes and hiding the artifacts generated in 3D clothing reconstruction.

## 1) Texture alignment using TPS transformation
Steps:
- Mask dilation (for extrapolating out of boundary)
- Get contours and matching points (padded points outside of boundary for extrapolation)
- Apply TPS transformation for extending textures
- Merge original eroded clothing texture with the extended parts


## 2) Texture alignment using dilation operation
Steps:
- Mask dilation (for extrapolating out of boundary)
- Apply gray dilation separately on each channel of the clothing image, then merge
- Merge original eroded clothing texture with the dilated parts