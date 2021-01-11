# Interactive Clothing to Body Alignment (Manual)
An user interactive tool for matching/aligning/fitting clothing manually to the body (standard), using the as-rigid-as-possible (ARAP) algorithm.

## Steps of implementation:
- Load clothing image and mask, standard body image/silhouette (target), and default landmarks (if available)
- Match clothing to the body roughly (using default landmarks)
- Create clothing mesh from the clothing contour, and add control/handle vertices periodically as the source points
- Get modified locations of the control/handle vertices from user interaction as the target points
- Apply As-Rigid-As-Possible (ARAP) transformation from source to target for alignment/matching

## Usage instructions (for interactive matching by user):
- Run `python main.py <cloth_path> <cloth_mask_path>`
- Left-Click on the blue circled points to select the control point (Mouse left-click)
- Right-Click anywhere after selecting a control point, to change/relocate that point
- Selection and relocating multiple control points before applying deformation is applicable
- After updating the control points press ‘0’ (zero key) to apply As-Rigid-As-Possible (ARAP) deformation on the cloth image
- Selecting/changing control points and cloth deformation can be done iteratively (Press ‘u’ key to reset)
- Press ‘q’ key to quit the program
