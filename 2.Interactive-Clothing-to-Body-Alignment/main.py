"""
# Project: 2D Cloth Matching Using OpenCV
# Author: Matiur Rahman Minar (minar09.bd@gmail.com), Heejune Ahn (heejune@seoultech.ac.kr)
# Date created: 2020-07-25
# Python version: 3.x
# Requirements: opencv-contrib-python, numpy, pandas
# Description: This program is to interactively and manually deform in-shop clothes according to SMPL body
#               templates for 2D matching
# Usage: Run main.py with in-shop cloth image and its binary mask location arguments,
#       e.g., 'python main.py 0_1.jpg 0_1.png'
"""

### IMPORT ###################################################################
import cv2  # Import the OpenCV library
import numpy as np  # Import Numpy library
import pandas as pd  # Import Pandas library
import sys  # Enables the passing of arguments
import arap  # Import ARAP deformation/visualization functions
import mesh_deformer  # Import ARAP python library

### GLOBAL VAR    ##############################################################

# CONSTANTs
WINDOW_NAME = 'Manual clothing matching'
# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255),
          'yellow': (0, 255, 255), 'magenta': (255, 0, 255),
          'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125),
          'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# VARs
# Images
cloth_image = None
gOriginalClothImg = None
gOriginalMaskImg = None
gInitialClothImg = None
gFinalDeformedClothImg = None
smpl_image = None
gSMPLContours = None
gClothContours = None
gOriginalClothContours = None

# Landmarks locations
gOriginalClothPts = None
gClothPts = None
gSMPLPts = None
# the landmark indices to use (e.g. only 3 used for Affine Transform)
gHandlePts = None
gRelocated = []

# User Interaction
full_window_image = None  # final view
gBaseWindowImage = None  # wo annotations of control markers
gLMousePressed = False
gRMousePressed = False
gSelControlVertex = -1
gTgtPos = (-1, -1)

# New class data structure for ARAP deformation
gInitialClothMesh = None
gDeformedClothMesh = None
gControlPtsTracks = None

# exception handling for opencv version 3 and 4
(cv_major, _, _) = cv2.__version__.split(".")
if cv_major != '4' and cv_major != '3':
    print('does not support opencv version')
    sys.exit()


# SMPL contours
def getSMPLContour(img):
    ret, thresh = cv2.threshold(cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY), 30, 255, 0)

    contours = None
    # get contours
    if cv_major == '4':
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # use only external contour points
    elif cv_major == '3':
        _, contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# Cloth contours
def getClothContour(img):
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255,
                                cv2.THRESH_BINARY_INV)  # background is white

    # get contours
    contours = None
    if cv_major == '4':
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    elif cv_major == '3':
        _, contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# get cloth mask contour
def getMaskContour(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours = None
    # get contours
    if cv_major == '4':
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # use only external contour points
    elif cv_major == '3':
        _, contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


#
# load images and landmarks
#
def loadData(input_img_name, input_mask_name):
    global cloth_image, smpl_image, full_window_image, gOriginalClothImg, gOriginalMaskImg
    global gSMPLContours, gClothContours, gOriginalClothContours

    # 1. the path of files
    input_name = input_img_name[:input_img_name.index(".")]
    output_csv = input_name + "_landmarks.csv"
    default_csv = "default_landmarks.csv"
    smpl_image_path = "template.png"

    # 2. Load the images and store into variables
    # -1 means load unchanged
    cloth_image = cv2.imread(input_img_name, -1)
    mask_image = cv2.imread(input_mask_name, 0)
    smpl_image = cv2.imread(smpl_image_path, -1)
    gOriginalClothImg = cloth_image.copy()
    gOriginalMaskImg = mask_image.copy()

    # 3. prepare
    gSMPLContours = getSMPLContour(smpl_image)
    # gClothContours = getClothContour(cloth_image)
    gClothContours = getMaskContour(mask_image)
    # gOriginalClothContours = getClothContour(cloth_image)
    gOriginalClothContours = getMaskContour(mask_image)

    full_window_image = np.zeros_like(smpl_image)
    loadLandmarks(output_csv, default_csv)


#  find the closest vertex in vs
#  return the index and distance^2  (when < 10pt*10pt)
#         -1                         o.w.
def findClosestLandmark(x, y, vs):
    min_dist, nHit = 10 * 10, -1

    # for i in range(vs.shape[0]):
    for i in range(len(vs)):
        # if i in gControlPts:  # only control points
        # x1, y1 = vs[i, 0], vs[i, 1]
        x1, y1 = vs[i][0], vs[i][1]
        dist2 = (x1 - x) * (x1 - x) + (y1 - y) * (y1 - y)
        # print(">> (", x, ",", y, ") vs (",  x1, ",", y1, ")")

        if dist2 < min_dist:
            nHit = i
            min_dist = dist2

    return nHit, min_dist


# mouse callback function on mouse click operations
def onMouseCallback(event, x, y, flags, param):
    global gLMousePressed, gRMousePressed
    global gSelControlVertex, gTgtPos  # , key_pressed, color
    global gHandlePts

    # L-button for choosing a source vertex
    if event == cv2.EVENT_LBUTTONDOWN:

        gLMousePressed = True
        # check location (x,y) and check any vertex is chosen
        nHit, dist2 = findClosestLandmark(x, y, gHandlePts)

        if nHit >= 0:
            print("selected vertex:", nHit, ":",
                  gHandlePts[nHit])

            gSelControlVertex = nHit
            gTgtPos = (-1, -1)  # invalidate the target location
            draw_selected_landmark()

    elif event == cv2.EVENT_LBUTTONUP:
        gLMousePressed = False

    # R-button for target position
    elif event == cv2.EVENT_RBUTTONDOWN:
        if gSelControlVertex != -1:
            gRMousePressed = True
            gTgtPos = (x, y)
            # set new control vertex position
            gControlPtsTracks.tgtPos[gSelControlVertex] = gTgtPos
            draw_target_landmark()

    # move the target position
    elif event == cv2.EVENT_MOUSEMOVE:
        if gRMousePressed:
            gTgtPos = (x, y)
            drawMatch()

    elif event == cv2.EVENT_RBUTTONUP:
        gRMousePressed = False


#
# Load landmarks for initial setup
#
def loadLandmarks(OUTPUT_CSV, DEFAULT_CSV):
    global gOriginalClothPts, gSMPLPts, gClothPts

    try:
        df = pd.read_csv(OUTPUT_CSV)
    except:
        df = pd.read_csv(DEFAULT_CSV)

    if df is None:
        return

    n = df.to_numpy()

    gClothPts = n[:, 0:3]
    gClothPts[:, 2] = 1  # homogeneous
    gOriginalClothPts = gClothPts.copy()  # original
    gSMPLPts = n[:, 3:6]
    gSMPLPts[:, 2] = 1  # homogeneous


'''

  RoughMatching 

  use 4 points for rough matching 
  use the shoulders and waists (some cloth is short and short sleeve) 

    9    10 

    5    6  

  @TODO


'''


# match the in-shop cloth roughly to the SMPL template using affine matching
def matchRoughly():
    global cloth_image, full_window_image, gInitialClothImg
    global gSMPLPts, gClothPts, gOriginalClothPts, gOriginalClothImg, gOriginalMaskImg
    global gClothContours

    c = []
    s = []
    # pos = [9, 10, 5]  # , 6]
    pos = [4, 5, 9]  # , 8]
    for i in pos:
        c.append([gOriginalClothPts[i, 0], gOriginalClothPts[i, 1]])
        s.append([gSMPLPts[i, 0], gSMPLPts[i, 1]])

    ptsSrc = np.float32(c)  # lmCloth[pos,:])
    ptsDst = np.float32(s)  # lmSMPL[pos,:])

    M = cv2.getAffineTransform(ptsSrc, ptsDst)  # matrix 계산

    height, width = full_window_image.shape[0:2]
    cloth_image = cv2.warpAffine(gOriginalClothImg, M, (width, height))
    mask_image = cv2.warpAffine(gOriginalMaskImg, M, (width, height))
    gClothContours = getMaskContour(mask_image)

    ####### FOR DEMO: use only 3 pts for Affine transform ###################
    gClothPts = np.matmul(gOriginalClothPts, M.transpose()
                          )  # transform the points

    # save initial cloth image
    gInitialClothImg = cloth_image.copy()


#
# deform Cloth using the modified landmark movement
#
# Iterative deformation with ARAP
#
def deformCloth():
    global cloth_image, gFinalDeformedClothImg
    global gSelControlVertex, gTgtPos, gHandlePts
    global gDeformedClothMesh, gControlPtsTracks

    ##############################################
    # As rigid as possible algorithm
    ##############################################
    gDeformedClothMesh = arap.setupDeformation(
        gInitialClothMesh, gControlPtsTracks)

    # apply arap deformation
    # original to deform with list of h-vertexes
    mesh_deformer.arapDeform(
        gInitialClothMesh, gDeformedClothMesh, gControlPtsTracks)

    # apply ARAP for cloth deformation (deform the initially set cloth image)
    cloth_to_deform = gInitialClothImg.copy()
    deformed_cloth = arap.arap_transform(
        cloth_to_deform, gInitialClothMesh, gDeformedClothMesh)
    gFinalDeformedClothImg = deformed_cloth.copy()
    cloth_image = deformed_cloth.copy()

    ##############################################
    # cloth is deformed so.....
    ##############################################
    gBaseWindowImage = cloth_image.copy()
    arap.draw_cloth_mesh(gBaseWindowImage, gDeformedClothMesh)
    # @TODO
    # smpl contours @TODO not changing
    cv2.drawContours(gBaseWindowImage, gSMPLContours, -1, (0, 255, 0), 1)

    # show (after drawing markers)
    drawMatch()  # initial output

    ##############################################
    # update variables for next deformation
    ##############################################
    gSelControlVertex = -1
    gTgtPos = (-1, -1)
    gHandlePts = gControlPtsTracks.tgtPos.copy()  # new handle points for visualization
    for i, _ in enumerate(gRelocated):
        gRelocated[i] = False


'''
  overlay cloth and smpl silhouette

'''


# display updated cloth and mesh
def drawMatch(show_initial=False):
    # add annotations
    # draw mesh and contours
    gBaseWindowImage = cloth_image.copy()  # deformed cloth
    if show_initial:
        gBaseWindowImage = gInitialClothImg.copy()  # show initial roughly matched cloth to deform again

    # draw cloth mesh
    if not show_initial and gDeformedClothMesh:
        arap.draw_cloth_mesh(gBaseWindowImage, gDeformedClothMesh)
    else:
        arap.draw_cloth_mesh(gBaseWindowImage, gInitialClothMesh)

    cv2.drawContours(gBaseWindowImage, gSMPLContours, -
    1, (0, 255, 0), 1)  # smpl contours

    full_window_image[:, :, :] = gBaseWindowImage[:, :, :]


# 3. draw current landmarks
def draw_selected_landmark(show_initial=False):
    radius = 3
    for i in range(len(gControlPtsTracks.indices)):
        color = colors['blue']
        if i == gSelControlVertex:
            color = colors['red']

        x, y = int(gControlPtsTracks.tgtPos[i][0]), int(
            gControlPtsTracks.tgtPos[i][1])
        if show_initial:
            x, y = int(gControlPtsTracks.srcPos[i][0]), int(
                gControlPtsTracks.srcPos[i][1])

        cv2.circle(full_window_image, (x, y),
                   radius=radius, color=color, thickness=-1)


# 4. draw the moving landmarks
def draw_target_landmark():
    global gHandlePts
    radius = 3
    if gSelControlVertex != -1 and gTgtPos[0] != -1 and gTgtPos[1] != -1:
        if not gRelocated[gSelControlVertex]:  # check if already relocated after last deformation
            cv2.circle(full_window_image, gTgtPos, radius, (255, 255, 255), 1)  # -1: filled
            x, y = int(gHandlePts[gSelControlVertex][0]), int(
                gHandlePts[gSelControlVertex][1])
            cv2.arrowedLine(full_window_image, (x, y), gTgtPos,
                            (0, 255, 255), 1)  # draw arrow
            gRelocated[gSelControlVertex] = True


def main():
    # Get the argument and image names and load data
    input_cloth_name = sys.argv[1]     # get clothing image location
    input_mask_name = sys.argv[2]     # get clothing image location
    warped_cloth_name = input_cloth_name[:-4] + "_warped.png"
    print(warped_cloth_name)
    loadData(input_cloth_name, input_mask_name)

    # global variables
    global gBaseWindowImage
    global gInitialClothMesh, gDeformedClothMesh
    global gControlPtsTracks, gHandlePts

    print("In-shop cloth to Standard Model Matching!\n")
    print("Click blue points on the screen to change that point...\n")

    # We create a named window where the mouse callback will be established
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # to apply zoomed screen
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO,
                          cv2.WINDOW_KEEPRATIO)  # bigger screen

    # first initial match
    matchRoughly()  # initial match

    # Build mesh
    gInitialClothMesh, gControlPtsTracks = arap.create_cloth_mesh(
        cloth_image, gClothContours)
    gHandlePts = gControlPtsTracks.srcPos.copy()
    for i in gHandlePts:
        gRelocated.append(False)

    # show (after drawing markers)
    drawMatch()  # initial output
    draw_selected_landmark()

    ###############################################################################
    # User interaction
    ###############################################################################
    # We set the mouse callback function to 'draw_circle':
    cv2.setMouseCallback(WINDOW_NAME, onMouseCallback)

    while True:
        # Show image <WINDOW_NAME>:
        cv2.imshow(WINDOW_NAME, full_window_image)

        # Continue until 'q' is pressed:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            # break loop and quit program to save deformed cloth
            break
        elif k == ord('0'):  # Update using interactive
            if gSelControlVertex != -1:
                deformCloth()  # deform
                drawMatch()  # update content
                draw_selected_landmark()
        elif k == ord('u'):  # undo and revert to initial matched cloth
            # show and go back to initially matched cloth
            gControlPtsTracks.tgtPos = gControlPtsTracks.srcPos.copy()
            drawMatch(show_initial=True)  # update content
            draw_selected_landmark(show_initial=True)

    cv2.destroyAllWindows()

    # Save 2D matched/warped cloth with white Background
    if gFinalDeformedClothImg is not None:
        final_image = gFinalDeformedClothImg
    else:
        final_image = gInitialClothImg

    final_image[final_image <= 0] = 255
    cv2.imwrite(warped_cloth_name, final_image)


if __name__ == '__main__':

    # Check command arguments
    if len(sys.argv) != 3:
        print('Usage for manual cloth matching: %s cloth_path cloth_mask_path' %
              sys.argv[0])
        exit()

    main()  # Run the interactive GUI
