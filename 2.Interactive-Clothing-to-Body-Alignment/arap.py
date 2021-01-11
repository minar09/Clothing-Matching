"""
# Project: 2D Cloth Matching Using OpenCV
# Author: Heejune Ahn (heejune@snut.ac.kr), Matiur Rahman Minar (minar09.bd@gmail.com)
# Date created: 2020-07-27
# Python version: 3.x
# Requirements: numpy, opencv-contrib-python, scikit-image, sortedcontainers
# Description: This program applies As-Rigid-As-Possible (ARAP) transformation on the cloth image
"""

import cv2
import painter
import numpy as np
import mesh_deformer
from sortedcontainers import SortedList
from skimage.transform import PiecewiseAffineTransform, warp


#
# build a triangle mesh for cloth area using contours
#
#  vertices:
#  1) the sampled contours (less than 20)
#  2)
#
#  control vertices: only vertex points
#
# cloth_img:  cloth rgb image
# cloth_contours:  the contours from cv2.findcountours
#
#
#  o_mesh : the original mesh vertices
#  handle : the control vertex indices
#  ##triimg : mesh image with triangles
#

def create_cloth_mesh(cloth_img, cloth_contours):

    points = []

    height, width, _ = cloth_img.shape

    # 1.1 get the biggest/external contour
    maxcontouridx = 0
    maxcontourlen = 0
    for i in range(len(cloth_contours)):
        if maxcontourlen < len(cloth_contours[i]):
            maxcontourlen = len(cloth_contours[i])
            maxcontouridx = i
    max_contour = cloth_contours[maxcontouridx]

    # get all mesh points/vertices and handle points

    # note control vertices
    handle_v_list = []

    # 1.2 add sampled points from contour
    # seglen = maxcontourlen//20
    seglen = maxcontourlen//30

    vidx = 0
    for ind, each in enumerate(max_contour):

        if ind % seglen == 0 and ind > 0 and not _checkCloseToHandles(handle_v_list, points, each[0]):
            if vidx not in handle_v_list:
                handle_v_list.append(vidx)  # now we add only contours for handles
            points.append(tuple(each[0]))   # add mesh vertices also
            vidx = vidx + 1
        else:
            try:
                # check angles of the points, take acute or smaller obtuse angles for adding to control points
                this_angle = get_angle(max_contour[ind - 5][0], max_contour[ind][0], max_contour[ind + 5][0])
                if this_angle < 150 and not _checkCloseToHandles(handle_v_list, points, each[0]):
                    # if this_angle < 150:
                    if vidx not in handle_v_list:
                        handle_v_list.append(vidx)  # now we add only contours for handles
                    points.append(tuple(each[0]))  # add mesh vertices also
                    vidx = vidx + 1
            except Exception as err:
                print(err)

    # 2. the bounding box

    # get minimum and maximum of the contour
    mc_x = max_contour[:, :, 0]
    mc_y = max_contour[:, :, 1]
    xmin = min(mc_x)[0]
    ymin = min(mc_y)[0]
    xmax = max(mc_x)[0]
    ymax = max(mc_y)[0]

    seglen = (xmax - xmin)//10  # 10  # 20
    # add points from inside cloth
    for _y in range(int((ymax - ymin) / seglen)):
        for _x in range(int((xmax - xmin) / seglen)):
            x, y = xmin + seglen * _x, ymin + seglen * _y  # bug fixed 2020. 8. 16

            if ymin <= y <= ymax and xmin <= x <= xmax:
                # >= 0:  # check if inside cloth contour
                dist = cv2.pointPolygonTest(max_contour, (x, y), True)
                if dist > 0 and dist > seglen/4:  # inside and not too close to the contours
                    points.append((x, y))

    # 3 list to numpy array @TODO
    o_vertices = np.asarray(points)
    o_handles = np.asarray(handle_v_list)

    # 4 build triangles
    # @Note: now we generate rectangle mesh only, so do not need to use Subdiv2D
    rect = (xmin, ymin, xmax, ymax)
    # @TODO Do we need to use opencv ? We could build more easily
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    # why we get points for minus coordinates
    triangleList = subdiv.getTriangleList()

    # 5 build triangles (of indices of vertices) from point locations
    triangles = np.zeros((len(triangleList), 3), dtype=int)

    tidx = 0
    for t in triangleList:

        x, y = t[0], t[1]
        if (xmin > x or x > xmax) or (ymin > y or y > ymax):  # often subdiv2d gives out of rectangle
            continue
        x, y = t[2], t[3]
        if (xmin > x or x > xmax) or (ymin > y or y > ymax):  # often subdiv2d gives out of rectangle
            continue
        x, y = t[4], t[5]
        if (xmin > x or x > xmax) or (ymin > y or y > ymax):  # often subdiv2d gives out of rectangle
            continue

        idx0 = _findNearestinMesh(o_vertices, (t[0], t[1]))
        idx1 = _findNearestinMesh(o_vertices, (t[2], t[3]))
        idx2 = _findNearestinMesh(o_vertices, (t[4], t[5]))

        # get the triangle center
        if False:
            centerX = (points[idx0][0] + points[idx1][0] + points[idx2][0]) / 3
            centerY = (points[idx0][1] + points[idx1][1] + points[idx2][1]) / 3

            # check if inside cloth contour
            if cv2.pointPolygonTest(max_contour, (centerX, centerY), True) >= 0:
                triangles[tidx] = (idx0, idx1, idx2)
                tidx = tidx + 1
        else:
            triangles[tidx] = (idx0, idx1, idx2)
            tidx = tidx + 1

    triangles = np.resize(triangles, (tidx, 3))  # remove triangle out of cloth

    # 3. Finally create meshes and handle points objects
    o_mesh = TriangleMesh(o_vertices, triangles, o_handles)
    handle_tracker = ControlPtsTrack(handle_v_list)

    for each in handle_v_list:
        handle_tracker.srcPos.append(o_mesh.vertices[each, :])

    handle_tracker.tgtPos = handle_tracker.srcPos.copy()

    return o_mesh, handle_tracker


def get_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


'''
   
    draw cloth mesh 
   
'''


def draw_cloth_mesh(tgtImg, mesh, color=(255, 0, 0)):

    height, width, _ = tgtImg.shape

    # triangleList = subdiv.getTriangleList()
    # @TODO, even though we remvove the traingles outside, still the area should be covered. O.W. it has problem.
    # painter.draw_delaunay2(triimg, triangles, (0, 0, 255), max_contour)  # outside triangles should be removed.
    # outside triangles should be removed.
    painter.draw_delaunay2(tgtImg, mesh, color)


#
# return the closest index
# TODO: try to use sub2Div  or any other library to do this.
#
def _findNearestinMesh(vertices, point):
    min_value, idx = 1000000000.0, -1
    nVerts = len(vertices)
    for i in range(nVerts):
        x1, y1 = vertices[i, 0], vertices[i, 1]
        x, y = point[0], point[1]

        dist = (x1 - x) * (x1 - x) + (y1 - y) * (y1 - y)
        if dist < min_value:
            idx = i
            min_value = dist
    return idx


# check whether the point is too close to any handle points
def _checkCloseToHandles(handles, vertices, point):
    threshold = 64  # 100
    for i in handles:
        x1, y1 = vertices[i][0], vertices[i][1]
        x, y = point[0], point[1]

        dist = (x1 - x) * (x1 - x) + (y1 - y) * (y1 - y)
        if dist < threshold:
            return True
    return False


def arap_transform(cloth, gOriginalMesh, gDeformedMesh):

    pwtform = PiecewiseAffineTransform()
    pwtform.estimate(gDeformedMesh.vertices, gOriginalMesh.vertices)
    warpedUpperClothfloat = warp(
        cloth, pwtform, output_shape=cloth.shape)

    # 4.4 convert type from float64 to uint8
    warpedUpperClothfloat = 255 * warpedUpperClothfloat  # Now scale by 255
    warpedUpperCloth = warpedUpperClothfloat.astype(np.uint8)

    return warpedUpperCloth


########################################################################################
# Test case generation
#
# simple squared mesh is used
#
########################################################################################

# construct a squared mesh (a special type)
def MakeSampleMesh(nRowLen, offset=0.0, step=1.0):

    # 1.vertices and triangle space
    vertices = np.zeros((nRowLen * nRowLen, 2), dtype=float)
    triangles = np.zeros(((nRowLen - 1) * (nRowLen - 1) * 2, 3), dtype=int)

    # 2. fill values
    xStep = step
    yStep = step
    xOffset = offset
    yOffset = offset

    # 2.1 positions of vertices
    index = 0
    for yi in range(nRowLen):
        y = yOffset + yi * yStep
        for xi in range(nRowLen):
            x = xOffset + xi * xStep
            vertices[index] = (x, y)
            index += 1

    # 2.2 vertices of triangle
    index = 0
    for yi in range(nRowLen - 1):
        nRow1 = yi * nRowLen
        nRow2 = (yi + 1) * nRowLen
        for xi in range(nRowLen - 1):
            triangles[index] = (nRow1 + xi, nRow2 + xi + 1, nRow1 + xi + 1)
            index += 1
            triangles[index] = (nRow1 + xi, nRow2 + xi, nRow2 + xi + 1)
            index += 1

    # 3. construct triangle mesh (cannot change it)
    mesh = TriangleMesh(vertices, triangles)

    return mesh  # this will be the original mesh-triangle


#
# set deformed mesh with constraints
#
# return initial mesh to deform and handles
#
#  upper_joint_position = joints[uppernum-1]
#    model_joint_position = joints[modelnum-1]
def setupDeformation(o_mesh, handle_tracker):

    # 1. clone n_mesh from o_mesh
    n_vertices = np.array(o_mesh.vertices, copy=True)
    n_triangles = np.array(o_mesh.triangles, copy=True)
    n_mesh = TriangleMesh(n_vertices, n_triangles, handle_tracker.indices)

    for i in range(len(handle_tracker.indices)):
        vidx = handle_tracker.indices[i]
        n_mesh.vertices[vidx, :] = handle_tracker.tgtPos[i]

    return n_mesh


##############################################################################
# class declarations
##############################################################################

# mesh of triangles
#
# @WRONG: this impl is only work for rectangles
#
# originally it should be extensible, adding new triangles and vertices
# but since we do not change them, we will set numpy arrays for this.
#
class TriangleMesh:
    def __init__(self, vertices, triangles, handles):
        self.vertices = vertices   # n by 2 array (x,y)
        self.triangles = triangles  # m by 3 array (vertex id)
        self.handles = handles  # c by 1 array (vertex id)


#
# data structure to keep track of location changes of control points
#
# indices of control/handle vertices from the initial cloth mesh
# source coordinates of the control vertices from the initial cloth mesh
# new target coordinates of the control vertices from the interactive GUI by mouse clicks
#
class ControlPtsTrack:
    def __init__(self, indices):
        # n by 1 array (vertex id in the mesh vertices)
        self.indices = indices
        self.srcPos = []        # s by 2 array (x, y)
        self.tgtPos = []        # t by 2 array (x, y)
