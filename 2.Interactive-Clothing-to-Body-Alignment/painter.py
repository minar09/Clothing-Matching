"""
# Project: 2D Cloth Matching Using OpenCV
# Author: Heejune Ahn (heejune@snut.ac.kr), Matiur Rahman Minar (minar09.bd@gmail.com)
# Date created: 2020-07-27
# Python version: 3.x
# Requirements: numpy
# Description: This program visualizes mesh triangles for As-Rigid-As-Possible (ARAP) transformation on the cloth image
"""

import cv2


# To check OpenCV D-triangulation
def draw_delaunay2(img, mesh, color):

    thickness = 1
    line_type = 8
    # line_type = cv2.LINE_AA

    for tidx in range(mesh.triangles.shape[0]):

        # 1. vertex indices for a triangle
        idx1 = mesh.triangles[tidx, 0]
        idx2 = mesh.triangles[tidx, 1]
        idx3 = mesh.triangles[tidx, 2]

        # 2. pts for the vertices
        pt1 = (mesh.vertices[idx1, 0], mesh.vertices[idx1, 1])
        pt2 = (mesh.vertices[idx2, 0], mesh.vertices[idx2, 1])
        pt3 = (mesh.vertices[idx3, 0], mesh.vertices[idx3, 1])

        # 3 line with 3 pts
        cv2.line(img, pt1, pt2, color, thickness, line_type)
        cv2.line(img, pt2, pt3, color, thickness, line_type)
        cv2.line(img, pt3, pt1, color, thickness, line_type)
