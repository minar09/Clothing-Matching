import numpy as np
import vectormath as vmath
from sortedcontainers import SortedList

# constants
INFINITY = 2 ** 63  # integer max (python 3 has no bound)
DEBUG = False


############################################################################
# sub classes for algorithm
#
############################################################################
# sub triangle data (vertex indexes, coordinates, scales, precomputed Matrix)
class Triangle:
    def __init__(self, v1, v2, v3):  # 3 vertices for a trangle
        self.nVerts = [v1, v2, v3]
        self.vTriCoords = []  # 2D position (x,y)
        self.vScaled = np.zeros((3, 2), dtype=float)  # un-scaled triangle
        # GMatrix: pre-computed matrices for triangle scaling step
        self.mF = self.mC = [[]]


# simply 2D coordinate
# class Vertex:
#	def __init__(self, x, y):
#		self.x, self.y = x, y

class Constraint:
    def __init__(self, nVertex, vec):
        self.nVertex = nVertex
        self.vConstrainedPos = vec

    def __lt__(self, other):
        return self.nVertex < other.nVertex


# LU-decomp, matrix and pivot
class LUData:  # information of LU decompositions
    def __init__(self, matrix, vPivots):
        self.mLU = matrix
        self.vPivots = vPivots


##############################################################################
# global variables : m is member variable
# @TODO make it as a class
###############################################################################
m_bSetupValid = None
m_mFirstMatrix = None  # G' matrix
m_vConstraints = SortedList()

m_vInitialVerts = []  # initial positions of points
m_vDeformedVerts = []  # current deformed positions of points
m_vTriangles = []  # contains deformed triangles

m_vVertexMap = []  # m_vVertexMap
m_mHXPrime, m_mHYPrime = None, None  # m_mHXPrime, m_mHYPrime
m_mDX, m_mDY = None, None  # m_mDX, m_mDY
m_mLUDecompX, m_mLUDecompY = None, None  # m_mLUDecompX, m_mLUDecompY


# functions
def Error():
    print("ERROR")
    exit()


def _invalidateSetup():
    # global m_bSetupValid
    m_bSetupValid = False


def _getInitialVert(nVert, Verts):
    ret = vmath.Vector2(float(Verts[nVert][0]), float(Verts[nVert][1]))
    return ret


def _normalize(vec):
    l = vec.length
    return vec / l


def _squared_length(vec):
    return vec.length * vec.length


def _extractSubMatrix(mFrom, nRowOffset, nColOffset, row, col):
    ret = np.zeros((row, col), dtype=float)
    for i in range(row):
        for j in range(col):
            ret[i][j] = mFrom[i + nRowOffset][j + nColOffset]
    return ret


####################################################################
# Static Matrices
#
####################################################################
#
# 1. scale-free transfrom matrix
#
def _precomputeOrientationMatrix():
    if DEBUG:
        print("\nprecomputeOrientationMatrix()")
    # m_vConstraints = shared.m_vConstraints

    # put constraints into vConstraintVec
    vConstraintVec = []
    for i in range(len(m_vConstraints)):
        vConstraintVec.append(m_vConstraints[i])

    # resize matrix and clear to zero
    nVerts = len(m_vDeformedVerts)
    G = np.zeros((nVerts * 2, nVerts * 2), dtype=float)  # G' matrix in eqn (8)

    nConstraints = len(vConstraintVec)
    nFreeVerts = nVerts - nConstraints
    if DEBUG:
        print("nConstraints =", nConstraints, ", Free =", nFreeVerts)

    # figure out vertices ordering. First free vertices and then constraints
    nRow = 0

    m_vVertexMap = np.zeros(nVerts, dtype=int)
    for i in range(nVerts):
        c = Constraint(i, [0.0, 0.0])
        if m_vConstraints.count(c) > 0:
            continue
        m_vVertexMap[i] = nRow
        nRow += 1

    if nRow != nFreeVerts:
        Error()

    for i in range(nConstraints):
        m_vVertexMap[vConstraintVec[i].nVertex] = nRow
        nRow += 1

    if nRow != nVerts:
        Error()

    # test vectors
    gUTest = np.zeros(nVerts * 2, dtype=float)
    for i in range(nVerts):
        c = Constraint(i, [0.0, 0.0])
        if m_vConstraints.count(c) > 0:
            continue
        Row = m_vVertexMap[i]
        gUTest[Row * 2] = m_vInitialVerts[i][0]
        gUTest[Row * 2 + 1] = m_vInitialVerts[i][1]

    for i in range(nConstraints):
        Row = m_vVertexMap[vConstraintVec[i].nVertex]
        gUTest[Row * 2] = vConstraintVec[i].vConstrainedPos[0]
        gUTest[Row * 2 + 1] = vConstraintVec[i].vConstrainedPos[1]

    # fill matrix
    line = 1
    nTri = len(m_vTriangles)
    for i in range(nTri):
        t = m_vTriangles[i]
        fTriSumErr = 0  # Error of the triangles

        for j in range(3):
            fTriErr = 0  # Error of the subtriangles
            n0x = 2 * m_vVertexMap[t.nVerts[j]]
            n0y = n0x + 1
            n1x = 2 * m_vVertexMap[t.nVerts[(j + 1) % 3]]
            n1y = n1x + 1
            n2x = 2 * m_vVertexMap[t.nVerts[(j + 2) % 3]]
            n2y = n2x + 1
            x, y = t.vTriCoords[j][0], t.vTriCoords[j][1]

            v0 = vmath.Vector2(float(gUTest[n0x]), float(gUTest[n0y]))
            v1 = vmath.Vector2(float(gUTest[n1x]), float(gUTest[n1y]))
            v2 = vmath.Vector2(float(gUTest[n2x]), float(gUTest[n2y]))
            v01 = v1 - v0
            v01Perp = vmath.Vector2(v01[1], -v01[0])
            vTest = v0 + x * v01 + y * v01Perp
            fDist = (vTest - v2).dot(vTest - v2)

            """
            add line = 1 for debug
            print("debug line", line, ":", x, y)
            print("debug line", line, ":", v0[0], v0[1])
            print("debug line", line, ":", v1[0], v1[1])
            print("debug line", line, ":", v2[0], v2[1])
            print("debug line", line, ":", v01[0], v01[1])
            print("debug line", line, ":", v01Perp[0], v01Perp[1])
            print("debug line", line, ":", vTest[0], vTest[1])
            line += 1

            if fDist > 0.0001:
                Error()
            """

            G[n0x][n0x] += 1 - 2 * x + x * x + y * y
            G[n0x][n1x] += 2 * x - 2 * x * x - 2 * y * y
            G[n0x][n1y] += 2 * y
            G[n0x][n2x] += -2 + 2 * x
            G[n0x][n2y] += -2 * y

            fTriErr += (1 - 2 * x + x * x + y * y) * gUTest[n0x] * gUTest[n0x]
            fTriErr += (2 * x - 2 * x * x - 2 * y * y) * \
                gUTest[n0x] * gUTest[n1x]
            fTriErr += (2 * y) * gUTest[n0x] * gUTest[n1y]
            fTriErr += (-2 + 2 * x) * gUTest[n0x] * gUTest[n2x]
            fTriErr += (-2 * y) * gUTest[n0x] * gUTest[n2y]

            G[n0y][n0y] += 1 - 2 * x + x * x + y * y
            G[n0y][n1x] += -2 * y
            G[n0y][n1y] += 2 * x - 2 * x * x - 2 * y * y
            G[n0y][n2x] += 2 * y
            G[n0y][n2y] += -2 + 2 * x

            fTriErr += (1 - 2 * x + x * x + y * y) * gUTest[n0y] * gUTest[n0y]
            fTriErr += (-2 * y) * gUTest[n0y] * gUTest[n1x]
            fTriErr += (2 * x - 2 * x * x - 2 * y * y) * \
                gUTest[n0y] * gUTest[n1y]
            fTriErr += (2 * y) * gUTest[n0y] * gUTest[n2x]
            fTriErr += (-2 + 2 * x) * gUTest[n0y] * gUTest[n2y]

            G[n1x][n1x] += x * x + y * y
            G[n1x][n2x] += -2 * x
            G[n1x][n2y] += 2 * y

            fTriErr += (x * x + y * y) * gUTest[n1x] * gUTest[n1x]
            fTriErr += (-2 * x) * gUTest[n1x] * gUTest[n2x]
            fTriErr += (2 * y) * gUTest[n1x] * gUTest[n2y]

            G[n1y][n1y] += x * x + y * y
            G[n1y][n2x] += -2 * y
            G[n1y][n2y] += -2 * x

            fTriErr += (x * x + y * y) * gUTest[n1y] * gUTest[n1y]
            fTriErr += (-2 * y) * gUTest[n1y] * gUTest[n2x]
            fTriErr += (-2 * x) * gUTest[n1y] * gUTest[n2y]

            G[n2x][n2x] += 1
            G[n2y][n2y] += 1

            fTriErr += gUTest[n2x] * gUTest[n2x] + gUTest[n2y] * gUTest[n2y]

            fTriSumErr += fTriErr

    gUTemp = np.matmul(G, gUTest)
    fSum = gUTemp.dot(gUTest)
    # print("(test) Residual =", fSum)

    # extract G00 matrix
    G00 = np.zeros((2 * nFreeVerts, 2 * nFreeVerts), dtype=float)
    dim = np.shape(G00)
    row, col = dim[0], dim[1]
    G00 = _extractSubMatrix(G, 0, 0, row, col)

    # extract G01 and G10 matrices
    G01 = np.zeros((2 * nFreeVerts, 2 * nConstraints), dtype=float)
    dim = np.shape(G01)
    row, col = dim[0], dim[1]
    G01 = _extractSubMatrix(G, 0, 2 * nFreeVerts, row, col)

    G10 = np.zeros((2 * nConstraints, 2 * nFreeVerts), dtype=float)
    dim = np.shape(G10)
    row, col = dim[0], dim[1]
    G10 = _extractSubMatrix(G, 2 * nFreeVerts, 0, row, col)

    # compute GPrime = G00 + Transpose(G00) and B = G01 + Transpose(G10) eqn (8)
    GPrime = G00 + np.transpose(G00)
    B = G01 + np.transpose(G10)

    # invert GPrime and final result = -GPrimeInverse * B
    GPrimeInverse = np.linalg.inv(GPrime)
    mFinal = np.matmul(GPrimeInverse, B)

    return -mFinal
    # checked: gUTest, m_vVertexMap, G, G00, G01, G10, GPrime, B, GPrimeInverse, mFinal


#
# LUDecompostion for Scale Matrix calculation
#
def _LUDecompose(mMatrix, vDecomp):  # return tuple(ifSquare, vDecomp)

    dim = np.shape(mMatrix)
    row, col = dim[0], dim[1]
    if row != col:
        return False, vDecomp

    # initialize vDecomp
    vDecomp = LUData(np.zeros((row, row), dtype=float), np.zeros(row, int))
    vPivots = vDecomp.vPivots  # need to assign value back
    mLUMatrix = vDecomp.mLU  # need to assign value back

    mLUMatrix = mMatrix

    # scaling of each row
    dRowSwaps, dTemp = 1, None
    vScale = np.zeros(row, dtype=float)
    for i in range(row):
        dLargest = 0.0
        for j in range(row):
            dTemp = abs(float(mLUMatrix[i][j]))
            if (dTemp > dLargest):
                dLargest = dTemp

        if dLargest == 0:
            return False, vDecomp
        vScale[i] = 1.0 / dLargest

    niMax = 0
    for j in range(row):
        for i in range(j):
            dSum = mLUMatrix[i][j]
            for k in range(i):
                dSum -= mLUMatrix[i][k] * mLUMatrix[k][j]
            mLUMatrix[i][j] = dSum

        dLargestPivot = 0.0
        for i in range(j, row):
            dSum = mLUMatrix[i][j]
            for k in range(j):
                dSum -= mLUMatrix[i][k] * mLUMatrix[k][j]
            mLUMatrix[i][j] = dSum
            dTemp = vScale[i] * abs(float(dSum))
            if dTemp > dLargestPivot:
                dLargestPivot = dTemp
                niMax = i

        if j != niMax:
            for k in range(row):
                dSwap = mLUMatrix[niMax][k]
                mLUMatrix[niMax][k] = mLUMatrix[j][k]
                mLUMatrix[j][k] = dSwap
            dRowSwaps = -dRowSwaps
            vScale[niMax] = vScale[j]

        vPivots[j] = niMax
        if mLUMatrix[j][j] == 0:
            mLUMatrix[j][j] = EPSILON

        if j != row - 1:
            dScale = 1.0 / mLUMatrix[j][j]
            for i in range(j + 1, row):
                mLUMatrix[i][j] *= dScale

    vDecomp = LUData(mLUMatrix, vPivots)
    return True, vDecomp


#
# 2. scaling matrix
#
#
def _precomputeScalingMatrices(nTriangle):
    if DEBUG:
        print("precomputeScalingMatrices(", nTriangle, ")")

    t = m_vTriangles[nTriangle]
    t.mF = np.zeros((4, 4), dtype=float)
    t.mC = np.zeros((4, 6), dtype=float)

    # precompute coefficients
    x01 = t.vTriCoords[0][0]
    y01 = t.vTriCoords[0][1]
    x12 = t.vTriCoords[1][0]
    y12 = t.vTriCoords[1][1]
    x20 = t.vTriCoords[2][0]
    y20 = t.vTriCoords[2][1]

    k1 = x12 * y01 + (-1 + x01) * y12
    k2 = -x12 + x01 * x12 - y01 * y12
    k3 = -y01 + x20 * y01 + x01 * y20
    k4 = -y01 + x01 * y01 + x01 * y20
    k5 = -x01 + x01 * x20 - y01 * y20

    a = -1 + x01
    a1 = pow(-1 + x01, 2) + pow(y01, 2)
    a2 = pow(x01, 2) + pow(y01, 2)
    b = -1 + x20
    b1 = pow(-1 + x20, 2) + pow(y20, 2)
    c2 = pow(x12, 2) + pow(y12, 2)

    r1 = 1 + 2 * a * x12 + a1 * pow(x12, 2) - 2 * y01 * y12 + a1 * pow(y12, 2)
    r2 = -(b * x01) - b1 * pow(x01, 2) + y01 * (-(b1 * y01) + y20)
    r3 = -(a * x12) - a1 * pow(x12, 2) + y12 * (y01 - a1 * y12)
    r5 = a * x01 + pow(y01, 2)
    r6 = -(b * y01) - x01 * y20
    r7 = 1 + 2 * b * x01 + b1 * pow(x01, 2) + b1 * pow(y01, 2) - 2 * y01 * y20

    # setup F matrix
    t.mF[0][0] = 2 * a1 + 2 * a1 * c2 + 2 * r7
    t.mF[0][1] = 0
    t.mF[0][2] = 2 * r2 + 2 * r3 - 2 * r5
    t.mF[0][3] = 2 * k1 + 2 * r6 + 2 * y01

    t.mF[1][0] = 0
    t.mF[1][1] = 2 * a1 + 2 * a1 * c2 + 2 * r7
    t.mF[1][2] = -2 * k1 + 2 * k3 - 2 * y01
    t.mF[1][3] = 2 * r2 + 2 * r3 - 2 * r5

    t.mF[2][0] = 2 * r2 + 2 * r3 - 2 * r5
    t.mF[2][1] = -2 * k1 + 2 * k3 - 2 * y01
    t.mF[2][2] = 2 * a2 + 2 * a2 * b1 + 2 * r1
    t.mF[2][3] = 0

    t.mF[3][0] = 2 * k1 - 2 * k3 + 2 * y01
    t.mF[3][1] = 2 * r2 + 2 * r3 - 2 * r5
    t.mF[3][2] = 0
    t.mF[3][3] = 2 * a2 + 2 * a2 * b1 + 2 * r1

    mFInverse = np.linalg.inv(t.mF)
    mFInverse *= -1.0
    t.mF = mFInverse

    # setup C matrix
    t.mC[0][0] = 2 * k2
    t.mC[0][1] = -2 * k1
    t.mC[0][2] = 2 * (-1 - k5)
    t.mC[0][3] = 2 * k3
    t.mC[0][4] = 2 * a
    t.mC[0][5] = -2 * y01

    t.mC[1][0] = 2 * k1
    t.mC[1][1] = 2 * k2
    t.mC[1][2] = -2 * k3
    t.mC[1][3] = 2 * (-1 - k5)
    t.mC[1][4] = 2 * y01
    t.mC[1][5] = 2 * a

    t.mC[2][0] = 2 * (-1 - k2)
    t.mC[2][1] = 2 * k1
    t.mC[2][2] = 2 * k5
    t.mC[2][3] = 2 * r6
    t.mC[2][4] = -2 * x01
    t.mC[2][5] = 2 * y01

    t.mC[3][0] = 2 * k1
    t.mC[3][1] = 2 * (-1 - k2)
    t.mC[3][2] = -2 * k3
    t.mC[3][3] = 2 * k5
    t.mC[3][4] = -2 * y01
    t.mC[3][5] = -2 * x01

    # np.set_printoptions(precision = 4, suppress = True)
    # print("t.mC:", t.mC)
    # print("t.mF:", t.mF)

    return t


#
# 3. Fitting Matrix
#
def _precomputeFittingMatrices():
    if DEBUG:
        print("precomputeFittingMatrices()")
    global m_mHXPrime, m_mHYPrime, m_mDX, m_mDY, m_vConstraints
    # put constraints into vConstraintVec
    vConstraintVec = []
    for i in range(len(m_vConstraints)):
        vConstraintVec.append(m_vConstraints[i])

    # resize matrix and clear to zero
    nVerts = len(m_vDeformedVerts)
    nConstraints = len(vConstraintVec)
    nFreeVerts = nVerts - nConstraints

    # figure out vertices ordering. First free vertices and then constraints
    nRow = 0
    global m_vVertexMap

    m_vVertexMap = np.zeros(nVerts, dtype=int)
    for i in range(nVerts):
        c = Constraint(i, [0.0, 0.0])
        if m_vConstraints.count(c) > 0:
            continue
        m_vVertexMap[i] = nRow
        nRow += 1

    if nRow != nFreeVerts:
        Error()

    for i in range(nConstraints):
        m_vVertexMap[vConstraintVec[i].nVertex] = nRow
        nRow += 1

    if nRow != nVerts:
        Error()

    # test vectors
    gUTestX = np.zeros(nVerts, dtype=float)
    gUTestY = np.zeros(nVerts, dtype=float)
    for i in range(nVerts):
        c = Constraint(i, [0.0, 0.0])
        if m_vConstraints.count(c) > 0:
            continue
        row = m_vVertexMap[i]
        gUTestX[row] = m_vInitialVerts[i][0]
        gUTestY[row] = m_vInitialVerts[i][1]

    for i in range(nConstraints):
        row = m_vVertexMap[vConstraintVec[i].nVertex]
        gUTestX[row] = vConstraintVec[i].vConstrainedPos[0]
        gUTestY[row] = vConstraintVec[i].vConstrainedPos[1]

    # construct Hy and Hx matrices
    mHX = np.zeros((nVerts, nVerts), dtype=float)
    mHY = np.zeros((nVerts, nVerts), dtype=float)

    nTri = len(m_vTriangles)
    for i in range(nTri):
        t = m_vTriangles[i]
        for j in range(3):
            nA, nB = m_vVertexMap[t.nVerts[j]
                                  ], m_vVertexMap[t.nVerts[(j + 1) % 3]]
            mHX[nA][nA] += 2
            mHX[nA][nB] += -2
            mHX[nB][nA] += -2
            mHX[nB][nB] += 2
            mHY[nA][nA] += 2
            mHY[nA][nB] += -2
            mHY[nB][nA] += -2
            mHY[nB][nB] += 2

    # extract HX00 and HY00 matrices
    mHX00 = np.zeros((nFreeVerts, nFreeVerts), dtype=float)
    mHY00 = np.zeros((nFreeVerts, nFreeVerts), dtype=float)

    dim = np.shape(mHX00)
    row, col = dim[0], dim[1]
    mHX00 = _extractSubMatrix(mHX, 0, 0, row, col)

    dim = np.shape(mHY00)
    row, col = dim[0], dim[1]
    mHY00 = _extractSubMatrix(mHY, 0, 0, row, col)

    # extract HX01 and HX10 matrices
    mHX01 = np.zeros((nFreeVerts, nConstraints), dtype=float)
    mHX10 = np.zeros((nConstraints, nFreeVerts), dtype=float)

    dim = np.shape(mHX01)
    row, col = dim[0], dim[1]
    mHX01 = _extractSubMatrix(mHX, 0, nFreeVerts, row, col)

    dim = np.shape(mHX10)
    row, col = dim[0], dim[1]
    mHX10 = _extractSubMatrix(mHX, nFreeVerts, 0, row, col)

    # extract HY01 and HY10 matrices
    mHY01 = np.zeros((nFreeVerts, nConstraints), dtype=float)
    mHY10 = np.zeros((nConstraints, nFreeVerts), dtype=float)

    dim = np.shape(mHY01)
    row, col = dim[0], dim[1]
    mHY01 = _extractSubMatrix(mHY, 0, nFreeVerts, row, col)

    dim = np.shape(mHY10)
    row, col = dim[0], dim[1]
    mHY10 = _extractSubMatrix(mHY, nFreeVerts, 0, row, col)

    # compute m_mHXPrime and m_mHYPrime in eqn (16)
    m_mHXPrime, m_mHYPrime = mHX00, mHY00
    m_mDX, m_mDY = mHX01, mHY01

    # precompute LU decompositions
    global m_mLUDecompX, m_mLUDecompY
    bResult, m_mLUDecompX = _LUDecompose(m_mHXPrime, m_mLUDecompX)
    if (not bResult):
        Error()

    bResult, m_mLUDecompY = _LUDecompose(m_mHYPrime, m_mLUDecompY)
    if (not bResult):
        Error()


#########################################################################
# Update Matrix and vertices
#########################################################################

# calculate scales for triangles using updated m_vDeformedVerts (Step 1)
# nTriangle : triangle index
def _updateScaledTriangle(nTriangle):
    if DEBUG:
        print("Update scaled triangle", nTriangle)

    t = m_vTriangles[nTriangle]  # need to return

    vDeformedV0 = m_vDeformedVerts[t.nVerts[0]]
    vDeformedV1 = m_vDeformedVerts[t.nVerts[1]]
    vDeformedV2 = m_vDeformedVerts[t.nVerts[2]]
    vDeformed = [vDeformedV0[0], vDeformedV0[1], vDeformedV1[0],
                 vDeformedV1[1], vDeformedV2[0], vDeformedV2[1]]

    if DEBUG:
        print("t.mC:", t.mC)
        print("t.mF:", t.mF)
    # print("m_vInitialVerts:", m_vInitialVerts)

    mCVec = np.matmul(t.mC, vDeformed)

    vSolution = np.matmul(t.mF, mCVec)

    vFitted0 = vmath.Vector2(float(vSolution[0]), float(vSolution[1]))
    vFitted1 = vmath.Vector2(float(vSolution[2]), float(vSolution[3]))

    x01, y01 = t.vTriCoords[0][0], t.vTriCoords[0][1]
    vFitted01 = vFitted1 - vFitted0
    vFitted01Perp = vmath.Vector2(vFitted01[1], -vFitted01[0])
    vFitted2 = vFitted0 + float(x01) * vFitted01 + float(y01) * vFitted01Perp

    vOrig0 = vmath.Vector2(float(m_vInitialVerts[t.nVerts[0]][0]), float(
        m_vInitialVerts[t.nVerts[0]][1]))
    vOrig1 = vmath.Vector2(float(m_vInitialVerts[t.nVerts[1]][0]), float(
        m_vInitialVerts[t.nVerts[1]][1]))
    fScale = (vOrig1 - vOrig0).length / vFitted01.length

    # print("vOrig0:", vOrig0 )
    # print("vOrig1:", vOrig1 )
    # print("vFitted01:", vFitted01)
    # print("length:", (vOrig1 - vOrig0).length, vFitted01.length)

    # now scale !
    # find center of mass
    vCentroid = vFitted0 + vFitted1 + vFitted2 / 3.0
    # convert to vectors, scale and restore
    vFitted0 -= vCentroid
    vFitted0 *= fScale
    vFitted0 += vCentroid
    vFitted1 -= vCentroid
    vFitted1 *= fScale
    vFitted1 += vCentroid
    vFitted2 -= vCentroid
    vFitted2 *= fScale
    vFitted2 += vCentroid

    t.vScaled[0], t.vScaled[1], t.vScaled[2] = vFitted0, vFitted1, vFitted2
    # print("updated scales:", vFitted0, vFitted1, vFitted2)
    return t


# LUback subsraction
def _LUBackSub(vDecomp, vRHS, vSolution):
    vPivots, mLUMatrix = vDecomp.vPivots, vDecomp.mLU  # need to assign back
    dim = np.shape(mLUMatrix)
    row, col = dim[0], dim[1]

    # there is a bug in LUBackSub function in C++ code, fixed here
    if (row != col) or (len(vPivots) != row):
        return False, vSolution

    for i in range(len(vRHS)):
        vSolution[i] = vRHS[i]

    nNonVanish = INFINITY
    for i in range(row):
        nPivot = vPivots[i]
        dSum = vSolution[nPivot]
        vSolution[nPivot] = vSolution[i]

        if nNonVanish != INFINITY:
            for j in range(nNonVanish, i):
                dSum -= mLUMatrix[i][j] * vSolution[j]
        elif dSum != 0:
            nNonVanish = i

        vSolution[i] = dSum

    for i in range(row - 1, -1, -1):
        dSum = vSolution[i]
        for j in range(i + 1, row):
            dSum -= mLUMatrix[i][j] * vSolution[j]
        vSolution[i] = dSum / mLUMatrix[i][i]

    return True, vSolution


# scale Fitting
def _applyFittingStep():
    global m_mLUDecompX, m_vDeformedVerts, m_vConstraints
    if DEBUG:
        print("\nApply Fitting step")

    vConstraintVec = []
    for i in range(len(m_vConstraints)):
        vConstraintVec.append(m_vConstraints[i])

    nVerts, nConstraints = len(m_vDeformedVerts), len(vConstraintVec)
    nFreeVerts = nVerts - nConstraints

    vFX = np.zeros(nVerts, dtype=float)
    vFY = np.zeros(nVerts, dtype=float)

    nTri = len(m_vTriangles)
    for i in range(nTri):
        t = m_vTriangles[i]

        for j in range(3):
            nA, nB = m_vVertexMap[t.nVerts[j]
                                  ], m_vVertexMap[t.nVerts[(j + 1) % 3]]

            vDeformedA, vDeformedB = t.vScaled[j], t.vScaled[(j + 1) % 3]

            vFX[nA] += -2 * vDeformedA[0] + 2 * vDeformedB[0]
            vFX[nB] += 2 * vDeformedA[0] - 2 * vDeformedB[0]

            vFY[nA] += -2 * vDeformedA[1] + 2 * vDeformedB[1]
            vFY[nB] += 2 * vDeformedA[1] - 2 * vDeformedB[1]

    # construct F0 vectors
    vF0X = np.zeros(nFreeVerts, dtype=float)
    vF0Y = np.zeros(nFreeVerts, dtype=float)
    for i in range(nFreeVerts):
        vF0X[i], vF0Y[i] = vFX[i], vFY[i]

    # construct Q vectors (constraints)
    vQX = np.zeros(nConstraints, dtype=float)
    vQY = np.zeros(nConstraints, dtype=float)
    for i in range(nConstraints):
        vQX[i], vQY[i] = vConstraintVec[i].vConstrainedPos[0], vConstraintVec[i].vConstrainedPos[1]

    # compute RHS for X
    vRHSX = np.matmul(m_mDX, vQX)
    vRHSX += vF0X
    vRHSX *= -1
    vSolutionX = np.zeros(nFreeVerts, dtype=float)

    bResult, vSolutionX = _LUBackSub(m_mLUDecompX, vRHSX, vSolutionX)
    if not bResult:
        Error()

    # compute RHS for Y
    vRHSY = np.matmul(m_mDY, vQY)
    vRHSY += vF0Y
    vRHSY *= -1
    vSolutionY = np.zeros(nFreeVerts, dtype=float)
    bResult, vSolutionY = _LUBackSub(m_mLUDecompY, vRHSY, vSolutionY)
    if not bResult:
        Error()

    # finish
    for i in range(nVerts):
        c = Constraint(i, [0.0, 0.0])
        if m_vConstraints.count(c) > 0:
            continue
        row = m_vVertexMap[i]
        m_vDeformedVerts[i][0] = float(vSolutionX[row])
        m_vDeformedVerts[i][1] = float(vSolutionY[row])

    # checked: mHX, mHY, mHX00, mHY00, mHX01, mHX10, mHY01, mHY10, m_mLUDecompX, m_mLUDecompY


#
# computed deformed mesh
#
def _validateDeformedMesh(bRigid):
    # global m_vTriangles, m_vDeformedVerts, m_vConstraints
    if DEBUG:
        print("validateDeformedMesh()........................")

    nConstraints = len(m_vConstraints)
    if nConstraints < 2:
        return

    # if not precomputed, compute the static matrix
    validateSetup()

    # compute the deformed Mesh
    vQ = np.zeros(nConstraints * 2, dtype=float)
    k = 0
    for i in range(len(m_vConstraints)):
        c = m_vConstraints[i]
        vQ[k * 2] = c.vConstrainedPos[0]
        vQ[k * 2 + 1] = c.vConstrainedPos[1]

        # print("c: k =", k, ", n =", c.nVertex, " ", c.vConstrainedPos[0], " ", c.vConstrainedPos[1])
        x, y = vQ[k * 2], vQ[k * 2 + 1]
        # print("vQ: k = ", k, ", ", x, " ", y)
        k += 1

    # step 1: computation using m_mFirstMatrix (G) precomputed in eqn (8)
    vU = np.matmul(m_mFirstMatrix, vQ)
    # print("vU", vU)

    nVerts = len(m_vDeformedVerts)
    for i in range(nVerts):
        c = Constraint(i, [0.0, 0.0])
        if m_vConstraints.count(c) > 0:
            continue

        nRow = m_vVertexMap[i]
        fX, fY = vU[nRow * 2], vU[nRow * 2 + 1]
        m_vDeformedVerts[i] = vmath.Vector2(float(fX), float(fY))

    if DEBUG:
        print("rigid: ", bRigid)
    # step 2: scale triangles
    if bRigid:
        nTris = len(m_vTriangles)
        for i in range(nTris):
            m_vTriangles[i] = _updateScaledTriangle(i)

        _applyFittingStep()  # bug fix


def _SetVertex(mesh, index, xy):
    mesh.vertices[index][0] = xy[0]
    mesh.vertices[index][1] = xy[1]


#
# 2 cases:
# 1) one constraint vertex are moving.
# 2) new constraint vertext added.
#
# @Question: why not separate them?
#
def _updateConstraint(cons):
    global m_vDeformedVerts, m_vConstraints
    if DEBUG:
        print("Update constraint:", cons.nVertex, ",",
              cons.vConstrainedPos[0], ",", cons.vConstrainedPos[1])

    # 버텍스 index 만 비교해야하는데, 좌표값까지 비교하고 있음.
    # print("m_vConstraints:", m_vConstraints)
    # for i in range(len( m_vConstraints)):
    #    print(" ==>", i, m_vConstraints[i])

    # print("m_vConstraints.count(cons):", m_vConstraints.count(cons))

    # case 1: existing constraint vertex
    # if the handle is already in m_vConstraints
    if m_vConstraints.count(cons) > 0:

        for i in range(len(m_vConstraints)):
            if m_vConstraints[i].nVertex == cons.nVertex:
                m_vConstraints[i].vConstrainedPos = cons.vConstrainedPos
        # to check here
        """if False:
                c = Constraint(cons.nVertex, cons.vConstrainedPos)
                m_vConstraints.remove(cons)
                m_vConstraints.add(c)
        else:
                t = m_vConstraints.index(cons)
                m_vConstraints[t].vConstrainedPos = cons.vConstrainedPos
        """

        # print("replaced: ") #, c.nVertex, ",", c.vConstrainedPos[0], ",", c.vConstrainedPos[1])

        m_vDeformedVerts[cons.nVertex] = cons.vConstrainedPos

    # 2. new constraints vertexes
    else:
        # print("new constraint:", cons.nVertex, ",", cons.vConstrainedPos[0], ",", cons.vConstrainedPos[1])
        m_vConstraints.add(cons)  # add a new contraint
        # update location
        m_vDeformedVerts[cons.nVertex] = cons.vConstrainedPos
        # set flag for re-computing Static Matrix
        # this is not to recompute the static Matrix  whenever new constrants are added
        _invalidateSetup()


########################################################################
# API
#
# How2use
# 1. initFromMesh(initialMesh)
# 2. setDeformedHandle(vertex)  # as many as # of constraints vertecies
# (3. validateSetup() # static matrix computation, if not called, step4 will based on flag)
# 4. updateMesh(deformedMesh, isRigid) # calculate and fill deformed mesh
#
########################################################################

#########################################################################
# 1. initialize mesh
# @TODO: inside of class constructor
#########################################################################
def initFromMesh(Mesh):
    global m_vDeformedVerts, m_vInitialVerts, m_vTriangles, m_vConstraints
    # global m_vConstraints
    if DEBUG:
        print("Initialized from Mesh")

    # m_vConstraints = SortedList()
    nVerts = len(Mesh.vertices)
    # print("Vertices in current mesh = ", nVerts)
    # print("Triangles in current mesh = ", len(Mesh.triangles))
    # print()

    # copy vertices
    m_vInitialVerts = np.array(Mesh.vertices, copy=True)
    m_vDeformedVerts = np.array(Mesh.vertices, copy=True)

    # copy triangles
    m_vTriangles = []
    for i in range(len(Mesh.triangles)):
        A, B, C = Mesh.triangles[i][0], Mesh.triangles[i][1], Mesh.triangles[i][2]
        m_vTriangles.append(Triangle(A, B, C))

    # set up triangle-local coordinate systems
    line = 1
    nTri = len(m_vTriangles)
    #print("nTri:", nTri)
    for i in range(nTri):
        tri = m_vTriangles[i]

        # input triangle check !! 2020. 8. 16 heejune
        if tri.nVerts[0] == tri.nVerts[1] or tri.nVerts[0] == tri.nVerts[2] or tri.nVerts[1] == tri.nVerts[2]:
            print("Got a wrong trangle at ", i,
                  " vertex indexes :", tri.nVerts)
            sys.exit()

        for j in range(3):
            n0, n1, n2 = j, (j + 1) % 3, (j + 2) % 3

            v0 = _getInitialVert(tri.nVerts[n0], m_vInitialVerts)
            v1 = _getInitialVert(tri.nVerts[n1], m_vInitialVerts)
            v2 = _getInitialVert(tri.nVerts[n2], m_vInitialVerts)

            # find coordinate system : eqn (1)
            v01 = v1 - v0  # vec from v0 to v1
            #print("triangle:", i, " j:",  j, "sqlen(v01):", _squared_length(v01), "v1:", v1, "v0:", v0)
            # some how traingle has same vertices index !!
            #print("triangle:", i,  "v0:", v0, "v1:", v1, "v2:", v2, " ",  tri.nVerts[n0], " ",  tri.nVerts[n1],  " ", tri.nVerts[n2])
            # if _squared_length(v01) < 0.00001:
            #    print("Too small and clos  se: ", " ",  tri.nVerts[n0], " ",  tri.nVerts[n1],  " ", tri.nVerts[n2])

            v01N = v01
            v01N = _normalize(v01N)
            v01Rot90 = vmath.Vector2(v01.y, -v01.x)
            v01Rot90N = v01Rot90
            v01Rot90N = _normalize(v01Rot90)

            # express v2 in coordinate system => eqn (1)
            vLocal = v2 - v0
            fX = float(vLocal.dot(v01)) / _squared_length(v01)  # x01
            fY = float(vLocal.dot(v01Rot90)) / _squared_length(v01Rot90)  # y01

            # sanity check, calculated v2 should be same as given v2
            v2test = v0 + fX * v01 + fY * v01Rot90
            fLen = (v2test - v2).length
            if fLen > 0.001:
                Error()

            m_vTriangles[i].vTriCoords.append(vmath.Vector2(fX, fY))
            # print("fX =", fX, ", fY = ", fY)


#########################################################################
# 2. add a constraints or change the location
# note: no calculation here, just update constraints
#########################################################################
def setDeformedHandle(indexHandle, vecHandle):
    c = Constraint(indexHandle, vecHandle)
    _updateConstraint(c)


########################################################################
#  (optinal explicit)
# 3. calcuate static Matrices
# called when constraint vertices is set
#########################################################################
def validateSetup():
    global m_bSetupValid, m_mFirstMatrix, m_vTriangles, m_vConstraints
    if DEBUG:
        print("\nValidate Setup: valid =", m_bSetupValid,
              ", size =", len(m_vConstraints))

    if (m_bSetupValid or len(m_vConstraints) < 2):
        return
    if DEBUG:
        print("Computing matrices for mesh with", len(
            m_vInitialVerts), "vertices... this might take a while\n")
    m_mFirstMatrix = _precomputeOrientationMatrix()

    # scale triangles
    nTris = len(m_vTriangles)
    for i in range(nTris):
        m_vTriangles[i] = _precomputeScalingMatrices(i)

    _precomputeFittingMatrices()

    # print("\nDone!")
    m_bSetupValid = True


########################################################################
# 4. request the updated deformation)
# Output: pMesh
########################################################################
def updateMesh(pMesh, bRigid):
    global m_vConstraints
    if DEBUG:
        print("UpdateDeformedMesh(): constraint size =", len(m_vConstraints))

    # 1. calculate deformed (**CORE *)
    _validateDeformedMesh(bRigid)

    # 2. depends on conditions, we should use either initial mesh information or deformed one
    vVerts = []
    if len(m_vConstraints) > 1:
        # make vVerts = m_vDeformedVerts
        for i in range(len(m_vDeformedVerts)):
            vVerts.append(m_vDeformedVerts[i])
    else:
        # make vVerts = m_vInitialVerts
        for i in range(len(m_vInitialVerts)):
            vVerts.append(m_vInitialVerts[i])

    # 3. fill the container of the caller
    nVerts = len(pMesh.vertices)
    for i in range(nVerts):
        vNewPos = vmath.Vector2(float(vVerts[i][0]), float(vVerts[i][1]))
        _SetVertex(pMesh, i, vNewPos)

    return pMesh


########################################################################
# All-in-one-API
#
# outmesh[constraints] = inmesh[constraints]
# calculate outmesh[~constraints]
########################################################################
def arapDeform(inmesh, outmesh, handle_tracker):
    # 1. create deformer
    initFromMesh(inmesh)
    # 2. set the constraints
    for i in range(len(handle_tracker.indices)):
        index = handle_tracker.indices[i]  # vertex id
        vVertex = outmesh.vertices[index]
        setDeformedHandle(index, vVertex)
    # 3. notify all constraints
    validateSetup()

    # 4. apply algorithm
    updateMesh(outmesh, bRigid=True)

    # clean up @TODO: make a class
    global m_bSetupValid, m_mFirstMatrix, m_vConstraints
    global m_vInitialVerts, m_vDeformedVerts, m_vTriangles
    global m_vVertexMap, m_mHXPrime, m_mHYPrime, m_mDX, m_mDY, m_mLUDecompX, m_mLUDecompY

    m_bSetupValid = None
    m_mFirstMatrix = None  # G' matrix
    m_vConstraints = SortedList()

    m_vInitialVerts = []  # initial positions of points
    m_vDeformedVerts = []  # current deformed positions of points
    m_vTriangles = []  # contains deformed triangles

    m_vVertexMap = []  # m_vVertexMap
    m_mHXPrime, m_mHYPrime = None, None  # m_mHXPrime, m_mHYPrime
    m_mDX, m_mDY = None, None  # m_mDX, m_mDY
    m_mLUDecompX, m_mLUDecompY = None, None  # m_mLUDecompX, m_mLUDecompY
