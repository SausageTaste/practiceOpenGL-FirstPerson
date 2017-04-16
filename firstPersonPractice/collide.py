import math
from typing import Tuple, Union, Optional

import mymath as mmath
import main_v2 as mainpy
from actor import Actor


UNION_NUM = Union[float, int]
VEC34 = Union[mmath.Vec3, mmath.Vec4]


def hitCheckPlaneSegment(pln:"Plane", seg:"Segment", printResult_b:bool=False) -> Tuple[int, Optional[mmath.Vec4], Optional[float]]:
    """
    return values
    [0] : Result code, described below.
    [1] : Meet coord.
    [2] : Vector's length - (from start point to meet point vector)'s length.
    
    -1 : Vector is too short to reach the plane.
    -2 : Plane and vector are parallel and never gonna meet.
    0 : Vector is heading opposite side from plane.
    -4 : Length is long enough to reach plane but heading wrong way.
    1 : They meat.
    
    """
    dotCheck = ( pln.a*seg.posVec4.x + pln.b*seg.posVec4.y + pln.c*seg.posVec4.z )
    if dotCheck >= 0.0:
        return -3, None, None

    l = pln.a*seg.posVec4.x + pln.b*seg.posVec4.y + pln.c*seg.posVec4.z + pln.d
    if printResult_b:
        print("\tDistance of plane and segment:", l)

    vec_l2 = seg.vecVec4.x**2 + seg.vecVec4.y**2 + seg.vecVec4.z**2
    if vec_l2 < l**2:
        return -1, None, None  # Vector is too short to reach the plane.

    vec_l = math.sqrt(vec_l2)
    if printResult_b:
        print("\tVector length:", vec_l)

    # cos 0을 내적을 사용해 구한다.
    vec = seg.vecVec4 * (1.0 / vec_l)  # Normalize
    dot = vec.dot( mmath.Vec4(-pln.a, -pln.b, -pln.c, 0.0) )
    if printResult_b:
        print("\tDot:", dot)

    if dot == 0.0:
        return -2, None, None  # Plane and vector are parallel and never gonna meet.

    pln_l = abs(l / dot)
    if printResult_b:
        print("\tDistance of start and meet points:", pln_l)
    hitPosVec = seg.posVec4 + vec * pln_l


    if pln_l > vec_l:
        return -4, hitPosVec, vec_l - pln_l  # Length is long enough to reach plane but heading wrong way.
    if dot < 0.0:
        return 0, hitPosVec, vec_l - pln_l  # Vector is heading opposite side from plane.

    return 1, hitPosVec, vec_l - pln_l


def triangleCheckInner(tri:"Triangle", p:mmath.Vec4, printResult_b:bool=False) -> bool:
    vt0 = tri.getPoint0() - p
    vt1 = tri.getPoint1() - p
    vt2 = tri.getPoint2() - p

    v0 = tri.getPoint1() - tri.getPoint0()
    v1 = tri.getPoint2() - tri.getPoint1()
    v2 = tri.getPoint0() - tri.getPoint2()

    c0 = vt0.cross(v0)
    c1 = vt1.cross(v1)
    c2 = vt2.cross(v2)

    dot0 = c0.dot(c1)
    dot1 = c1.dot(c2)
    dot2 = c2.dot(c0)

    if printResult_b:
        print( "\t\t{} = {} - {}".format(c0, vt0, v0) )
        print( "\t\t{} = {} - {}".format(c1, vt1, v1) )
        print( "\t\t{} = {} - {}".format(c2, vt2, v2) )
        print("\t\tDot result:", dot0, dot1, dot2 )

    if math.copysign(1, dot0) == math.copysign(1, dot1) == math.copysign(1, dot2):
        return True
    else:
        return False


def hitCheckTriangleSegment(tri:"Triangle", seg:"Segment", printResult_b:bool=False) -> Tuple[int, int, Optional[mmath.Vec4], Optional[float]]:
    resultCode_i, hit_temp, insideLen_f = hitCheckPlaneSegment(tri.getPlane(), seg, printResult_b)
    if resultCode_i < 0:
        return -1, resultCode_i, None, None

    if not triangleCheckInner(tri, hit_temp, printResult_b):
        return -2, resultCode_i, hit_temp, insideLen_f
    else:
        return 1, resultCode_i, hit_temp, insideLen_f


def hitCheckPlanePlane(pln0:"Plane", pln1:"Plane"):
    dot = (pln0.a + pln1.a) + (pln0.b + pln1.b) + (pln0.c + pln1.c)
    if (dot >= 1.0) or (dot <= -1.0):
        return False  # When two planes are parallel.

    vec = mmath.Vec4(pln0.a, pln0.b, pln0.c, 0).cross(mmath.Vec4(pln1.a, pln1.b, pln1.c, 0))

    if vec.z != 0.0:
        pos = mmath.Vec4(
            ( pln0.b * pln1.d - pln1.b * pln0.d ) / vec.z,
            ( pln1.a * pln0.d + pln0.a * pln1.d ) / vec.z,
            0, 1
        )
    elif vec.y != 0.0:
        pos = mmath.Vec4(
            ( pln1.c * pln0.d + pln0.c * pln1.d ) / vec.y,
            0.0,
            ( pln0.a * pln1.d + pln1.a * pln0.d ) / vec.y,
            1.0
        )
    else:
        pos = mmath.Vec4(
            0.0,
            ( pln0.c * pln1.d - pln1.c * pln0.d ) / vec.x,
            ( pln1.b * pln0.d - pln0.b * pln1.d ) / vec.x,
            1.0
        )

    return True, Segment(*pos.getXYZ(), *vec.getXYZ())


def hitCheckPlaneTriangle(pln:"Plane", tri:"Triangle"):
    side0 = pln.pointInFront(tri.getPoint0())
    side1 = pln.pointInFront(tri.getPoint1())
    if (side0 and not side1) or (not side0 and side1):
        return True

    side2 = pln.pointInFront(tri.getPoint2())
    if (side1 and not side2) or (not side1 and side2):
        return True

    return False


def hitCheckTriangleTriangle(tri0:"Triangle", tri1:"Triangle", printResult_b:bool=False):
    vec0_0 = tri0.getPoint1() - tri0.getPoint0()
    vec0_1 = tri0.getPoint2() - tri0.getPoint1()
    vec0_2 = tri0.getPoint0() - tri0.getPoint2()


    if not printResult_b:
        hit0_0 = hitCheckTriangleSegment(tri1, Segment(*tri0.getPoint0().getXYZ(), *vec0_0.getXYZ()))
        hit0_1 = hitCheckTriangleSegment(tri1, Segment(*tri0.getPoint1().getXYZ(), *vec0_1.getXYZ()))
        hit0_2 = hitCheckTriangleSegment(tri1, Segment(*tri0.getPoint2().getXYZ(), *vec0_2.getXYZ()))
    else:
        print()
        print("="*80)
        print()
        print(tri1.getPoint012())

        print(Segment( *tri0.getPoint0().getXYZ(), *vec0_0.getXYZ() ))
        hit0_0 = hitCheckTriangleSegment( tri1, Segment(*tri0.getPoint0().getXYZ(), *vec0_0.getXYZ()), printResult_b )
        print("\thit0_0:", hit0_0)

        print(Segment( *tri0.getPoint1().getXYZ(), *vec0_1.getXYZ() ))
        hit0_1 = hitCheckTriangleSegment(tri1, Segment(*tri0.getPoint1().getXYZ(), *vec0_1.getXYZ()), printResult_b)
        print("\thit0_1:", hit0_1)

        print(Segment( *tri0.getPoint2().getXYZ(), *vec0_2.getXYZ() ))
        hit0_2 = hitCheckTriangleSegment(tri1, Segment(*tri0.getPoint2().getXYZ(), *vec0_2.getXYZ()), printResult_b)
        print("\thit0_2:", hit0_2)

        print()

    if (hit0_0[0] > 0 and hit0_1[0] > 0) or (hit0_1[0] > 0 and hit0_2[0] > 0) or (hit0_2[0] > 0 and hit0_0[0] > 0):
        return 1

    vec1_0 = tri1.getPoint1() - tri1.getPoint0()
    vec1_1 = tri1.getPoint2() - tri1.getPoint1()
    vec1_2 = tri1.getPoint0() - tri1.getPoint2()


    if not printResult_b:
        hit1_0 = hitCheckTriangleSegment( tri0, Segment(*tri1.getPoint0().getXYZ(), *vec1_0.getXYZ()) )
        hit1_1 = hitCheckTriangleSegment( tri0, Segment(*tri1.getPoint1().getXYZ(), *vec1_1.getXYZ()) )
        hit1_2 = hitCheckTriangleSegment( tri0, Segment(*tri1.getPoint2().getXYZ(), *vec1_2.getXYZ()) )
    else:
        print(tri0.getPoint012())

        print( Segment( *tri1.getPoint0().getXYZ(), *vec1_0.getXYZ() ) )
        hit1_0 = hitCheckTriangleSegment(tri0, Segment(*tri1.getPoint0().getXYZ(), *vec1_0.getXYZ()), printResult_b)
        print("\thit1_0:", hit1_0)

        print( Segment( *tri1.getPoint1().getXYZ(), *vec1_1.getXYZ() ) )
        hit1_1 = hitCheckTriangleSegment(tri0, Segment(*tri1.getPoint1().getXYZ(), *vec1_1.getXYZ()), printResult_b)
        print("\thit1_1:", hit1_1)

        print( Segment( *tri1.getPoint2().getXYZ(), *vec1_2.getXYZ() ) )
        hit1_2 = hitCheckTriangleSegment(tri0, Segment(*tri1.getPoint2().getXYZ(), *vec1_2.getXYZ()), printResult_b)
        print("\thit1_2:", hit1_2)

        print()

    if (hit1_0[0] > 0 and hit1_1[0] > 0) or (hit1_1[0] > 0 and hit1_2[0] > 0) or (hit1_2[0] > 0 and hit1_0[0] > 0):
        return 2

    if (hit0_0[0] > 0 or hit0_1[0] > 0 or hit0_2[0] > 0) and (hit1_0[0] > 0 or hit1_1[0] > 0 or hit1_2[0] > 0):
        return 3

    return 0


def hitCheckAabbSegment(self:"Aabb", seg:"Segment"):
    if seg.vecVec4.x == 0.0:
        if ( seg.posVec4.x < self.min.x ) or ( seg.posVec4.x > self.max.x ):
            return False
        tx_min = 0.0
        tx_max = 1.0
    else:
        t0 = ( self.min.x - seg.posVec4.x ) / seg.vecVec4.x
        t1 = ( self.max.x - seg.posVec4.x ) / seg.vecVec4.x
        if t0 < t1:
            tx_min = t0
            tx_max = t1
        else:
            tx_min = t1
            tx_max = t0
        if (tx_max < 0.0) or (tx_min > 1.0):
            return False
    t_min = tx_min
    t_max = tx_max

    if seg.vecVec4.y == 0.0:
        if ( seg.posVec4.y < self.min.y ) or ( seg.posVec4.y > self.max.y ):
            return False
        ty_min = 0.0
        ty_max = 1.0
    else:
        t0 = ( self.min.y - seg.posVec4.y ) / seg.vecVec4.y
        t1 = ( self.max.y - seg.posVec4.y ) / seg.vecVec4.y
        if t0 < t1:
            ty_min = t0
            ty_max = t1
        else:
            ty_min = t1
            ty_max = t0
        if ( ty_max < 0.0 ) or ( ty_min > 1.0 ):
            return False
    if ( t_max < ty_min ) or ( t_min > ty_max ):
        return False
    if t_min < ty_min:
        t_min = ty_min
    if t_max > ty_max:
        t_max = ty_max

    if seg.vecVec4.z == 0.0:
        if ( seg.posVec4.z < self.min.z ) or ( seg.posVec4.z > self.max.z ):
            return False
        tz_min = 0.0
        tz_max = 1.0
    else:
        t0 = ( self.min.z - seg.posVec4.z ) / seg.vecVec4.z
        t1 = ( self.max.z - seg.posVec4.z ) / seg.vecVec4.z
        if t0 < t1:
            tz_min = t0
            tz_max = t1
        else:
            tz_min = t1
            tz_max = t0
        if ( tz_max < 0.0 ) or ( tz_min > 1.0 ):
            return False
    if ( t_max < tz_max ) or ( t_min > tz_max ):
        return False
    if t_min < tz_min:
        t_min = tz_max
    if t_max > tz_max:
        t_max = tz_max

    if (t_min > 1.0) or (t_max < 0.0):
        return False
    else:
        return True


def getDistanceToPushBackAabbAabb(a:"Aabb", b:"Aabb"):
    xMaxA, yMaxA, zMaxA = a.getMaxXYZ()
    xMinA, yMinA, zMinA = a.getMinXYZ()

    xMaxB, yMaxB, zMaxB = b.getMaxXYZ()
    xMinB, yMinB, zMinB = b.getMinXYZ()

    xOne = xMaxA - xMinB
    xTwo = xMinA - xMaxB
    xDistance = xOne if abs(xOne) < abs(xTwo) else xTwo

    yOne = yMaxA - yMinB
    yTwo = yMinA - yMaxB
    yDistance = yOne if abs(yOne) < abs(yTwo) else yTwo

    zOne = zMaxA - zMinB
    zTwo = zMinA - zMaxB
    zDistance = zOne if abs(zOne) < abs(zTwo) else zTwo

    return -xDistance, -yDistance, -zDistance


def hitCheckAabbAabb(a:"Aabb", b:"Aabb"):
    xMaxA, yMaxA, zMaxA = a.getMaxXYZ()
    xMinA, yMinA, zMinA = a.getMinXYZ()

    xMaxB, yMaxB, zMaxB = b.getMaxXYZ()
    xMinB, yMinB, zMinB = b.getMinXYZ()

    if xMaxA < xMinB or xMinA > xMaxB:
        return False
    if yMaxA < yMinB or yMinA > yMaxB:
        return False
    if zMaxA < zMinB or zMinA > zMaxB:
        return False
    return True


class Segment:
    def __init__(self, xPos:UNION_NUM, yPos:UNION_NUM, zPos:UNION_NUM,
                       xVec:UNION_NUM, yVec:UNION_NUM, zVec:UNION_NUM):
        self.posVec4 = mmath.Vec4(xPos, yPos, zPos, 1.0)
        self.vecVec4 = mmath.Vec4(xVec, yVec, zVec, 0.0)

    def __str__(self):
        return "<Segment object, pos: {}, vec: {}>".format(self.posVec4.getXYZ(), self.vecVec4.getXYZ())


class Sphere:
    def __init__(self, xPos:UNION_NUM, yPos:UNION_NUM, zPos:UNION_NUM, radius_f:UNION_NUM):
        self.centerPosVec4 = mmath.Vec4(xPos, yPos, zPos, 1.0)
        self.radius_f = float(radius_f)


class Plane:
    def __init__(self, normalVec:VEC34, p0:VEC34):
        self.a = None
        self.b = None
        self.c = None
        self.d = None

        self.__create(normalVec, p0)

    @staticmethod
    def from3Dots(p0, p1, p2):
        n = p0.cross3(p1, p2)
        return Plane(n, p0)

    def __create(self, normalVec:VEC34, p0:VEC34):
        n = normalVec.normalize()

        self.a = float(n.x)
        self.b = float(n.y)
        self.c = float(n.z)
        self.d = float( -( (n.x * p0.x) + (n.y * p0.y) + (n.z * p0.z) ) )

    def distance(self, p:VEC34):
        return self.a*p.x + self.b*p.y + self.c*p.z + self.d

    def pointInFront(self, p:VEC34):
        if self.distance(p) < 0.0:
            return False
        else:
            return True


class Triangle(Actor):
    def __init__(self, p0:VEC34, p1:VEC34, p2:VEC34, parent:"mainpy.Actor"=None, initPos_l:list=None):
        super().__init__(parent, initPos_l)
        if isinstance(p0, mmath.Vec3):
            p0 = mmath.Vec4(p0, 1.0)
        if isinstance(p1, mmath.Vec3):
            p1 = mmath.Vec4(p1, 1.0)
        if isinstance(p2, mmath.Vec3):
            p2 = mmath.Vec4(p2, 1.0)

        self.points_l = [p0, p1, p2]
        self.plane = Plane.from3Dots(p0, p1, p2)

    def __str__(self):
        return "<Triangle object, {}>".format(self.getPoint012())

    def getPoint0(self) -> mmath.Vec4:
        return self.points_l[0] + mmath.Vec4(*self.getWorldXYZ(), 1.0)

    def getPoint1(self) -> mmath.Vec4:
        return self.points_l[1] + mmath.Vec4(*self.getWorldXYZ(), 1.0)

    def getPoint2(self) -> mmath.Vec4:
        return self.points_l[2] + mmath.Vec4(*self.getWorldXYZ(), 1.0)

    def getPoint012(self):
        return self.getPoint0(), self.getPoint1(), self.getPoint2()

    def getPlane(self):
        return Plane.from3Dots(self.getPoint0(), self.getPoint1(), self.getPoint2())


class Aabb:
    def __init__(self, minPos:VEC34, maxPos:VEC34):
        x0, y0, z0 = minPos.getXYZ()
        x1, y1, z1 = maxPos.getXYZ()

        xS = x0 if x0 < x1 else x1
        yS = y0 if y0 < y1 else y1
        zS = z0 if z0 < z1 else z1

        xB = x0 if x0 >= x1 else x1
        yB = y0 if y0 >= y1 else y1
        zB = z0 if z0 >= z1 else z1

        self.min = mmath.Vec4(xS, yS, zS, 1)
        self.max = mmath.Vec4(xB, yB, zB, 1)

    def __str__(self):
        return "<AABB object, min: {}, max: {}>".format(self.min.getXYZ(), self.max.getXYZ())

    def getMaxXYZ(self):
        return self.max.getXYZ()

    def getMinXYZ(self):
        return self.min.getXYZ()


class AabbActor(Aabb, Actor):
    def __init__(self, minPos:VEC34, maxPos:VEC34, parent=None, initPos_t=None):
        mainpy.Actor.__init__(self, parent, initPos_t)
        Aabb.__init__(self, minPos, maxPos)

    def getMaxXYZ(self):
        x,y,z = self.max.getXYZ()
        xW,yW,zW = self.getWorldXYZ()
        return x + xW, y + yW, z + zW

    def getMinXYZ(self):
        x, y, z = self.min.getXYZ()
        xW, yW, zW = self.getWorldXYZ()
        return x + xW, y + yW, z + zW


class Obb:
    def __init__(self, pos:VEC34, rot:VEC34, scale:VEC34):
        self.__mat = self._createMat(pos, rot, scale)
        print(self.__mat.getI())

    @staticmethod
    def _createMat(pos:VEC34, rot:VEC34, scale:VEC34):
        mat = mmath.translateMat4(*pos.getXYZ())
        mat *= mmath.rotateMat4(rot.x, 1, 0, 0)
        mat *= mmath.rotateMat4(rot.y, 0, 1, 0)
        mat *= mmath.rotateMat4(rot.z, 0, 0, 1)
        mat *= mmath.scaleMat4(*scale.getXYZ())

        return mat
