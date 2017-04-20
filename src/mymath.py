from typing import Tuple, Union, Iterable
from math import sin, cos, tan, pi, sqrt, acos

import numpy as np
from numpy.matrixlib import defmatrix


TYPE_MATRIX = defmatrix.matrix


class Vec3:
    def __init__(self, *args):
        if len(args) == 3:
            x, y, z = args
        elif len(args) == 1:
            x, y, z = args[0].getXYZ()
        else:
            raise ValueError

        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __str__(self) -> str:
        return "<{}.Vec3 object at 0x{:0>16X}, x: {:0.2f}, y: {:0.2f}, z: {:0.2f}".\
               format(__name__, id(self), self.x, self.y, self.z)

    def __repr__(self) -> str:
        if __name__ == '__main__':
            return "Vec3({}, {}, {})".format(self.x, self.y, self.z)
        else:
            return "{}.Vec3({}, {}, {})".format(__name__, self.x, self.y, self.z)

    def __add__(self, other:"Vec3") -> "Vec3":
        xSum_f = self.x + other.x
        ySum_f = self.y + other.y
        zSum_f = self.z + other.z

        return Vec3(xSum_f, ySum_f, zSum_f)

    def __sub__(self, other):
        xSub_f = self.x - other.x
        ySub_f = self.y - other.y
        zSub_f = self.z - other.z

        return Vec3(xSub_f, ySub_f, zSub_f)

    def getXYZ(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def setXYZ(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def rotate(self, degree, x, y, z):
        rotateMat = rotateMat4(degree, x, y, z)
        a = rotateMat.transpose() * np.matrix(self.getXYZ() + (0,), np.float32).transpose()
        self.x = a.item(0)
        self.y = a.item(1)
        self.z = a.item(2)

    def normalize(self) -> "Vec3":
        divider_f = sqrt(self.x**2 + self.y**2 + self.z**2)
        xNew_f = self.x / divider_f
        yNew_f = self.y / divider_f
        zNew_f = self.z / divider_f

        return Vec3(xNew_f, yNew_f, zNew_f)

    def cross(self, other:"Vec3") -> "Vec3":
        return Vec3(self.y*other.z - self.z*other.y, self.z*other.x - self.x*other.z, self.x*other.y - self.y*other.x)

    def dot(self, other) -> float:
        return self.x*other.x + self.y*other.y + self.z*other.z

    def radianDiffer(self, other) -> "Angle":
        radian_f = acos(
            (self.dot(other)) /
            ( sqrt(self.x**2 + self.y**2 + self.z**2)*sqrt(other.x**2 + other.y**2 + other.z**2) )
        )
        return Angle.radian(radian_f)


class Vec4(Vec3):
    def __init__(self, *args):
        if len(args) == 4:
            x, y, z, w = args
            super().__init__(x, y, z)
            self.w = float(w)
        elif len(args) == 2:
            vec3, w = args
            super().__init__(*vec3.getXYZ())
            self.w = float(w)
        else:
            raise ValueError

    def __str__(self) -> str:
        return "<{}.Vec4 object at 0x{:0>16X}, x: {:0.2f}, y: {:0.2f}, z: {:0.2f}, w: {:0.2f}>".\
               format(__name__, id(self), self.x, self.y, self.z, self.w)

    def __repr__(self) -> str:
        if __name__ == '__main__':
            return "Vec4({}, {}, {}, {})".format(self.x, self.y, self.z, self.w)
        else:
            return "{}.Vec4({}, {}, {}, {})".format(__name__, self.x, self.y, self.z, self.w)

    def __add__(self, other:"Vec4") -> "Vec4":
        xSum_f = self.x + other.x
        ySum_f = self.y + other.y
        zSum_f = self.z + other.z
        wSum_f = self.w + other.w
        if wSum_f > 0:
            wSum_f = 1.0

        return Vec4(xSum_f, ySum_f, zSum_f, wSum_f)

    def __sub__(self, other:"Vec4") -> "Vec4":
        xSub_f = self.x - other.x
        ySub_f = self.y - other.y
        zSub_f = self.z - other.z
        wSub_f = self.w = other.w

        return Vec4(xSub_f, ySub_f, zSub_f, wSub_f)

    def __mul__(self, other:float) -> "Vec4":
        other = float(other)
        xNew_f = self.x * other
        yNew_f = self.y * other
        zNew_f = self.z * other

        return Vec4(xNew_f, yNew_f, zNew_f, self.w)

    def length(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def getXYZW(self):
        return self.x, self.y, self.z, self.w

    def setXYZW(self, x, y, z, w):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def rotate(self, degree, x, y, z):
        rotateMat = rotateMat4(degree, x, y, z)
        a = rotateMat.transpose() * np.matrix(self.getXYZW(), np.float32).transpose()
        self.x = a.item(0)
        self.y = a.item(1)
        self.z = a.item(2)
        self.w = a.item(3)

    def normalize(self) -> "Vec4":
        divider_f = self.length()
        xNew_f = self.x / divider_f
        yNew_f = self.y / divider_f
        zNew_f = self.z / divider_f
        wNew_f = self.w / divider_f

        return Vec4(xNew_f, yNew_f, zNew_f, wNew_f)

    def cross(self, other: "Vec4") -> "Vec4":
        return Vec4(self.y * other.z - self.z * other.y,
                    self.z * other.x - self.x * other.z,
                    self.x * other.y - self.y * other.x,
                    0.0)

    def cross3(self, other:"Vec4", theOther:"Vec4"):
        v0 = other - self  # self -> other
        v1 = theOther - other  # other -> theOther

        return v0.cross(v1).normalize()

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def radianDiffer(self, other) -> "Angle":
        radian_f = acos(
            (self.dot(other)) /
            (sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2) * sqrt(other.x ** 2 + other.y ** 2 + other.z ** 2))
        )
        return Angle.radian(radian_f)

    def transform(self, *args:Iterable[TYPE_MATRIX]):
        a = np.matrix(self.getXYZW(), np.float32)
        for tranMat in args:
            a *= tranMat
        return Vec4(a.item(0), a.item(1), a.item(2), a.item(3))


def getMoveVector(speed:float, lookingAngle:"Angle") -> Tuple[float, float]:
    radian_f = lookingAngle.getRadian()
    return speed * cos(radian_f), speed * sin(radian_f)


def getMoveVector3(lookHorAngle:"Angle", lookVerAngle:"Angle"):
    yVec = lookVerAngle.getDegree() / 90
    if yVec > 1.0 or yVec < -1.0:
        raise FileNotFoundError(yVec)

    divider_f = 1.0 - abs(yVec)
    xVec, zVec = getMoveVector(1, lookHorAngle)
    xVec *= divider_f
    zVec *= -divider_f
    a = Vec3(xVec, yVec, zVec)
    a.normalize()
    return a


class Angle:
    def __init__(self, degree_f:float):
        self.__degree_f = float(degree_f)

    @classmethod
    def radian(cls, radian_f:float):
        return Angle(radian_f*180/pi)

    def __str__(self) -> str:
        return "<{}.Angle object at 0x{:0>16X}, degree: {}, radian: {}>".format(__name__, id(self), self.getDegree(),
                                                                                 self.getRadian())

    def __repr__(self) -> str:
        if __name__ == '__main__':
            return "Angle({})".format(self.getDegree())
        else:
            return "{}.Angle({})".format(__name__, self.getDegree())

    #### Only these uses attributes ####

    def getDegree(self) -> float:
        return self.__degree_f

    def setDegree(self, degree_f:Union[float, int]) -> None:
        for _ in range(10000):
            if degree_f >= 360.0:
                degree_f -= 360.0
            elif degree_f < 0.0:
                degree_f += 360.0
            else:
                break
        else:
            raise ValueError("Too big value.")

        self.__degree_f = float(degree_f)

    ####  ####

    def getRadian(self) -> float:
        return self.getDegree() / 180 * pi

    def setRadian(self, radian_f:Union[float, int]) -> None:
        self.setDegree(radian_f * 180 / pi)

    def addDegree(self, degree_f:Union[float, int]) -> None:
        self.setDegree(self.getDegree() + degree_f)

    def addRadian(self, radian_f:Union[float, int]) -> None:
        self.setRadian(self.getRadian() + radian_f)

    def copy(self) -> "Angle":
        return Angle(self.__degree_f)


def normalizeVec3(x:float, y:float, z:float) -> Tuple[float, float, float]:
    divider_f = sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= divider_f
    y /= divider_f
    z /= divider_f

    return x, y, z


def identityMat4():
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)


def translateMat4(x:float, y:float, z:float, scala=1.0):
    return np.matrix([[1.0, 0.0, 0.0, x*scala],
                      [0.0, 1.0, 0.0, y*scala],
                      [0.0, 0.0, 1.0, z*scala],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).transpose()


def rotateMat4(degree_f:float, x:float, y:float, z:float):
    radian_f = degree_f / 180 * pi
    xa = radian_f * x
    ya = radian_f * y
    za = radian_f * z

    return np.matrix([[cos(ya)*cos(za), cos(xa)*sin(za) + sin(xa)*sin(ya)*cos(za), sin(xa)*sin(za) - cos(xa)*sin(ya)*cos(za), 0.0],
                      [-1*cos(ya)*sin(za), cos(xa)*cos(za) - sin(xa)*sin(ya)*sin(za), sin(xa)*cos(za) + cos(xa)*sin(ya)*sin(za), 0.0],
                      [sin(ya), -1*sin(xa)*cos(ya), cos(xa)*cos(ya), 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).transpose()


def scaleMat4(x, y, z):
    return np.matrix([[x, 0.0, 0.0, 0.0],
                      [0.0, y, 0.0, 0.0],
                      [0.0, 0.0, z, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).transpose()


def frustumMat4(left:float, right:float, bottom:float, top:float, n:float, f:float):
    if right == left or top == bottom or n == f or n < 0.0 or f < 0.0:
        return identityMat4()
    else:
        return np.matrix([[2*n,  0,                         (right + left) / (right - left),  0                     ],
                          [0,    (2 * n) / (top - bottom),  (top + bottom) / (top - bottom),  0                     ],
                          [0,    0,                         (n + f) / (n - f),                (2 * n * f) / (n - f) ],
                          [0,    0,                         -1,                               1                     ]], dtype=np.float32)


def perspectiveMat4(fov, aspect, n, f):
    fov = fov / 180 * pi
    tanHalfFov_f = tan(fov / 2)
    return np.matrix([[1 / (aspect*tanHalfFov_f), 0, 0, 0],
                      [0,1 / tanHalfFov_f, 0, 0],
                      [0, 0, -1 * (f + n) / (f - n), -1 * (2*f*n) / (f - n)],
                      [0, 0, -1, 0]], dtype=np.float32).transpose()


def orthoMat4(l, r, b, t, n, f):
    return np.matrix([[2 / (r - l), 0,         0,         (l+r) / (l-r)],
                      [0,           2 / (t-b), 0,         (b+t) / (b-t)],
                      [0,           0,         2 / (n-f), (n+f) / (n-f)],
                      [0,           0,         0,         1]], dtype=np.float32).transpose()


def getlookatMat4(eye:Vec4, center:Vec4, up:Vec4) -> np.ndarray:
    f = (center - eye).normalize()
    upN = up.normalize()
    s = f.cross(upN)
    u = s.cross(f)
    return np.matrix([
        [ s.x,  u.x,  -f.x,  0.0 ],
        [ s.y,  u.y,  -f.y,  0.0 ],
        [ s.z,  u.z,  -f.z,  0.0 ],
        [ 0.0,  0.0,   0.0,  1.0 ]
    ], np.float32)


def main():
    a = Vec4(1, 0, 0, 0)
    a = a.transform(rotateMat4(1, 0, 90, 0))
    print(a)


if __name__ == '__main__':
    main()
