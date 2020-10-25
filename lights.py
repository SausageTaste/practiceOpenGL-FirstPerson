from typing import Tuple

from actor import Actor
import mymath as mmath


class PointLight:
    def __init__(self, position_t: Tuple[float, float, float], lightColor_t: Tuple[float, float, float],
                 maxDistance_f: float=5):
        self.x, self.y, self.z = tuple(map(lambda xx: float(xx), position_t))
        self.r, self.g, self.b = tuple(map(lambda xx: float(xx), lightColor_t))
        self.maxDistance_f = float(maxDistance_f)

    def getXYZ(self):
        return self.x, self.y, self.z

    def getRGB(self):
        return self.r, self.g, self.b

    def setXYZ(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class SpotLight(Actor):
    def __init__(self, position_t:Tuple[float, float, float], lightColor_t:Tuple[float, float, float],
                 maxDistance_f:float, cutoff:float, directionVec3:mmath.Vec3, parent:Actor=None):
        super().__init__(parent)
        self.pos_l = list(map(lambda xx:float(xx), position_t))
        self.lookHorDeg_f = 90.0
        self.r, self.g, self.b = tuple(map(lambda xx:float(xx), lightColor_t))
        self.maxDistance_f = float(maxDistance_f)
        self.cutoff_f = float(cutoff)
        self.directionVec3 = directionVec3
        self.directionVec3.normalize()

    def getRGB(self):
        return self.r, self.g, self.b

    def setXYZ(self, x, y, z):
        self.pos_l[0] = float(x)
        self.pos_l[1] = float(y)
        self.pos_l[2] = float(z)
