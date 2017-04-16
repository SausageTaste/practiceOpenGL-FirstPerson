from typing import Optional

import mymath as mmath


class Actor:
    """
    This is parent class for all actors.
    Actor means objects that can move, rotate, resize.
    
    Instance who inherited this class is able to store its location, looking angles, size data.
    And using those data, it can make translate, rotate, scale matrix.
    Also it has some methods like one that makes it move according to its looking angle by using trigonometrical function/
    
    이것은 모든 Actor들의 부모 클래스입니다.
    Actor란, 움직이고 회전하고 크기를 조절할 수 있는 물체를 말합니다.
    
    이 클래스를 상속받은 인스턴스는 위치, 바라보는 각도, 크기 데이터를 저장할 수 있습니다.
    그리고 이 데이터를 이용하여 translate, rotate, scale 행렬을 만들 수 있습니다.
    또한 삼각함수를 이용해여 보는 방향에 따라 이동을 하는 메소드 등 여러 메소드를 가집니다.
    """
    def __init__(self, parent:"Actor"=None, initPos_t:Optional[list]=None, initScale:Optional[list]=None):

        self.parent = parent

        self.collideModels_l = []
        self.collideActors_l = []

        self.physics = False

        if initPos_t is None:
            self.pos_l = [0.0, 0.0, 0.0]
        else:
            self.pos_l = list(initPos_t)
        if initScale is None:
            self.scale_l = [1, 1, 1]
        else:
            self.scale_l = list(initScale)

        self.lookVerDeg_f = 0.0
        self.lookHorDeg_f = 0.0

        self.moveSpeed_f = 10.0
        self.rotateSpeed_f = 100.0
        self.rotateMouseSpeed_f = 0.05

        self.flySpeed_f = 10

    def updateActor(self, timeDelta):
        if self.physics:
            self._applyGravity(timeDelta)

    def move(self, timeDelta: float, f: bool, b: bool, l: bool, r: bool, u: bool, d: bool) -> None:
        """
        This method takes keyboard input and move camera position.
        But forward, back, left, right control moves only horizontally.
        That means looking up and pressing forward key doesn't let you go up.
        This will not be used anymore.

        이 메소드는 키보드 입력을 받아 카메라 위치를 이동합니다.
        하지만 forward, back, left, right 컨트롤은 수평으로만 움직입니다.
        즉, 위를 바라보며 forward 키를 누른다고 해서 위로 올라갈 수 있지는 않습니다.
        이 메소드는 사용되지 않습니다.

        :param timeDelta: Time gap between previous frame and current frame.
        :param f: True if 'forward' key is toggled on.
        :param b: True if 'back' key is toggled on.
        :param l: True if 'left' key is toggled on.
        :param r: True if 'right' key is toggled on.
        :param u: True if 'up' key is toggled on.
        :param d: True if 'down' key is toggled on.
        """
        if f and not b:
            xVec, zVec = mmath.getMoveVector(self.moveSpeed_f, mmath.Angle(self.lookHorDeg_f + 90))
            self.pos_l[0] += xVec * timeDelta
            self.pos_l[2] -= zVec * timeDelta
        elif not f and b:
            xVec, zVec = mmath.getMoveVector(self.moveSpeed_f, mmath.Angle(self.lookHorDeg_f + 90))
            self.pos_l[0] -= xVec * timeDelta
            self.pos_l[2] += zVec * timeDelta

        if l and not r:
            xVec, zVec = mmath.getMoveVector(self.moveSpeed_f, mmath.Angle(self.lookHorDeg_f))
            self.pos_l[0] -= xVec * timeDelta
            self.pos_l[2] += zVec * timeDelta
        elif not l and r:
            xVec, zVec = mmath.getMoveVector(self.moveSpeed_f, mmath.Angle(self.lookHorDeg_f))
            self.pos_l[0] += xVec * timeDelta
            self.pos_l[2] -= zVec * timeDelta

        if u and not d:
            self.pos_l[1] += self.flySpeed_f * timeDelta
        elif not u and d:
            self.pos_l[1] -= self.flySpeed_f * timeDelta

        if self.physics:
            self._checkCollide()

        self._validateValues()

    def move3(self, timeDelta: float, f: bool, b: bool, l: bool, r: bool, u: bool, d: bool) -> None:
        """
        This method allows you to move more like real FPS games.
        With this, you can go up by looking up and pressing forward key.

        이 메소드를 사용하면 좀 더 실제 FPS 게임처럼 움직일 수 있습니다.
        위를 바라보며 forward 키를 누르면 위로 올라갈 수 있습니다.

        params all Same as self.move()
        """
        if f and not b:
            moveVec3 = mmath.getMoveVector3(mmath.Angle(self.lookHorDeg_f + 90), mmath.Angle(self.lookVerDeg_f))
            self.pos_l[0] += moveVec3.x * self.moveSpeed_f * timeDelta
            self.pos_l[1] += moveVec3.y * self.moveSpeed_f * timeDelta
            self.pos_l[2] += moveVec3.z * self.moveSpeed_f * timeDelta
        elif not f and b:
            moveVec3 = mmath.getMoveVector3(mmath.Angle(self.lookHorDeg_f + 90), mmath.Angle(self.lookVerDeg_f))
            self.pos_l[0] -= moveVec3.x * self.moveSpeed_f * timeDelta
            self.pos_l[1] -= moveVec3.y * self.moveSpeed_f * timeDelta
            self.pos_l[2] -= moveVec3.z * self.moveSpeed_f * timeDelta

        if l and not r:
            xVec, zVec = mmath.getMoveVector(self.moveSpeed_f, mmath.Angle(self.lookHorDeg_f))
            self.pos_l[0] -= xVec * timeDelta
            self.pos_l[2] += zVec * timeDelta
        elif not l and r:
            xVec, zVec = mmath.getMoveVector(self.moveSpeed_f, mmath.Angle(self.lookHorDeg_f))
            self.pos_l[0] += xVec * timeDelta
            self.pos_l[2] -= zVec * timeDelta

        if u and not d:
            self.pos_l[1] += self.flySpeed_f * timeDelta
        elif not u and d:
            self.pos_l[1] -= self.flySpeed_f * timeDelta

        self._validateValues()
        self._checkCollide()

    def moveForward(self, distance_f:float):
        moveVec3 = mmath.getMoveVector3(mmath.Angle(self.lookHorDeg_f + 90), mmath.Angle(self.lookVerDeg_f))
        self.pos_l[0] += moveVec3.x * distance_f
        self.pos_l[1] += moveVec3.y * distance_f
        self.pos_l[2] += moveVec3.z * distance_f

    def rotate(self, timeDelta: float, u: bool, d: bool, l: bool, r: bool) -> None:
        """
        It rotates camera angle as you press certain keys.
        This method is not needed since we've got better one.
        But this method is still used.
        Initiate the app and press arrow keys.

        이 메소드는 키 입력에 때라 카메라를 회전합니다.
        더 나은 메소드가 있기 때문에 이것은 필요없습니다.
        하지만 이 메소드는 여전히 사용되고는 있습니다.
        애플리케이션을 실행해서 화살표 키를 누르세요.

        :param timeDelta: Time gap between previous frame and current frame.
        :param u: True if 'look up' key is toggled on.
        :param d: True if 'look down' key is toggled on.
        :param l: True if 'look left' key is toggled on.
        :param r: True if 'look right' key is toggled on.
        """
        if u and not d:
            self.lookVerDeg_f += self.rotateSpeed_f * timeDelta
        elif not u and d:
            self.lookVerDeg_f -= self.rotateSpeed_f * timeDelta
        if l and not r:
            self.lookHorDeg_f += self.rotateSpeed_f * timeDelta
        elif not l and r:
            self.lookHorDeg_f -= self.rotateSpeed_f * timeDelta

        self._validateValues()

    def rotateMouse(self, mx: int, my: int) -> None:
        """
        Now you can look around with familiar controller, mouse.

        이제 당신은 친숙한 마우스를 이용하여 주위를 둘러볼 수 있습니다.

        :param mx: Number of x-axis pixels mouse has traveled on.
        :param my: Number of y-axis pixels mouse has traveled on.
        """
        self.lookHorDeg_f += mx * self.rotateMouseSpeed_f
        self.lookVerDeg_f += my * self.rotateMouseSpeed_f * 2

        self._validateValues()

    def getXYZ(self) -> tuple:
        return tuple(self.pos_l)

    def getWorldXYZ(self):
        xWorld_f, yWorld_f, zWorld_f = self.pos_l

        if self.parent is not None:
            xWorld_f += self.parent.pos_l[0]
            yWorld_f += self.parent.pos_l[1]
            zWorld_f += self.parent.pos_l[2]

        return xWorld_f, yWorld_f, zWorld_f

    def getWorldDegree(self):
        hor = self.lookHorDeg_f
        ver = self.lookVerDeg_f
        if self.parent is not None:
            hor += self.parent.lookHorDeg_f
            ver += self.parent.lookVerDeg_f
        return hor, ver

    def getLookVec3(self) -> mmath.Vec3:
        return mmath.getMoveVector3(mmath.Angle(self.lookHorDeg_f + 90), mmath.Angle(self.lookVerDeg_f))

    def getModelMatrix(self):
        if self.parent is None:
            return mmath.scaleMat4(*self.scale_l) * mmath.rotateMat4(-self.lookVerDeg_f, 1, 0, 0) * mmath.rotateMat4(-self.lookHorDeg_f, 0, 1, 0) *\
                   mmath.translateMat4(*self.pos_l)
        else:
            return mmath.rotateMat4(-self.lookVerDeg_f, 1, 0, 0) * mmath.rotateMat4(-self.lookHorDeg_f, 0, 1, 0) *\
                   mmath.translateMat4(*self.pos_l) * self.parent.getModelMatrix()

    def _applyGravity(self, timeDelta:float):
        distance = timeDelta * 10
        if distance > 0.5:
            distance = 0.5

        self.pos_l[1] -= distance

    def _validateValues(self) -> None:
        """
        Keeps lookHorDeg_f's value between 0 and 360.
        Keeps lookVerDeg_f's value between -90 and 90.

        lookHorDeg_f의 값이 0과 360 사이에 머물도록 합니다.
        lookVerDeg_f의 값이 -90과 90 사이에 머물도록 합니다.
        """
        if self.lookHorDeg_f > 360:
            self.lookHorDeg_f %= 360
        elif self.lookHorDeg_f < 0:
            for _ in range(10):
                self.lookHorDeg_f += 360
                if self.lookHorDeg_f >= 0:
                    break

        if self.lookVerDeg_f > 90:
            self.lookVerDeg_f = 90
        elif self.lookVerDeg_f < -90:
            self.lookVerDeg_f = -90

    def _checkCollide(self):
        pass

    def collideAction(self, moveToOut_t:tuple):
        raise FileNotFoundError
