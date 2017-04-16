import sys
import itertools
from math import cos
from typing import Tuple, Generator

import numpy as np
from PIL import Image
import pygame as p
import pygame.locals as pl
import OpenGL.GL as gl
from OpenGL.GL import shaders

import mymath as mmath
import mypygame as mp
import obj_load as ol
import collide as co
from actor import Actor
from camera import Camera


class Controller:
    def __init__(self, mainLoop:"MainLoop", target:"Actor"):
        self.oldState_t = p.key.get_pressed()
        self.newState_t = p.key.get_pressed()

        self.target = target
        self.mainLoop = mainLoop

        self.__mouseControl_b = False
        self.__blockBound_b = True

        self.setMouseControl(self.__mouseControl_b)

        self.target = self.mainLoop.level.boxManager.boxes_l[0]
        self.target.physics = True
        self.mainLoop.camera.parent = self.mainLoop.level.boxManager.boxes_l[0]
        self.mainLoop.camera.pos_l = [0, 1.6, 0]
        self.mainLoop.camera.lookHorDeg_f = 0.0
        self.mainLoop.camera.lookVerDeg_f = 0.0
        self.target.renderFlag = False

    def update(self, fDelta:float):
        self.oldState_t = self.newState_t
        self.newState_t = p.key.get_pressed()

        if self.getStateChange(pl.K_f) == 1:
            if self.mainLoop.flashLight_b:
                self.mainLoop.flashLight_b = False
            else:
                self.mainLoop.flashLight_b = True
        if self.getStateChange(pl.K_ESCAPE) == 1:
            p.quit()
            sys.exit(0)
        if self.getStateChange(pl.K_F6) == 1:
            if self.__mouseControl_b:
                self.setMouseControl(False)
            else:
                self.setMouseControl(True)
        if self.getStateChange(pl.K_F7) == 1:
            if not self.__blockBound_b:
                self.target.physics = True
                self.__blockBound_b = True
            else:
                self.target.physics = False
                self.__blockBound_b = False
            """
                if not self.__blockBound_b:
                self.target = self.mainLoop.level.boxManager.boxes_l[0]
                self.mainLoop.camera.parent = self.mainLoop.level.boxManager.boxes_l[0]
                self.mainLoop.camera.pos_l = [0, 1.6, 0]
                self.mainLoop.camera.lookHorDeg_f = 0.0
                self.mainLoop.camera.lookVerDeg_f = 0.0
                self.target.renderFlag = False

                self.__blockBound_b = True
            else:
                self.target = self.mainLoop.camera
                self.mainLoop.camera.parent = None
                self.mainLoop.level.boxManager.boxes_l[0].renderFlag = True

                self.__blockBound_b = False
            """

        self.target.move(fDelta, self.newState_t[pl.K_w], self.newState_t[pl.K_s], self.newState_t[pl.K_a],
                         self.newState_t[pl.K_d], self.newState_t[pl.K_SPACE], self.newState_t[pl.K_LSHIFT])
        self.target.rotate(fDelta, self.newState_t[pl.K_UP], self.newState_t[pl.K_DOWN], self.newState_t[pl.K_LEFT],
                           self.newState_t[pl.K_RIGHT])

        if self.__mouseControl_b:
            xMouse_i, yMouse_i = p.mouse.get_rel()
            if abs(xMouse_i) <= 1:
                xMouse_i = 0
            if abs(yMouse_i) <= 1:
                yMouse_i = 0
            self.target.rotateMouse(-xMouse_i, -yMouse_i)

    def getStateChangesGen(self) -> Generator[Tuple[int, int], None, None]:
        assert len(self.oldState_t) == len(self.oldState_t)

        for x in range(len(self.oldState_t)):
            if self.oldState_t[x] != self.newState_t[x]:
                yield x, self.newState_t[x]

    def getStateChange(self, index:int) -> int:
        if self.oldState_t[index] != self.newState_t[index]:
            return self.newState_t[index]
        else:
            return -1

    def setMouseControl(self, boolean:bool):
        if boolean:
            self.__mouseControl_b = True
            p.event.set_grab(True)
            p.mouse.set_visible(False)
        else:
            self.__mouseControl_b = False
            p.event.set_grab(False)
            p.mouse.set_visible(True)


class PointLight:
    def __init__(self, position_t:Tuple[float, float, float], lightColor_t:Tuple[float, float, float], maxDistance_f:float=5):
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


class TextureContainer:
    def __init__(self):
        self.data_d = {0x21:self.getTexture("assets\\textures\\21.bmp"),
                       0x22:self.getTexture("assets\\textures\\22.bmp"),
                       0x12:self.getTexture("assets\\textures\\12.bmp")}

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError(item)

        return self.data_d[item]

    @staticmethod
    def getTexture(textureDir_s:str):
        aImg = Image.open(textureDir_s)
        imgW_i = aImg.size[0]
        imgH_i = aImg.size[1]
        image_bytes = aImg.tobytes("raw", "RGBX", 0, -1)
        imgArray = np.array([x / 255 for x in image_bytes], dtype=np.float32)

        texId = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texId)
        gl.glTexStorage2D(gl.GL_TEXTURE_2D, 6, gl.GL_RGBA32F, imgW_i, imgH_i)

        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, imgW_i, imgH_i, gl.GL_RGBA, gl.GL_FLOAT, imgArray)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 6)

        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        if 1:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        else:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        return texId


class StaticSurface:
    def __init__(self, vec1:mmath.Vec3, vec2:mmath.Vec3, vec3:mmath.Vec3, vec4:mmath.Vec3, surfaceVec:mmath.Vec3,
                 textureId:int, textureVerNum_f:float, textureHorNum_f:float, shininess:float, specularStrength:float):
        self.vertex1 = vec1
        self.vertex2 = vec2
        self.vertex3 = vec3
        self.vertex4 = vec4

        self.normal = surfaceVec
        self.textureHorNum_f = textureHorNum_f
        self.textureVerNum_f = textureVerNum_f
        self.shininess_f = shininess
        self.specularStrength_f = specularStrength

        self.textureId = textureId

        #### Vertex Array Obj ####

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        #### Vertices ####

        vertices = np.array([*vec1.getXYZ(),
                             *vec2.getXYZ(),
                             *vec3.getXYZ(),
                             *vec1.getXYZ(),
                             *vec3.getXYZ(),
                             *vec4.getXYZ()], dtype=np.float32)
        size = vertices.size * vertices.itemsize

        self.verticesBuffer = gl.glGenBuffers(1)  # Create a buffer.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.verticesBuffer)  # Bind the buffer.
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, vertices, gl.GL_STATIC_DRAW)  # Allocate memory.

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Defines vertex attributes. What are those?
        gl.glEnableVertexAttribArray(0)

        del size, vertices

        #### Texture Coord ####

        textureCoords = np.array([0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1], dtype=np.float32)
        size = textureCoords.size * textureCoords.itemsize

        self.texCoordBuffer = gl.glGenBuffers(1)  # Create a buffer.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texCoordBuffer)  # Bind the buffer.
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, textureCoords, gl.GL_STATIC_DRAW)  # Allocate memory.

        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Defines vertex attributes. What are those?
        gl.glEnableVertexAttribArray(1)

        del size, textureCoords

    def update(self):
        gl.glBindVertexArray(self.vao)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)

        #### To vertex shader ####

        gl.glUniform3f(2, *self.normal.getXYZ())
        gl.glUniform1f(3, self.textureHorNum_f)
        gl.glUniform1f(4, self.textureVerNum_f)

        #### To fragment shader ####

        gl.glUniform1f(11, self.shininess_f)
        gl.glUniform1f(53, self.specularStrength_f)

        ####  ####

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    @staticmethod
    def getProgram() -> int:
        with open("shader_source\\2nd_vs.txt") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log_s = gl.glGetShaderInfoLog(vertexShader).decode()
        if log_s:
            raise TypeError(log_s)

        with open("shader_source\\2nd_fs.txt") as file:
            fragmentShader = shaders.compileShader(file.read(), gl.GL_FRAGMENT_SHADER)
        log_s = gl.glGetShaderInfoLog(fragmentShader).decode()
        if log_s:
            raise TypeError(log_s)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        gl.glLinkProgram(program)

        print("Linking Log:", gl.glGetProgramiv(program, gl.GL_LINK_STATUS))

        gl.glDeleteShader(vertexShader)
        gl.glDeleteShader(fragmentShader)

        gl.glUseProgram(program)

        return program


class Box(Actor):
    def __init__(self, vec1:mmath.Vec3, vec2:mmath.Vec3, vec3:mmath.Vec3, vec4:mmath.Vec3,
                 vec5:mmath.Vec3, vec6:mmath.Vec3, vec7:mmath.Vec3, vec8:mmath.Vec3,
                 textureId:int, textureVerNum_f:float, textureHorNum_f:float, shininess:float, specularStrength:float,
                 startPos_l:list):
        super().__init__()

        self.renderFlag = True

        self.vertices_l = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        self.textureId = textureId
        self.textureVerNum_f = float(textureVerNum_f)
        self.textureHorNum_f = float(textureHorNum_f)
        self.shininess_f = float(shininess)
        self.specularStrength_f = float(specularStrength)

        self.pos_l = startPos_l

        self.selectedTexId = self.textureId
        self.textureId2 = None

        self.collideModels_l.append( co.AabbActor(vec3, vec5, self) )

        #self.collideModels_l.append()

        ####  ####

        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        vertices = np.array([*vec1.getXYZ(),
                             *vec2.getXYZ(),
                             *vec3.getXYZ(),
                             *vec1.getXYZ(),
                             *vec3.getXYZ(),
                             *vec4.getXYZ(),

                             *vec2.getXYZ(),
                             *vec6.getXYZ(),
                             *vec7.getXYZ(),
                             *vec2.getXYZ(),
                             *vec7.getXYZ(),
                             *vec3.getXYZ(),

                             *vec3.getXYZ(),
                             *vec7.getXYZ(),
                             *vec8.getXYZ(),
                             *vec3.getXYZ(),
                             *vec8.getXYZ(),
                             *vec4.getXYZ(),

                             *vec4.getXYZ(),
                             *vec8.getXYZ(),
                             *vec5.getXYZ(),
                             *vec4.getXYZ(),
                             *vec5.getXYZ(),
                             *vec1.getXYZ(),

                             *vec1.getXYZ(),
                             *vec5.getXYZ(),
                             *vec6.getXYZ(),
                             *vec1.getXYZ(),
                             *vec6.getXYZ(),
                             *vec2.getXYZ(),

                             *vec8.getXYZ(),
                             *vec7.getXYZ(),
                             *vec6.getXYZ(),
                             *vec8.getXYZ(),
                             *vec6.getXYZ(),
                             *vec5.getXYZ()], dtype=np.float32)
        size = vertices.size * vertices.itemsize
        self.vertexSize_i =  vertices.size

        self.verticesBuffer = gl.glGenBuffers(1)  # Create a buffer.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.verticesBuffer)  # Bind the buffer.
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, vertices, gl.GL_STATIC_DRAW)  # Allocate memory.

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Defines vertex attributes. What are those?
        gl.glEnableVertexAttribArray(0)

        #### Texture Coord ####

        textureCoords = np.array([0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1,

                                  0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1,

                                  0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1,

                                  0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1,

                                  0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1,

                                  0, 1,
                                  0, 0,
                                  1, 0,
                                  0, 1,
                                  1, 0,
                                  1, 1], dtype=np.float32)
        size = textureCoords.size * textureCoords.itemsize

        self.texCoordBuffer = gl.glGenBuffers(1)  # Create a buffer.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.texCoordBuffer)  # Bind the buffer.
        gl.glBufferData(gl.GL_ARRAY_BUFFER, size, textureCoords, gl.GL_STATIC_DRAW)  # Allocate memory.

        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)  # Defines vertex attributes. What are those?
        gl.glEnableVertexAttribArray(1)

    def update(self, timeDelta):
        self.updateActor(timeDelta)
        if self.renderFlag:
            gl.glBindVertexArray(self.vao)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.selectedTexId)

            #### To vertex shader ####

            gl.glUniform1f(3, self.textureHorNum_f)
            gl.glUniform1f(4, self.textureVerNum_f)
            gl.glUniformMatrix4fv(7, 1, gl.GL_FALSE, self.getModelMatrix())

            #### To fragment shader ####

            gl.glUniform1f(11, self.shininess_f)
            gl.glUniform1f(53, self.specularStrength_f)

            ####  ####

            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertexSize_i)

    def _checkCollide(self):
        self.selectedTexId = self.textureId
        for colActor in self.collideActors_l:
            for colModel, colObj in itertools.product(self.collideModels_l, colActor.collideModels_l):
                if co.hitCheckAabbAabb(colModel, colObj):
                    self.collideAction( co.getDistanceToPushBackAabbAabb(colModel, colObj) )

    def collideAction(self, moveToOut_t:tuple):
        if self.textureId2 is not None:
            self.selectedTexId = self.textureId2
            pass

        x, y, z = moveToOut_t
        if abs(x) < abs(y) and abs(x) < abs(z):
            self.pos_l[0] += x
        elif abs(y) < abs(z):
            self.pos_l[1] += y
        else:
            self.pos_l[2] += z


class BoxManager:
    def __init__(self):
        self.boxes_l = []
        self.program = self._getProgram()

    def addBox(self, aBox:Box):
        self.boxes_l.append(aBox)

    def update(self, projectMatrix, viewMatrix, camera:Camera, ambient_t:Tuple[float, float, float],
               lightCount_i:int, lightPos_t:tuple, lightColor_t:tuple, lightMaxDistance_t:tuple,
               spotLightCount_i:int, spotLightPos_t:tuple, spotLightColor_t:tuple, spotLightMaxDistance_t:tuple,
               spotLightDirection_t:tuple, sportLightCutoff_t:tuple, flashLight:bool, timeDelta):

        gl.glUseProgram(self.program)

        # Vertex shader

        gl.glUniformMatrix4fv(5, 1, gl.GL_FALSE, projectMatrix)
        gl.glUniformMatrix4fv(6, 1, gl.GL_FALSE, viewMatrix)

        # Fragment shader

        gl.glUniform3f(8, *camera.getXYZ())
        gl.glUniform3f(9, *ambient_t)

        gl.glUniform1i(10, lightCount_i)
        gl.glUniform3fv(12, lightCount_i, lightPos_t)
        gl.glUniform3fv(17, lightCount_i, lightColor_t)
        gl.glUniform1fv(22, lightCount_i, lightMaxDistance_t)

        if flashLight:
            gl.glUniform1i(27, spotLightCount_i)
            gl.glUniform3fv(28, spotLightCount_i, spotLightPos_t)
            gl.glUniform3fv(33, spotLightCount_i, spotLightColor_t)
            gl.glUniform1fv(43, spotLightCount_i, spotLightMaxDistance_t)
            gl.glUniform3fv(38, spotLightCount_i, spotLightDirection_t)
            gl.glUniform1fv(48, spotLightCount_i, sportLightCutoff_t)
        else:
            gl.glUniform1i(27, 0)

        ####

        for box in self.boxes_l:
            box.update(timeDelta)

    @staticmethod
    def _getProgram() -> int:
        with open("shader_source\\2nd_vs_box.txt") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log_s = gl.glGetShaderInfoLog(vertexShader).decode()
        if log_s:
            raise TypeError(log_s)

        with open("shader_source\\2nd_fs_box.txt") as file:
            fragmentShader = shaders.compileShader(file.read(), gl.GL_FRAGMENT_SHADER)
        log_s = gl.glGetShaderInfoLog(fragmentShader).decode()
        if log_s:
            raise TypeError(log_s)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        gl.glLinkProgram(program)

        print("Linking Log in Box:", gl.glGetProgramiv(program, gl.GL_LINK_STATUS))

        gl.glDeleteShader(vertexShader)
        gl.glDeleteShader(fragmentShader)

        gl.glUseProgram(program)

        return program


class Level:
    def __init__(self, mainLoop:"MainLoop"):
        self.texCon = TextureContainer()
        self.mainLoop = mainLoop

        self.ambient_t = (0.4, 0.4, 0.4)
        self.pointLights_l = [
            PointLight((0,   4, -8),  (0.5, 0.5, 0.5), 10.0),
            PointLight((16,  4,   16),    (1.0, 0.0, 1.0), 10.0),
            PointLight((-16, 4, 16), (0.0, 1.0, 1.0), 10.0)
        ]

        self.spotLights_l = [
            SpotLight((0, 0, 0), (1, 1, 1), 30, cos(mmath.Angle(30).getRadian()), mmath.Vec3(1, 0, 0))
        ]

        self.surfaceProgram = StaticSurface.getProgram()

        size = 20
        self.floor = StaticSurface(
            mmath.Vec3(-size, 0, -size), mmath.Vec3(-size, 0, size), mmath.Vec3(size, 0, size), mmath.Vec3(size, 0, -size),
            mmath.Vec3(0, 1, 0),
            self.texCon[0x21],
            2*size, 2*size, 32, 0.0
        )
        self.wall1 = StaticSurface(
            mmath.Vec3(-size, 5, -size), mmath.Vec3(-size, 0, -size), mmath.Vec3(size, 0, -size), mmath.Vec3(size, 5, -size),
            mmath.Vec3(0, 0, 1),
            self.texCon[0x22],
            5, 2*size, 8, 0.0
        )
        self.wall2 = StaticSurface(
            mmath.Vec3(-size, 5, size), mmath.Vec3(-size, 0, size), mmath.Vec3(-size, 0, -size), mmath.Vec3(-size, 5, -size),
            mmath.Vec3(1, 0, 0),
            self.texCon[0x22],
            5, 2*size, 8, 0.0
        )
        self.wall3 = StaticSurface(
            mmath.Vec3(size, 5, size), mmath.Vec3(size, 0, size), mmath.Vec3(-size, 0, size), mmath.Vec3(-size, 5, size),
            mmath.Vec3(0, 0, -1),
            self.texCon[0x22],
            5, 2*size, 8, 0.0
        )
        self.wall4 = StaticSurface(
            mmath.Vec3(size, 5, -size), mmath.Vec3(size, 0, -size), mmath.Vec3(size, 0, size), mmath.Vec3(size, 5, size),
            mmath.Vec3(-1, 0, 0),
            self.texCon[0x22],
            5, 2*size, 8, 0.0
        )
        self.ceiling = StaticSurface(
            mmath.Vec3(-size, 5, size), mmath.Vec3(-size, 5, -size), mmath.Vec3(size, 5, -size), mmath.Vec3(size, 5, size),
            mmath.Vec3(0, -1, 0), self.texCon[0x12], 2*size, 2*size, 0, 0
        )

        self.boxManager = BoxManager()

        box1 = Box(mmath.Vec3(-0.5, 1.6, -0.5), mmath.Vec3(-0.5, 1.6, 0.5), mmath.Vec3(0.5, 1.6, 0.5), mmath.Vec3(0.5, 1.6, -0.5),
                mmath.Vec3(-0.5, -1, -0.5), mmath.Vec3(-0.5, -1, 0.5), mmath.Vec3(0.5, -1, 0.5), mmath.Vec3(0.5, -1, -0.5),
                self.texCon[0x12], 1, 1, 0, 0.0, [0,1.5,0])
        self.boxManager.addBox(box1)

        self.boxManager.addBox(
            Box(mmath.Vec3(-1, 1, -1), mmath.Vec3(-1, 1, 1), mmath.Vec3(1, 1, 1), mmath.Vec3(1, 1, -1),
                mmath.Vec3(-1, -1, -1), mmath.Vec3(-1, -1, 1), mmath.Vec3(1, -1, 1), mmath.Vec3(1, -1, -1),
                self.texCon[0x12], 1, 1, 0, 0.0, [0,1,-17])
        )

        box2 = self.boxManager.boxes_l[1]

        level = Actor()
        level.collideModels_l.append(co.Aabb( mmath.Vec3(-size, -1, -size), mmath.Vec3(size, 0, size) ))

        level.collideModels_l.append(co.Aabb( mmath.Vec3(-size, 0, -size), mmath.Vec3(size, 5, -size-1) ))
        level.collideModels_l.append(co.Aabb( mmath.Vec3(-size, 0, -size), mmath.Vec3(-size-1, 5, size) ))
        level.collideModels_l.append(co.Aabb( mmath.Vec3(-size, 0, size), mmath.Vec3(size, 5, size+1) ))
        level.collideModels_l.append(co.Aabb( mmath.Vec3(size, 0, size), mmath.Vec3(size+1, 5, -size) ))

        level.collideModels_l.append(co.Aabb( mmath.Vec3(-size, -1, -size), mmath.Vec3(size, 0, size) ))

        box1.collideActors_l.append(box2)
        box1.collideActors_l.append(level)
        box1.textureId2 = self.texCon[0x22]
        box1.physics = True

        self.loadedModelManager = ol.LoadedModelManager()

    def update(self, timeDelta, projectMatrix, viewMatrix, camera:Camera, flashLight):
        x, y, z = camera.getXYZ()
        y -= 0.5

        first = self.spotLights_l[0]
        first.parent = self.mainLoop.camera
        a = np.matrix([0, 0, 0, 1], np.float32) * first.getModelMatrix()
        pos = a.item(0), a.item(1), a.item(2)
        b = np.matrix([*first.directionVec3.getXYZ(), 0.0,], np.float32) * first.getModelMatrix()
        direction = b.item(0), b.item(1), b.item(2)

        lightPos_t = tuple()
        lightColor_t = tuple()
        lightMaxDistance_t = tuple()
        lightCount_i = 0
        for x in self.pointLights_l:
            lightPos_t += x.getXYZ()
            lightColor_t += x.getRGB()
            lightMaxDistance_t += (x.maxDistance_f,)
            lightCount_i += 1

        spotLightPos_t = list()
        spotLightColor_t = tuple()
        spotLightMaxDistance_t = tuple()
        spotLightDirection_t = tuple()
        sportLightCutoff_t = tuple()
        spotLightCount_i = 0
        for x in self.spotLights_l:
            spotLightPos_t += pos
            spotLightColor_t += x.getRGB()
            spotLightMaxDistance_t += (x.maxDistance_f,)
            spotLightDirection_t += direction[:3]
            sportLightCutoff_t += (x.cutoff_f,)
            spotLightCount_i += 1

        gl.glUseProgram(self.surfaceProgram)

        # Vertex shader

        gl.glUniformMatrix4fv(5, 1, gl.GL_FALSE, projectMatrix)
        gl.glUniformMatrix4fv(6, 1, gl.GL_FALSE, viewMatrix)
        gl.glUniformMatrix4fv(7, 1, gl.GL_FALSE, mmath.identityMat4())

        # Fragment shader

        gl.glUniform3f(8, *camera.getXYZ())
        gl.glUniform3f(9, *self.ambient_t)

        gl.glUniform1i(10, lightCount_i)
        gl.glUniform3fv(12, lightCount_i, lightPos_t)
        gl.glUniform3fv(17, lightCount_i, lightColor_t)
        gl.glUniform1fv(22, lightCount_i, lightMaxDistance_t)

        if flashLight:
            gl.glUniform1i(27, spotLightCount_i)
            gl.glUniform3fv(28, spotLightCount_i, spotLightPos_t)
            gl.glUniform3fv(33, spotLightCount_i, spotLightColor_t)
            gl.glUniform1fv(43, spotLightCount_i, spotLightMaxDistance_t)
            gl.glUniform3fv(38, spotLightCount_i, spotLightDirection_t)
            gl.glUniform1fv(48, spotLightCount_i, sportLightCutoff_t)
        else:
            gl.glUniform1i(27, 0)

        ####

        self.floor.update()
        self.wall1.update()
        self.wall2.update()
        self.wall3.update()
        self.wall4.update()
        self.ceiling.update()

        self.boxManager.update(projectMatrix, viewMatrix, camera, self.ambient_t,
                               lightCount_i, lightPos_t, lightColor_t, lightMaxDistance_t,
                               spotLightCount_i, spotLightPos_t, spotLightColor_t, spotLightMaxDistance_t,
                               spotLightDirection_t, sportLightCutoff_t, flashLight, timeDelta)

        self.loadedModelManager.update(
            projectMatrix, viewMatrix, camera, self.ambient_t,
            lightCount_i, lightPos_t, lightColor_t, lightMaxDistance_t,
            spotLightCount_i, spotLightPos_t, spotLightColor_t, spotLightMaxDistance_t,
            spotLightDirection_t, sportLightCutoff_t, flashLight
        )


class MainLoop:
    def __init__(self):
        p.init()
        p.font.init()
        self.winSize_t = (800, 600)
        self.centerPos_t = (self.winSize_t[0] / 2, self.winSize_t[1] / 2)
        self.dSurf = p.display.set_mode(self.winSize_t, pl.DOUBLEBUF | pl.OPENGL | pl.RESIZABLE)

        self.initGL()

        self.level = Level(self)
        self.camera = Camera()
        self.controller = Controller(self, self.camera)
        self.fManager = mp.FrameManager(True)

        self.flashLight_b = True

        self.projectMatrix = None

    @staticmethod
    def initGL():
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        pass

    def update(self):
        self.fManager.update()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearBufferfv(gl.GL_COLOR, 0, (0.0, 0.0, 0.0, 1.0))

        for event in p.event.get():
            if event.type == pl.QUIT:
                p.quit()
                sys.exit(0)
            elif event.type == pl.VIDEORESIZE:
                self.onResize(event.dict['w'], event.dict['h'])

        self.controller.update(self.fManager.getFrameDelta())

        if False:
            self.drawText((-0.95, 0.9, 0), "FPS : {}".format(self.fManager.getFPS()[0]))
            self.drawText((-0.95, 0.8, 0), "Pos : {:.2f}, {:.2f}, {:.2f}".format(*self.camera.pos_l))
            self.drawText((-0.95, 0.7, 0), "Looking : {:.2f}, {:.2f}".format(self.camera.lookHorDeg_f,
                                                                             self.camera.lookVerDeg_f))
        hor, ver = self.camera.getWorldDegree()
        viewMatrix = mmath.translateMat4(*self.camera.getWorldXYZ(), -1) *\
                     mmath.rotateMat4(hor, 0, 1, 0) *\
                     mmath.rotateMat4(ver, 1, 0, 0)
        self.level.update(self.fManager.getFrameDelta(), self.projectMatrix, viewMatrix, self.camera, self.flashLight_b)

        p.display.flip()

    def onResize(self, w, h):
        self.winSize_t = (w, h)
        self.centerPos_t = (w / 2, h / 2)
        gl.glViewport(0, 0, w, h)
        self.projectMatrix = mmath.perspectiveMat4(90.0, w / h, 0.1, 1000.0)

    @staticmethod
    def drawText(position, textString):
        font = p.font.Font(None, 32)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = p.image.tostring(textSurface, "RGBA", True)
        gl.glRasterPos3d(*position)
        gl.glDrawPixels(textSurface.get_width(), textSurface.get_height(), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textData)


def main():
    mainLoop = MainLoop()
    while True:
        mainLoop.update()


if __name__ == '__main__':
    main()
