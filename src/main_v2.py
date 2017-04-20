import traceback
import sys
import itertools
from time import time, sleep
from math import cos, sin, pi
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

        self.oldMouseState_t = p.mouse.get_pressed()
        self.newMouseState_t = p.mouse.get_pressed()

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

        self.oldMouseState_t = self.newMouseState_t
        self.newMouseState_t = p.mouse.get_pressed()

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
        if self.getStateChange(pl.K_F8) == 1:
            self.mainLoop.level.loadedModelManager.obj_l[0].load()
            self.mainLoop.worldLoaded_b = True
        if self.getStateChange(pl.K_F9) == 1:
            if self.mainLoop.lightSourceMode_b:
                self.mainLoop.lightSourceMode_b = False
            else:
                self.mainLoop.lightSourceMode_b = True
        if self.getStateChange(pl.K_F10) == 1:
            self.mainLoop.level.boxManager.rotateSpeed_i -= 10
        if self.getStateChange(pl.K_F11) == 1:
            self.mainLoop.level.boxManager.rotateSpeed_i += 10

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
                       0x12:self.getTexture("assets\\textures\\12.bmp"),
                       0x14:self.getTexture("assets\\textures\\14.png"),
                       0x13:self.getTexture("assets\\textures\\13.png")}

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError(item)

        return self.data_d[item]

    @staticmethod
    def getTexture(textureDir_s:str):
        aImg = Image.open(textureDir_s)
        imgW_i = aImg.size[0]
        imgH_i = aImg.size[1]
        try:
            image_bytes = aImg.tobytes("raw", "RGBA", 0, -1)
            alpha_b = True
        except ValueError:
            image_bytes = aImg.tobytes("raw", "RGBX", 0, -1)
            alpha_b = False
        imgArray = np.array([x / 255 for x in image_bytes], dtype=np.float32)

        texId = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texId)
        gl.glTexStorage2D(gl.GL_TEXTURE_2D, 6, gl.GL_RGBA32F, imgW_i, imgH_i)

        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, imgW_i, imgH_i, gl.GL_RGBA, gl.GL_FLOAT, imgArray)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 6)

        if alpha_b and False:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

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

        gl.glActiveTexture(gl.GL_TEXTURE0)
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

    def drawForShadow(self):
        gl.glBindVertexArray(self.vao)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)

        gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, mmath.identityMat4())

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    @staticmethod
    def getProgram() -> int:
        with open("shader_source\\2nd_vs.glsl") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log_s = gl.glGetShaderInfoLog(vertexShader).decode()
        if log_s:
            raise TypeError(log_s)

        with open("shader_source\\2nd_fs.glsl") as file:
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
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.selectedTexId)

            #### To vertex shader ####

            gl.glUniform1f(3, self.textureHorNum_f)
            gl.glUniform1f(4, self.textureVerNum_f)
            gl.glUniformMatrix4fv(7, 1, gl.GL_FALSE, self.getModelMatrix())

            #### To fragment shader ####

            gl.glUniform1f(11, self.shininess_f)
            gl.glUniform1f(54, self.specularStrength_f)

            ####  ####

            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertexSize_i)

    def drawForShadow(self, timeDelta):
        self.updateActor(timeDelta)
        if self.renderFlag or True:
            gl.glBindVertexArray(self.vao)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.selectedTexId)

            #### To vertex shader ####

            gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, self.getModelMatrix())

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
        self.rotating_l = []
        self.program = self._getProgram()

        self.rotateSpeed_i = 20

    def addBox(self, aBox:Box):
        self.boxes_l.append(aBox)

    def update(self, projectMatrix, viewMatrix, camera:Camera, ambient_t:Tuple[float, float, float],
               lightCount_i:int, lightPos_t:tuple, lightColor_t:tuple, lightMaxDistance_t:tuple,
               spotLightCount_i:int, spotLightPos_t:tuple, spotLightColor_t:tuple, spotLightMaxDistance_t:tuple,
               spotLightDirection_t:tuple, sportLightCutoff_t:tuple, flashLight:bool, timeDelta, shadowMat, depthMap,
               sunLightColor, sunLightDirection:mmath.Vec4):

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

        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "lightSpaceMatrix"), 1, gl.GL_FALSE, shadowMat)

        if flashLight:
            gl.glUniform1i(27, spotLightCount_i)
            gl.glUniform3fv(28, spotLightCount_i, spotLightPos_t)
            gl.glUniform3fv(33, spotLightCount_i, spotLightColor_t)
            gl.glUniform1fv(43, spotLightCount_i, spotLightMaxDistance_t)
            gl.glUniform3fv(38, spotLightCount_i, spotLightDirection_t)
            gl.glUniform1fv(48, spotLightCount_i, sportLightCutoff_t)
        else:
            gl.glUniform1i(27, 0)

        gl.glUniform1i(56, 0)
        gl.glUniform1i(57, 1)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, depthMap)

        gl.glUniform3f(gl.glGetUniformLocation(self.program, "sunLightColor"), *sunLightColor)
        gl.glUniform3f(gl.glGetUniformLocation(self.program, "sunLightDirection"), *sunLightDirection.getXYZ())

        ####

        for box in self.boxes_l:
            box.update(timeDelta)

        for box in self.rotating_l:
            box.update(timeDelta)

    def drawForShadow(self, timeDelta):
        for box in self.boxes_l:
            box.drawForShadow(timeDelta)

        for x, box in enumerate(self.rotating_l):
            angle = (x / 50 * 360 + time()*self.rotateSpeed_i)
            radian = angle / 180 * pi
            box.pos_l = [sin(radian)*40, sin(radian)*5+7, cos(radian)*40]
            box.lookHorDeg_f = angle
            box.drawForShadow(timeDelta)

    @staticmethod
    def _getProgram() -> int:
        with open("shader_source\\2nd_vs_box.glsl") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log_s = gl.glGetShaderInfoLog(vertexShader).decode()
        if log_s:
            raise TypeError(log_s)

        with open("shader_source\\2nd_fs_box.glsl") as file:
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
    def __init__(self, mainLoop:"MainLoop", depthMap):
        self.texCon = TextureContainer()
        self.mainLoop = mainLoop
        self.depthMap = depthMap

        self.ambient_t = (0.5, 0.5, 0.5)
        self.sunLightColor = (0.4, 0.4, 0.4)
        self.dynamicLight = PointLight((-25, 2, -25), (1, 1, 1), 20)
        self.pointLights_l = [
            PointLight((0,   4, -12),  (0.5, 0.5, 0.5), 10.0),
            PointLight((25,  6,   25),    (1.0, 0.0, 1.0), 10.0),
            PointLight((-25, 4, 25), (0.0, 1.0, 1.0), 10.0),
            self.dynamicLight
        ]
        #self.pointLights_l = []

        self.spotLights_l = [
            SpotLight((0, 0, 0), (1, 1, 1), 30, cos(mmath.Angle(30).getRadian()), mmath.Vec3(1, 0, 0))
        ]

        self.surfaceProgram = StaticSurface.getProgram()

        size = 50
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
            self.texCon[0x13],
            5, 2*size, 8, 0.0
        )
        self.ceiling = StaticSurface(
            mmath.Vec3(-size, 5, size), mmath.Vec3(-size, 5, -size), mmath.Vec3(size, 5, -size), mmath.Vec3(size, 5, size),
            mmath.Vec3(0, -1, 0), self.texCon[0x12], 2*size, 2*size, 0, 0
        )

        self.display = StaticSurfaceShadow(
            mmath.Vec3(-2, 4, -5), mmath.Vec3(-2, 0, -5), mmath.Vec3(2, 0, -5), mmath.Vec3(2, 4, -5),
            mmath.Vec3(0, 0, 1), depthMap, 1, 1, 0, 0
        )

        self.display2 = StaticSurfaceShadow(
            mmath.Vec3(2, 4, -5),mmath.Vec3(2, 0, -5),mmath.Vec3(-2, 0, -5),mmath.Vec3(-2, 4, -5),
            mmath.Vec3(0, 0, 1), depthMap, 1, 1, 0, 0
        )

        self.boxManager = BoxManager()

        box1 = Box(
            mmath.Vec3(-0.5, 1.6, -0.5), mmath.Vec3(-0.5, 1.6, 0.5), mmath.Vec3(0.5, 1.6, 0.5), mmath.Vec3(0.5, 1.6, -0.5),
            mmath.Vec3(-0.5, -1, -0.5), mmath.Vec3(-0.5, -1, 0.5), mmath.Vec3(0.5, -1, 0.5), mmath.Vec3(0.5, -1, -0.5),
            self.texCon[0x12], 1, 1, 0, 0.0, [0,1.5,0]
        )
        self.boxManager.addBox(box1)

        self.boxManager.addBox(
            Box(mmath.Vec3(-1, 1, -1), mmath.Vec3(-1, 1, 1), mmath.Vec3(1, 1, 1), mmath.Vec3(1, 1, -1),
                mmath.Vec3(-1, -1, -1), mmath.Vec3(-1, -1, 1), mmath.Vec3(1, -1, 1), mmath.Vec3(1, -1, -1),
                self.texCon[0x12], 1, 1, 0, 0.0, [0,1,-17])
        )
        self.boxManager.addBox(
            Box(mmath.Vec3(-1, 1, -1), mmath.Vec3(-1, 1, 1), mmath.Vec3(1, 1, 1), mmath.Vec3(1, 1, -1),
                mmath.Vec3(-1, -1, -1), mmath.Vec3(-1, -1, 1), mmath.Vec3(1, -1, 1), mmath.Vec3(1, -1, -1),
                self.texCon[0x12], 1, 1, 0, 0.0, [5, 1, -17])
        )

        self.boxManager.addBox(
            Box(mmath.Vec3(-1, 1, -1), mmath.Vec3(-1, 1, 1), mmath.Vec3(1, 1, 1), mmath.Vec3(1, 1, -1),
                mmath.Vec3(-1, -1, -1), mmath.Vec3(-1, -1, 1), mmath.Vec3(1, -1, 1), mmath.Vec3(1, -1, -1),
                self.texCon[0x12], 1, 1, 0, 0.0, [5, 3.5, -17])
        )

        for x in range(50):
            angle = (x/50*360) / 180 * pi
            self.boxManager.rotating_l.append(
                Box(mmath.Vec3(-1, 1, -1), mmath.Vec3(-1, 1, 1), mmath.Vec3(1, 1, 1), mmath.Vec3(1, 1, -1),
                    mmath.Vec3(-1, -1, -1), mmath.Vec3(-1, -1, 1), mmath.Vec3(1, -1, 1), mmath.Vec3(1, -1, -1),
                    self.texCon[0x12], 1, 1, 0, 0.0, [sin(angle)*40, 3.5, cos(angle)*40])
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
        box1.collideActors_l.append(self.boxManager.boxes_l[2])
        box1.collideActors_l.append(self.boxManager.boxes_l[3])
        for box in self.boxManager.rotating_l:
            box1.collideActors_l.append(box)
        box1.collideActors_l.append(level)
        box1.textureId2 = self.texCon[0x22]
        box1.physics = True

        self.loadedModelManager = ol.LoadedModelManager()

        self.grass = StaticSurface(
            mmath.Vec3(-1, 2, 3), mmath.Vec3(-1, 0, 3), mmath.Vec3(1, 0, 3), mmath.Vec3(1, 2, 3),
            mmath.Vec3(0, 0, 1), self.texCon[0x14], 1, 1, 0, 0
        )

    def update(self, timeDelta, projectMatrix, viewMatrix, camera:Camera, flashLight, shadowMat, sunLightDirection):
        a = sunLightDirection.dot(mmath.Vec4(0, -1, -1, 0)) - 0.5
        if a < 0.0:
            a = 0.0
        a = a*2
        if a< 0.2:
            a = 0.2
        elif a > 1.0:
            a = 1.0

        if False and 0.0 < a < 0.9:
            self.sunLightColor = (0.7, 0.3, 0.3)
            gl.glClearBufferfv(gl.GL_COLOR, 0, (a, a / 2, a/2, 1.0))
        else:
            self.sunLightColor = (0.5, 0.5, 0.5)
            gl.glClearBufferfv(gl.GL_COLOR, 0, ( a / 2 , a / 2 , a, 1.0))

        self.ambient_t = (0.25, 0.25, 0.25)

        self.dynamicLight.r = sin( (time()+1)/2 )
        self.dynamicLight.g = cos( (time()+1)/2 )
        self.dynamicLight.b = sin( (time()+1)/2) * cos((time()+1)/2 )

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

        self.boxManager.update(projectMatrix, viewMatrix, camera, self.ambient_t,
                               lightCount_i, lightPos_t, lightColor_t, lightMaxDistance_t,
                               spotLightCount_i, spotLightPos_t, spotLightColor_t, spotLightMaxDistance_t,
                               spotLightDirection_t, sportLightCutoff_t, flashLight, timeDelta, shadowMat,
                               self.depthMap, self.sunLightColor, sunLightDirection)

        self.loadedModelManager.update(
            projectMatrix, viewMatrix, camera, self.ambient_t,
            lightCount_i, lightPos_t, lightColor_t, lightMaxDistance_t,
            spotLightCount_i, spotLightPos_t, spotLightColor_t, spotLightMaxDistance_t,
            spotLightDirection_t, sportLightCutoff_t, flashLight, shadowMat, self.depthMap, self.sunLightColor, sunLightDirection
        )

        gl.glUseProgram(self.display.program)

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

        self.display.update()
        self.display2.update()

        gl.glUseProgram(self.surfaceProgram)

        # Vertex shader

        gl.glUniformMatrix4fv(5, 1, gl.GL_FALSE, projectMatrix)
        gl.glUniformMatrix4fv(6, 1, gl.GL_FALSE, viewMatrix)
        gl.glUniformMatrix4fv(7, 1, gl.GL_FALSE, mmath.identityMat4())

        gl.glUniformMatrix4fv(54, 1, gl.GL_FALSE, shadowMat)

        # Fragment shader

        gl.glUniform3f(8, *camera.getXYZ())
        gl.glUniform3f(9, *self.ambient_t)

        gl.glUniform1i(10, lightCount_i)
        gl.glUniform3fv(12, lightCount_i, lightPos_t)
        gl.glUniform3fv(17, lightCount_i, lightColor_t)
        gl.glUniform1fv(22, lightCount_i, lightMaxDistance_t)

        gl.glUniform1i(55, 0)
        gl.glUniform1i(56, 1)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depthMap)

        if flashLight:
            gl.glUniform1i(27, spotLightCount_i)
            gl.glUniform3fv(28, spotLightCount_i, spotLightPos_t)
            gl.glUniform3fv(33, spotLightCount_i, spotLightColor_t)
            gl.glUniform1fv(43, spotLightCount_i, spotLightMaxDistance_t)
            gl.glUniform3fv(38, spotLightCount_i, spotLightDirection_t)
            gl.glUniform1fv(48, spotLightCount_i, sportLightCutoff_t)
        else:
            gl.glUniform1i(27, 0)

        gl.glUniform3f(gl.glGetUniformLocation(self.surfaceProgram, "sunLightColor"), *self.sunLightColor)
        gl.glUniform3f(gl.glGetUniformLocation(self.surfaceProgram, "sunLightDirection"), *sunLightDirection.getXYZ())

        ####

        self.floor.update()
        self.wall1.update()
        self.wall2.update()
        self.wall3.update()
        #self.ceiling.update()
        self.wall4.update()
        self.grass.update()

    def drawForShadow(self, timeDelta):
        self.floor.drawForShadow()
        self.wall1.drawForShadow()
        self.wall2.drawForShadow()
        self.wall3.drawForShadow()
        #self.ceiling.drawForShadow()

        self.wall4.drawForShadow()
        self.grass.drawForShadow()

        self.boxManager.drawForShadow(timeDelta)

        self.loadedModelManager.drawForShadow()


class StaticSurfaceShadow:
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
        self.program = self.getProgram()

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

        gl.glActiveTexture(gl.GL_TEXTURE0)
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

    def drawForShadow(self):
        gl.glBindVertexArray(self.vao)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textureId)

        gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, mmath.identityMat4())

        #gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

    @staticmethod
    def getProgram() -> int:
        with open("shader_source\\vs_shadow_draw.glsl") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log_s = gl.glGetShaderInfoLog(vertexShader).decode()
        if log_s:
            raise TypeError(log_s)

        with open("shader_source\\fs_shadow_draw.glsl") as file:
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


class ShadowMap:
    def __init__(self):
        self.depthMapFbo = gl.glGenFramebuffers(1)

        self.shadowW_i = 1024*4
        self.shadowH_i = 1024*4

        self.program = self._getProgram()

        self.depthMapTex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depthMapTex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, self.shadowW_i, self.shadowH_i, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        #gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (1.0, 1.0, 1.0, 1.0))

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depthMapFbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.depthMapTex, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print( "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def renderDepthMap(self, lightProjection, lightView, level:Level, timeDelta):
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glUseProgram(self.program)
        gl.glViewport(0, 0, self.shadowW_i, self.shadowH_i)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.depthMapFbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        gl.glUniformMatrix4fv( 1, 1, gl.GL_FALSE, lightView * lightProjection )

        level.drawForShadow(timeDelta)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    @staticmethod
    def _getProgram() -> int:
        with open("shader_source\\vs_shadow.glsl") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log_s = gl.glGetShaderInfoLog(vertexShader).decode()
        if log_s:
            raise TypeError(log_s)

        with open("shader_source\\fs_shadow.glsl") as file:
            fragmentShader = shaders.compileShader(file.read(), gl.GL_FRAGMENT_SHADER)
        log_s = gl.glGetShaderInfoLog(fragmentShader).decode()
        if log_s:
            raise TypeError(log_s)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        gl.glLinkProgram(program)

        print("Linking Log in Shadow:", gl.glGetProgramiv(program, gl.GL_LINK_STATUS))

        gl.glDeleteShader(vertexShader)
        gl.glDeleteShader(fragmentShader)

        gl.glUseProgram(program)

        return program


class FrameBuffer:
    def __init__(self):
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 800, 600, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR);
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR);

        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, texture, 0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDeleteFramebuffers(1, self.fbo)


class MainLoop:
    def __init__(self):
        p.init()
        p.font.init()
        self.winSize_t = (800, 600)
        self.centerPos_t = (self.winSize_t[0] / 2, self.winSize_t[1] / 2)
        self.dSurf = p.display.set_mode(self.winSize_t, pl.DOUBLEBUF | pl.OPENGL | pl.RESIZABLE)
        p.display.set_caption("First Person Practice")

        self.initGL()

        self.shadowMap = ShadowMap()
        self.level = Level(self, self.shadowMap.depthMapTex)
        self.camera = Camera()
        self.controller = Controller(self, self.camera)
        self.fManager = mp.FrameManager(True)

        self.flashLight_b = True

        self.projectMatrix = None
        self.lightSourceMode_b = False

        self.worldLoaded_b = False

    @staticmethod
    def initGL():
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        pass

    def update(self):
        self.fManager.update()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for event in p.event.get():
            if event.type == pl.QUIT:
                p.quit()
                sys.exit(0)
            elif event.type == pl.VIDEORESIZE:
                self.winSize_t = (event.dict['w'], event.dict['h'])
                self.onResize()

        self.controller.update(self.fManager.getFrameDelta())

        if False:
            self.drawText((-0.95, 0.9, 0), "FPS : {}".format(self.fManager.getFPS()[0]))
            self.drawText((-0.95, 0.8, 0), "Pos : {:.2f}, {:.2f}, {:.2f}".format(*self.camera.pos_l))
            self.drawText((-0.95, 0.7, 0), "Looking : {:.2f}, {:.2f}".format(self.camera.lookHorDeg_f,
                                                                             self.camera.lookVerDeg_f))
        hor, ver = self.camera.getWorldDegree()
        viewMatrix = mmath.translateMat4(*self.camera.getWorldXYZ(), -1) * mmath.rotateMat4(hor, 0, 1, 0) * mmath.rotateMat4(ver, 1, 0, 0)
        # viewMatrix = mmath.translateMat4(*self.camera.getWorldXYZ(), -1) * mmath.getlookatMat4( mmath.Vec4(*self.camera.getWorldXYZ(),1), mmath.Vec4(0,0,0,1), mmath.Vec4(0, 1, 0, 0) )

        if self.worldLoaded_b:
            lightProjection = mmath.orthoMat4(-400.0, 400.0, -400.0, 400.0, -300.0, 300.0)
        else:
            lightProjection = mmath.orthoMat4(-75.0, 75.0, -75.0, 75.0, -75.0, 75.0)

        if self.lightSourceMode_b:
            lightView = mmath.getlookatMat4(mmath.Vec4(*self.camera.getWorldXYZ(), 1), mmath.Vec4(0, 0, 0, 1), mmath.Vec4(0, 1, 0, 0))
        else:
            sunLightDirection = mmath.Vec4(0, -1, -0.25, 0).normalize()
            sunLightDirection = sunLightDirection.transform(mmath.rotateMat4(time()*10%360, 0, 0, -1))
            lightView = mmath.getlookatMat4(mmath.Vec4(0, 0, 0, 1), mmath.Vec4(*sunLightDirection.getXYZ(), 0), mmath.Vec4(0, 1, 0, 0))

        self.shadowMap.renderDepthMap(lightProjection, lightView, self.level, self.fManager.getFrameDelta())
        self.onResize()
        self.level.update(self.fManager.getFrameDelta(), self.projectMatrix, viewMatrix, self.camera, self.flashLight_b, lightView*lightProjection, sunLightDirection)

        p.display.flip()

    def onResize(self):
        w, h = self.winSize_t
        self.centerPos_t = (w / 2, h / 2)
        gl.glViewport(0, 0, w, h)
        self.projectMatrix = mmath.perspectiveMat4(90.0, w / h, 0.1, 1000.0)
        #self.projectMatrix = mmath.orthoMat4(-10.0, 10.0, -10.0, 10.0, 0.0, 1000.0)

    @staticmethod
    def drawText(position, textString):
        font = p.font.Font(None, 32)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = p.image.tostring(textSurface, "RGBA", True)
        gl.glRasterPos3d(*position)
        gl.glDrawPixels(textSurface.get_width(), textSurface.get_height(), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textData)


def main():
    try:
        mainLoop = MainLoop()
        while True:
            mainLoop.update()
    except SystemExit:  # When the app is closed my clicking close button.
        pass
    except:  # I want to see what exception crashed my god damn app.
        p.quit()
        print('SERIOUS ERROR OCCURRED!!')
        traceback.print_exc()
        sleep(1)
        input("Press any key to continue...")


if __name__ == '__main__':
    main()
