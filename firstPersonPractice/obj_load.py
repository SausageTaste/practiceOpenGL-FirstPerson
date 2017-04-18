from time import time
from typing import Tuple

import OpenGL.GL as gl
from OpenGL.GL import shaders
import numpy as np

import main_v2 as mainpy
from actor import Actor
from camera import Camera


class LoadedModelManager:
    """
    This class has common things for LoadedModel's instances.
    Such as shader program and tons of parameters for self.update().
    """
    def __init__(self):
        self.obj_l = []
        self.program = self._getProgram()

        self.obj_l.append( LoadedModel("assets\\models\\seoul_v2.obj", initPos_t=[0, -50, 0], initScale=[50, 50, 50]) )



    def update(self, projectMatrix, viewMatrix, camera:Camera, ambient_t:Tuple[float, float, float],
               lightCount_i:int, lightPos_t:tuple, lightColor_t:tuple, lightMaxDistance_t: tuple,
               spotLightCount_i:int, spotLightPos_t:tuple, spotLightColor_t:tuple, spotLightMaxDistance_t:tuple,
               spotLightDirection_t:tuple, sportLightCutoff_t:tuple, flashLight:bool, depthMap, sunLightColor):
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

        gl.glUniformMatrix4fv(gl.glGetUniformLocation(self.program, "lightSpaceMatrix"), 1, gl.GL_FALSE, depthMap)

        gl.glUniform1i(56, 0)
        gl.glUniform1i(57, 1)

        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, depthMap)

        gl.glUniform3f(gl.glGetUniformLocation(self.program, "sunLightColor"), *sunLightColor)

        ####

        for obj in self.obj_l:
            obj.update()

    def drawForShadow(self):
        for obj in self.obj_l:
            obj.drawForShadow()

    def loadAll(self):
        for x in self.obj_l:
            if not x.loaded:
                x.load()

    @staticmethod
    def _getProgram() -> int:
        with open("shader_source\\2nd_vs_loaded.txt") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log_s = gl.glGetShaderInfoLog(vertexShader).decode()
        if log_s:
            raise TypeError(log_s)

        with open("shader_source\\2nd_fs_loaded.txt") as file:
            fragmentShader = shaders.compileShader(file.read(), gl.GL_FRAGMENT_SHADER)
        log_s = gl.glGetShaderInfoLog(fragmentShader).decode()
        if log_s:
            raise TypeError(log_s)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        gl.glLinkProgram(program)

        print("Linking Log in LoadedModel:", gl.glGetProgramiv(program, gl.GL_LINK_STATUS))

        gl.glDeleteShader(vertexShader)
        gl.glDeleteShader(fragmentShader)

        gl.glUseProgram(program)

        return program


class LoadedModel(Actor):
    """
    This class is child class of Actor.
    
    This class's instance gets a location of obj file, loads models.
    Also it load mtl file that has same name as obj file.
    A pair of files has vertices, texture coor, vertex normal, texture file location.
    
    Draw loaded models every time self.update() is called.
    self.update() is called in LoadedModelManager.update().
    """
    def __init__(self, objFileDir_s:str, initPos_t=(0, 0, 0), initScale=(1, 1, 1)):
        super().__init__(initPos_t=initPos_t, initScale=initScale)
        self.__objDir_s = objFileDir_s

        self.renderFlag_b = True
        self.shininess_f = 32.0
        self.specularStrength_f = 1.0
        self.loaded = False

        self.vertices_d = {}
        self.materials_d = {}

    def update(self):
        if not self.loaded:
            return None
        if not self.renderFlag_b:
            return None

        for key_s in self.vertices_d:
            obj_d = self.vertices_d[key_s]
            gl.glBindVertexArray(obj_d["vao"])
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, obj_d["txid"])

            #### To vertex shader ####

            gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, self.getModelMatrix())

            #### To fragment shader ####

            gl.glUniform1f(11, self.shininess_f)
            gl.glUniform1f(53, self.specularStrength_f)

            ####  ####

            gl.glDrawArrays(gl.GL_TRIANGLES, 0, obj_d["ver_num"])

    def drawForShadow(self):
        if not self.loaded:
            return None
        if not self.renderFlag_b:
            return None

        for key_s in self.vertices_d:
            obj_d = self.vertices_d[key_s]
            gl.glBindVertexArray(obj_d["vao"])
            gl.glBindTexture(gl.GL_TEXTURE_2D, obj_d["txid"])

            gl.glUniformMatrix4fv(3, 1, gl.GL_FALSE, self.getModelMatrix())

            gl.glDrawArrays(gl.GL_TRIANGLES, 0, obj_d["ver_num"])

    def getObjDir(self):
        return self.__objDir_s

    def getMtlDir(self):
        return self.getObjDir()[:self.getObjDir().rindex('.')] + ".mtl"

    def load(self):
        self._loadMaterial()
        self._loadVertices()

        self.loaded = True

    def _loadVertices_dum(self):
        vertices_d = {}

        with open(self.getObjDir()) as file:
            for x_s in file:
                x_s = x_s.rstrip('\n')

                if x_s.startswith("o "):
                    curObj_s = x_s.split()[1]
                    if curObj_s in vertices_d:
                        raise FileExistsError
                    vertices_d[curObj_s] = {}
                elif x_s.startswith("v "):
                    if curObj_s not in vertices_d:
                        raise FileNotFoundError
                    if "v" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["v"] = []
                    _, x, y, z = x_s.split()
                    vertices_d[curObj_s]["v"].append( ( float(x), float(y), float(z) ) )
                    del _, x, y, z
                elif x_s.startswith("vt "):
                    if curObj_s not in vertices_d:
                        raise FileNotFoundError
                    if "vt" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["vt"] = []
                    _, x, y = x_s.split()
                    vertices_d[curObj_s]["vt"].append( ( float(x), float(y) ) )
                    del _, x, y
                elif x_s.startswith("vn "):
                    if curObj_s not in vertices_d:
                        raise FileNotFoundError
                    if "vn" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["vn"] = []
                    _, x, y, z = x_s.split()
                    vertices_d[curObj_s]["vn"].append( ( float(x), float(y), float(z) ) )
                    del _, x, y, z
                elif x_s.startswith("usemtl "):
                    if curObj_s not in vertices_d:
                        raise FileNotFoundError
                    vertices_d[curObj_s]["mtl"] = x_s.split()[1]
                elif x_s.startswith("f "):
                    if curObj_s not in vertices_d:
                        raise FileNotFoundError
                    if "v_i" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["v_i"] = []
                    if "vt_i" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["vt_i"] = []
                    if "vn_i" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["vn_i"] = []

                    _, v0_s, v1_s, v2_s = x_s.split()
                    verticesIndices_t = (
                        tuple(map(lambda xx: int(xx), v0_s.split('/'))),
                        tuple(map(lambda xx: int(xx), v1_s.split('/'))),
                        tuple(map(lambda xx: int(xx), v2_s.split('/')))
                    )

                    for v_l in verticesIndices_t:
                        try:
                            vertices_d[curObj_s]["v_i" ].append( vertices_d[curObj_s]["v" ][v_l[0] - 1] )
                        except IndexError:
                            print( "{}: {}, {} << {}".format( curObj_s, v_l[0] - 1, len(vertices_d[curObj_s]["v" ]), x_s ) )
                        try:
                            vertices_d[curObj_s]["vt_i"].append(vertices_d[curObj_s]["vt"][v_l[1] - 1])
                        except IndexError:
                            print( "{}: {}, {} << {}".format( curObj_s, v_l[1] - 1, len(vertices_d[curObj_s]["vt" ]), x_s ) )
                        try:
                            vertices_d[curObj_s]["vn_i"].append(vertices_d[curObj_s]["vn"][v_l[2] - 1])
                        except IndexError:
                            print( "{}: {}, {} << {}".format( curObj_s, v_l[2] - 1, len(vertices_d[curObj_s]["vn" ]), x_s ) )

        del vertices_d[curObj_s]["v"], vertices_d[curObj_s]["vt"], vertices_d[curObj_s]["vn"]

        for x in vertices_d:
            aa = vertices_d[x]
            assert len(aa["vt_i"]) == len(aa["vn_i"]) == len(aa["v_i"])
            del x, aa

        self.vertices_d = {}
        for objName_s in vertices_d.keys():
            self.vertices_d[objName_s] = {}
            localObj_d = vertices_d[objName_s]  # Dictionary that contains vertex data atm. Local variable.
            attrObj_d = self.vertices_d[objName_s]  # Class attribute to store vertex data.

            attrObj_d["vao"] = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(attrObj_d["vao"])

            vertices = np.array(localObj_d["v_i"], dtype=np.float32)
            size = vertices.size * vertices.itemsize
            attrObj_d["ver_num"] = vertices.size // 3

            attrObj_d["v"] = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, attrObj_d["v"])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, vertices, gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(0)

            #### Texture Coord ####

            textureCoords = np.array(localObj_d["vt_i"], dtype=np.float32)
            size = textureCoords.size * textureCoords.itemsize

            attrObj_d["vt"] = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, attrObj_d["vt"])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, textureCoords, gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(1)

            #### Normal ####

            normalsArray = np.array(localObj_d["vn_i"], dtype=np.float32)
            size = normalsArray.size * normalsArray.itemsize

            attrObj_d["vn"] = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, attrObj_d["vn"])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, normalsArray, gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(2)

            attrObj_d["txid"] = mainpy.TextureContainer.getTexture(self.materials_d[localObj_d["mtl"]]["map_Kd"])

    def _loadVertices(self):
        vertices_d = {}

        st = time()
        with open(self.getObjDir()) as file:
            for x_s in file:
                x_s = x_s.rstrip('\n')

                if x_s.startswith("o "):
                    curObj_s = x_s.split()[1]
                    if curObj_s in vertices_d:
                        raise FileExistsError
                    vertices_d[curObj_s] = {}

                elif x_s.startswith("v "):
                    if "v" not in vertices_d:
                        vertices_d["v"] = []
                    _, x, y, z = x_s.split()
                    vertices_d["v"].append( ( float(x), float(y), float(z) ) )
                    del _, x, y, z
                elif x_s.startswith("vt "):
                    if "vt" not in vertices_d:
                        vertices_d["vt"] = []
                    _, x, y = x_s.split()
                    vertices_d["vt"].append( ( float(x), float(y) ) )
                    del _, x, y
                elif x_s.startswith("vn "):
                    if "vn" not in vertices_d:
                        vertices_d["vn"] = []
                    _, x, y, z = x_s.split()
                    vertices_d["vn"].append( ( float(x), float(y), float(z) ) )
                    del _, x, y, z

                elif x_s.startswith("usemtl "):
                    if curObj_s not in vertices_d:
                        raise FileNotFoundError
                    vertices_d[curObj_s]["mtl"] = x_s.split()[1]
                elif x_s.startswith("f "):
                    if curObj_s not in vertices_d:
                        raise FileNotFoundError
                    if "v_i" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["v_i"] = []
                    if "vt_i" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["vt_i"] = []
                    if "vn_i" not in vertices_d[curObj_s]:
                        vertices_d[curObj_s]["vn_i"] = []

                    _, v0_s, v1_s, v2_s = x_s.split()
                    verticesIndices_t = (
                        tuple(map(lambda xx: int(xx), v0_s.split('/'))),
                        tuple(map(lambda xx: int(xx), v1_s.split('/'))),
                        tuple(map(lambda xx: int(xx), v2_s.split('/')))
                    )

                    for v_l in verticesIndices_t:
                        try:
                            vertices_d[curObj_s]["v_i" ].append( vertices_d["v"][v_l[0] - 1] )
                        except IndexError:
                            print( "{}: {}, {}, v  <<  {}".format( curObj_s, v_l[0] - 1, len(vertices_d["v"]), repr(x_s) ) )
                        try:
                            vertices_d[curObj_s]["vt_i"].append(vertices_d["vt"][v_l[1] - 1])
                        except IndexError:
                            print( "{}: {}, {}, vt  <<  {}".format( curObj_s, v_l[1] - 1, len(vertices_d["vt" ]), repr(x_s) ) )
                        try:
                            vertices_d[curObj_s]["vn_i"].append(vertices_d["vn"][v_l[2] - 1])
                        except IndexError:
                            print( "{}: {}, {}, vn  <<  {}".format( curObj_s, v_l[2] - 1, len(vertices_d["vn" ]), repr(x_s) ) )
        print( "File job:", time() - st )
        del vertices_d["v"], vertices_d["vt"], vertices_d["vn"]

        for x in vertices_d:
            aa = vertices_d[x]
            assert len(aa["vt_i"]) == len(aa["vn_i"]) == len(aa["v_i"])
            del x, aa

        self.vertices_d = {}
        for objName_s in vertices_d.keys():
            st = time()
            self.vertices_d[objName_s] = {}
            localObj_d = vertices_d[objName_s]  # Dictionary that contains vertex data atm. Local variable.
            attrObj_d = self.vertices_d[objName_s]  # Class attribute to store vertex data.

            attrObj_d["vao"] = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(attrObj_d["vao"])

            vertices = np.array(localObj_d["v_i"], dtype=np.float32)
            size = vertices.size * vertices.itemsize
            attrObj_d["ver_num"] = vertices.size // 3

            attrObj_d["v"] = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, attrObj_d["v"])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, vertices, gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(0)

            #### Texture Coord ####

            textureCoords = np.array(localObj_d["vt_i"], dtype=np.float32)
            size = textureCoords.size * textureCoords.itemsize

            attrObj_d["vt"] = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, attrObj_d["vt"])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, textureCoords, gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(1)

            #### Normal ####

            normalsArray = np.array(localObj_d["vn_i"], dtype=np.float32)
            size = normalsArray.size * normalsArray.itemsize

            attrObj_d["vn"] = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, attrObj_d["vn"])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, size, normalsArray, gl.GL_STATIC_DRAW)

            gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            gl.glEnableVertexAttribArray(2)

            attrObj_d["txid"] = mainpy.TextureContainer.getTexture(self.materials_d[localObj_d["mtl"]]["map_Kd"])
            print(objName_s, "vao loaded:", time() - st)

    def _loadMaterial(self):
        with open(self.getMtlDir()) as file:
            curMat_s = ""

            for x, x_s in enumerate(file):
                x_s = x_s.strip('\n')

                if x_s.startswith("newmtl "):
                    curMat_s = x_s.split()[1]
                    if curMat_s in self.materials_d.keys():
                        raise FileExistsError
                    self.materials_d[curMat_s] = {}

                elif x_s.startswith("map_Kd "):
                    if curMat_s not in self.materials_d.keys():
                        raise FileNotFoundError
                    self.materials_d[curMat_s]["map_Kd"] = x_s.split()[1]
