import OpenGL.GL as gl
from OpenGL.GL import shaders


def get_shader_log(shader) -> str:
    # It is sometimes str and sometimes bytes.
    log = gl.glGetShaderInfoLog(shader)

    try:
        return log.decode()
    except:
        pass

    try:
        return str(log)
    except:
        pass

    return "could not get shader log str"


class ShadowMap:
    def __init__(self):
        self.__depthMapFbo = gl.glGenFramebuffers(1)

        self.__shadowW_i = 1024 * 4
        self.__shadowH_i = 1024 * 4

        self.__program = self._getProgram()

        self.__depthMapTex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.__depthMapTex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, self.__shadowW_i, self.__shadowH_i, 0,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
        # gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (1.0, 1.0, 1.0, 1.0))

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__depthMapFbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.__depthMapTex, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print( "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" )
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def startRenderOn(self, lightProjection, lightView):
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glUseProgram(self.__program)
        gl.glViewport(0, 0, self.__shadowW_i, self.__shadowH_i)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.__depthMapFbo)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        gl.glUniformMatrix4fv(1, 1, gl.GL_FALSE, lightView * lightProjection)

    def finishRenderOn(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def getTex(self):
        return self.__depthMapTex

    @staticmethod
    def _getProgram() -> int:
        with open("shader_source\\vs_shadow.glsl") as file:
            vertexShader = shaders.compileShader(file.read(), gl.GL_VERTEX_SHADER)
        log = get_shader_log(vertexShader)
        if log:
            raise TypeError(log)

        with open("shader_source\\fs_shadow.glsl") as file:
            fragmentShader = shaders.compileShader(file.read(), gl.GL_FRAGMENT_SHADER)
        log = get_shader_log(fragmentShader)
        if log:
            raise TypeError(log)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertexShader)
        gl.glAttachShader(program, fragmentShader)
        gl.glLinkProgram(program)

        print("Linking Log in Shadow:", gl.glGetProgramiv(program, gl.GL_LINK_STATUS))

        gl.glDeleteShader(vertexShader)
        gl.glDeleteShader(fragmentShader)

        gl.glUseProgram(program)

        return program
