import OpenGL.GL as gl


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
