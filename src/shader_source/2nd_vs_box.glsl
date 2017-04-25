// Thanks to "https://learnopengl.com/"

#version 430 core


// From buffer
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 texCoordIn;

// From object, application
layout (location = 3) uniform float textureHorNum_f;
layout (location = 4) uniform float textureVerNum_f;
layout (location = 7) uniform mat4 modelMatrix;

// From manager, application
layout (location = 5) uniform mat4 projectMatrix;
layout (location = 6) uniform mat4 viewMatrix;
uniform mat4 lightSpaceMatrix;

// Output for fragment shader
out vec2 texCoord;
out vec3 normalVec;
out vec3 fragPos;
out vec4 fragPosLightSpace;


vec4 getSurfaceNormal(int vertexIndex_i)
{
    int surfaceIndex_i = vertexIndex_i / 6;

    if (surfaceIndex_i == 0)
        return vec4(0.0, 1.0, 0.0, 0.0);
    else if (surfaceIndex_i == 1)
        return vec4(0.0, 0.0, 1.0, 0.0);
    else if (surfaceIndex_i == 2)
        return vec4(1.0, 0.0, 0.0, 0.0);
    else if (surfaceIndex_i == 3)
        return vec4(0.0, 0.0, -1.0, 0.0);
    else if (surfaceIndex_i == 4)
        return vec4(-1.0, 0.0, 0.0, 0.0);
    else if (surfaceIndex_i == 5)
        return vec4(0.0, -1.0, 0.0, 0.0);
    else
        return vec4(0.0, 1.0, 0.0, 0.0);
}


void main(void)
{
    gl_Position = projectMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
    texCoord = vec2(texCoordIn.x * textureHorNum_f, texCoordIn.y * textureVerNum_f);
    normalVec = normalize(vec3(modelMatrix * getSurfaceNormal(gl_VertexID)));
    fragPos = vec3(modelMatrix * vec4(position, 1.0));
    fragPosLightSpace = lightSpaceMatrix * vec4(fragPos, 1.0);
}
