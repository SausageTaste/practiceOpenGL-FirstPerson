// Thanks to "https://learnopengl.com/"

#version 430 core


// From buffer
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoordIn;

// From application via glUniform...()
layout (location = 2) uniform vec3 normal;
layout (location = 3) uniform float textureHorNum_f;
layout (location = 4) uniform float textureVerNum_f;

layout (location = 5) uniform mat4 projectMatrix;
layout (location = 6) uniform mat4 viewMatrix;
layout (location = 7) uniform mat4 modelMatrix;

// Output for fragment shader
out vec2 texCoord;
out vec3 normalVec;
out vec3 fragPos;


void main(void)
{
    gl_PointSize = 50.0;
    gl_Position = projectMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
    //gl_Position = vec4(position, 1.0);
    texCoord = vec2(texCoordIn.x * textureHorNum_f, texCoordIn.y * textureVerNum_f);
    normalVec = normalize(vec3(modelMatrix * vec4(normal, 0.0)));
    fragPos = vec3(modelMatrix * vec4(position, 1.0));
}
