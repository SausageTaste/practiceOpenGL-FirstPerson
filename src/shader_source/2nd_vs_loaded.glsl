// Thanks to "https://learnopengl.com/"

#version 430 core


// From buffer
layout (location = 0) in vec3 position;
layout (location = 1) in vec2 texCoordIn;
layout (location = 2) in vec3 normal;

// From object, application
layout (location = 3) uniform mat4 modelMatrix;

// From manager, application
layout (location = 5) uniform mat4 projectMatrix;
layout (location = 6) uniform mat4 viewMatrix;
uniform mat4 lightSpaceMatrix;

// Output for fragment shader
out vec2 texCoord;
out vec3 normalVec;
out vec3 fragPos;
out vec4 fragPosLightSpace;


void main(void)
{
    gl_Position = projectMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
    texCoord = texCoordIn;
    normalVec = normalize( vec3( modelMatrix * vec4(normal, 0.0) ) );
    fragPos = vec3(modelMatrix * vec4(position, 1.0));
    fragPosLightSpace = lightSpaceMatrix * vec4( fragPos, 1.0 );
}
