#version 430 core

layout (location = 0) in vec3 position;

layout (location = 1) uniform mat4 lightProject;
//layout (location = 2) uniform mat4 lightView;
layout (location = 3) uniform mat4 model;

void main()
{
    gl_Position = lightProject * model * vec4(position, 1.0);
}