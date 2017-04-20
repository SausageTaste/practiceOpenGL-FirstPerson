#version 430 core


// From binded texture
uniform sampler2D s;

// From application
layout (location = 8) uniform vec3 viewPos;
layout (location = 9) uniform vec3 ambient;
layout (location = 11) uniform float shininess;
layout (location = 53) uniform float specularStrength;

layout (location = 10) uniform int lightCount;
layout (location = 12) uniform vec3 lightPos[5];
layout (location = 17) uniform vec3 lightColor[5];
layout (location = 22) uniform float lightMaxDistance[5];

layout (location = 27) uniform int spotLightCount;
layout (location = 28) uniform vec3 spotLightPoses[5];
layout (location = 33) uniform vec3 spotLightColors[5];
layout (location = 38) uniform vec3 spotLightDirections[5];
layout (location = 43) uniform float spotLightMaxDistances[5];
layout (location = 48) uniform float spotLightCutoff[5];

// From vertex shader
in vec2 texCoord;
in vec3 normalVec;
in vec3 fragPos;

// For whatever who takes color to draw pixel
out vec4 color;


vec3 accumPointLight(vec3 viewDir, vec3 curlightPos, vec3 curlightColor, float curlightMaxDistance)
{
	float distance_f = length(curlightPos - fragPos);
	vec3 lightDir = normalize(curlightPos - fragPos);
	float distanceDecreaser = -1 / (curlightMaxDistance*curlightMaxDistance) * (distance_f*distance_f) + 1;
	
	vec3 diffuse;
	if (distance_f > curlightMaxDistance)
	{
		diffuse = vec3(0.0);
	}
	else
	{
		// Calculate diffuse lighting.
		
		float diff_f = max(dot(normalVec, lightDir), 0.0);
		diffuse = max(diff_f * curlightColor, vec3(0.0));
	}

    // Calculate specular lighting.
	if (shininess == 0.0)
	{
		return diffuse * distanceDecreaser;
	}
	else
	{
		vec3 reflectDir = reflect(-lightDir, normalVec);
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
		vec3 specular = max(specularStrength * spec * curlightColor, vec3(0.0));
		
		return diffuse * distanceDecreaser + specular * (distanceDecreaser + 1);
	}
}


vec3 accumSpotLight(vec3 viewDir, vec3 curlightPos, vec3 curlightColor, vec3 curLightDirection, float curlightMaxDistance, float curLightCutoff)
{
	float distance_f = length(curlightPos - fragPos);
	vec3 lightDir = normalize(curlightPos - fragPos);
	float theta = dot(lightDir, normalize(-curLightDirection));
	
	
	if (theta > curLightCutoff) 
	{       
		return accumPointLight(viewDir, curlightPos, curlightColor, curlightMaxDistance) * 10 *(theta - curLightCutoff);
	}
	else
		return vec3(0.0);
}
	
void main(void)
{
	
	vec3 accumLight = vec3(0.0);
	
	vec3 viewDir = normalize(viewPos - fragPos);
	for (int i = 0; i < lightCount; i++)
	{
		accumLight += max( accumPointLight(viewDir, lightPos[i], lightColor[i], lightMaxDistance[i]), vec3(0.0) );
	}
	
	for (int i = 0; i < spotLightCount; i++)
	{
		accumLight += max( accumSpotLight(viewDir, spotLightPoses[i], spotLightColors[i], spotLightDirections[i], spotLightMaxDistances[i], spotLightCutoff[i]),  vec3(0.0) );
	}
	
	vec4 texColor = texture(s, texCoord);
    vec4 shit = texColor * vec4(accumLight + ambient, 1.0);
	
	float depthValue = texture(s, texCoord).r;
	color = vec4(vec3(depthValue), 1.0);
}