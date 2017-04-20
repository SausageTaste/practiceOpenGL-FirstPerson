#version 430 core


// From binded texture
layout (location = 56) uniform sampler2D s;
layout (location = 57) uniform sampler2D shadowMap;

// From object, application
layout (location = 11) uniform float shininess;
layout (location = 54) uniform float specularStrength;

// From manager, application
layout (location = 8) uniform vec3 viewPos;
layout (location = 9) uniform vec3 ambient;

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

uniform vec3 sunLightColor;
uniform vec3 sunLightDirection;

// From vertex shader
in vec2 texCoord;
in vec3 normalVec;
in vec3 fragPos;
in vec4 fragPosLightSpace;

// For whatever who takes color to draw pixel
out vec4 color;


vec3 accumSunLight(vec3 viewDir)
{
	vec3 lightDirection = normalize( sunLightDirection * -1 );
	vec3 curlightColor = sunLightColor;
	
	// Calculate diffuse lighting.
		
	float diff_f = max(dot(normalVec, lightDirection), 0.0);
	vec3 diffuse = max(diff_f * curlightColor, vec3(0.0));
	
    // Calculate specular lighting.
	if (shininess == 0.0)
	{
		//return diffuse * distanceDecreaser;
		return diffuse;
	}
	else
	{
		vec3 reflectDir = reflect(-lightDirection, normalVec);
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
		vec3 specular = max(specularStrength * spec * curlightColor, vec3(0.0));
		
		//return diffuse * distanceDecreaser + specular * (distanceDecreaser + 1);
		return diffuse + specular;
	}
}


float calculateShadow()
{
	// perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	if (projCoords.z > 1.0)
		return 0.0;
    // Transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // Get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // Get depth of current fragment from light's perspective
	if (projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
		return 0.0;
    float currentDepth = projCoords.z;
    // Check whether current frag pos is in shadow
	//float bias = 0.001;
	vec3 lightDir = normalize(viewPos - fragPos);
	float bias = max(0.05 * (1.0 - dot(normalVec, lightDir)), 0.005);
	bias = 0.002;
	float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;  
   
    return shadow;
}


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
	
	float shadow = calculateShadow();
	accumLight += accumSunLight(viewDir) * (1-shadow);
	
	for (int i = 0; i < lightCount; i++)
	{
		accumLight += max( accumPointLight(viewDir, lightPos[i], lightColor[i], lightMaxDistance[i]), vec3(0.0) );
	}
	
	for (int i = 0; i < spotLightCount; i++)
	{
		accumLight += max( accumSpotLight(viewDir, spotLightPoses[i], spotLightColors[i], spotLightDirections[i], spotLightMaxDistances[i], spotLightCutoff[i]),  vec3(0.0) );
	}
	
    color = texture(s, texCoord) * vec4(accumLight + ambient, 1.0);
}