#version 430 core

// need to study Voraldo's simultaneous use of image/texture bindings of the render texture
// fragment shader, samples from current color buffer
// layout( binding = 0 ) uniform sampler2DRect image_data;

// render texture, which is read from by this shader
layout( binding = 0, rgba8ui ) uniform uimage2D image_data;

uniform vec2 resolution;
out vec4 fragment_output;

// this requires refitting in a couple places to use samplers instead of the imageLoad -
// this is similar to what's done in Voraldo for multiple types of access to the same data

// this implementation shared by Inigo Quilez https://www.shadertoy.com/view/MllBWf 
vec4 myTexture( sampler2D tex, vec2 uv)
{
    vec2 res = vec2(textureSize(tex,0));
    
    uv = uv*res;
    vec2 seam = floor(uv+0.5);
    uv = seam + clamp( (uv-seam)/fwidth(uv), -0.5, 0.5);
    return texture(tex, uv/res);
}

void main()
{
	// fragment_output = texture(image_data, gl_FragCoord.xy);
	// fragment_output = imageLoad(image_data, ivec2(gl_FragCoord.xy));

	ivec2 position = ivec2((gl_FragCoord.xy / resolution.xy) * imageSize(image_data));
	fragment_output = vec4(imageLoad(image_data, position)) / 255.;

}
