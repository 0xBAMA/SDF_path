#version 430 core
layout( local_size_x = 8, local_size_y = 8, local_size_z = 1 ) in;

// render texture, this is written to by this shader
layout( binding = 0, rgba8ui ) uniform uimage2D current;

#define M_PI 3.1415926535897932384626433832795

#define MAX_STEPS 400
#define MAX_DIST  100.
#define EPSILON   0.002 // closest surface distance

#define AA 2

uniform vec3 basic_diffuse;
uniform vec3 fog_color;

uniform int tonemap_mode;
uniform float gamma;

uniform vec3 lightPos1;
uniform vec3 lightPos2;
uniform vec3 lightPos3;

// flicker factors
uniform float flickerfactor1;
uniform float flickerfactor2;
uniform float flickerfactor3;

// diffuse light colors
uniform vec3 lightCol1d;
uniform vec3 lightCol2d;
uniform vec3 lightCol3d;
// specular light colors
uniform vec3 lightCol1s;
uniform vec3 lightCol2s;
uniform vec3 lightCol3s;
// specular powers per light
uniform float specpower1;
uniform float specpower2;
uniform float specpower3;
// sharpness terms per light
uniform float shadow1;
uniform float shadow2;
uniform float shadow3;

uniform float AO_scale;

uniform vec3 basis_x;
uniform vec3 basis_y;
uniform vec3 basis_z;

uniform float fov;

uniform vec3 ray_origin;
uniform float time;

uniform float depth_scale;
uniform int depth_falloff;




// tonemapping stuff
// APPROX
// --------------------------
vec3 cheapo_aces_approx(vec3 v)
{
	v *= 0.6f;
	float a = 2.51f;
	float b = 0.03f;
	float c = 2.43f;
	float d = 0.59f;
	float e = 0.14f;
	return clamp((v*(a*v+b))/(v*(c*v+d)+e), 0.0f, 1.0f);
}


// OFFICIAL
// --------------------------
mat3 aces_input_matrix = mat3(
	0.59719f, 0.35458f, 0.04823f,
	0.07600f, 0.90834f, 0.01566f,
	0.02840f, 0.13383f, 0.83777f
);

mat3 aces_output_matrix = mat3(
	1.60475f, -0.53108f, -0.07367f,
	-0.10208f,  1.10813f, -0.00605f,
	-0.00327f, -0.07276f,  1.07602f
);

vec3 mul(mat3 m, vec3 v)
{
	float x = m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2];
	float y = m[1][0] * v[1] + m[1][1] * v[1] + m[1][2] * v[2];
	float z = m[2][0] * v[1] + m[2][1] * v[1] + m[2][2] * v[2];
	return vec3(x, y, z);
}

vec3 rtt_and_odt_fit(vec3 v)
{
	vec3 a = v * (v + 0.0245786f) - 0.000090537f;
	vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
	return a / b;
}

vec3 aces_fitted(vec3 v)
{
	v = mul(aces_input_matrix, v);
	v = rtt_and_odt_fit(v);
	return mul(aces_output_matrix, v);
}


vec3 uncharted2(vec3 v)
{
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    float W = 11.2;

    float ExposureBias = 2.0f;
    v *= ExposureBias;

    return (((v*(A*v+C*B)+D*E)/(v*(A*v+B)+D*F))-E/F)*(((W*(A*W+C*B)+D*E)/(W*(A*W+B)+D*F))-E/F);
}

vec3 rienhard(vec3 v)
{
    return v / (vec3(1.) + v);
}

vec3 rienhard2(vec3 v)
{
    const float L_white = 4.0;
    return (v * (vec3(1.) + v / (L_white * L_white))) / (1.0 + v);
}

vec3 tonemap_uchimura(vec3 v)
{
    const float P = 1.0;  // max display brightness
    const float a = 1.0;  // contrast
    const float m = 0.22; // linear section start
    const float l = 0.4;  // linear section length
    const float c = 1.33; // black
    const float b = 0.0;  // pedestal

    // Uchimura 2017, "HDR theory and practice"
    // Math: https://www.desmos.com/calculator/gslcdxvipg
    // Source: https://www.slideshare.net/nikuque/hdr-theory-and-practicce-jp
    float l0 = ((P - m) * l) / a;
    float L0 = m - m / a;
    float L1 = m + (1.0 - m) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    vec3 w0 = 1.0 - smoothstep(0.0, m, v);
    vec3 w2 = step(m + l0, v);
    vec3 w1 = 1.0 - w0 - w2;

    vec3 T = m * pow(v / m, vec3(c)) + vec3(b);
    vec3 S = P - (P - S1) * exp(CP * (v - S0));
    vec3 L = m + a * (v - vec3(m));

    return T * w0 + L * w1 + S * w2;
}

vec3 tonemap_uchimura2(vec3 v)
{
    const float P = 1.0;  // max display brightness
    const float a = 1.7;  // contrast
    const float m = 0.1; // linear section start
    const float l = 0.0;  // linear section length
    const float c = 1.33; // black
    const float b = 0.0;  // pedestal

    float l0 = ((P - m) * l) / a;
    float L0 = m - m / a;
    float L1 = m + (1.0 - m) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    vec3 w0 = 1.0 - smoothstep(0.0, m, v);
    vec3 w2 = step(m + l0, v);
    vec3 w1 = 1.0 - w0 - w2;

    vec3 T = m * pow(v / m, vec3(c)) + vec3(b);
    vec3 S = P - (P - S1) * exp(CP * (v - S0));
    vec3 L = m + a * (v - vec3(m));

    return T * w0 + L * w1 + S * w2;
}

vec3 tonemap_unreal3(vec3 v)
{
    return v / (v + 0.155) * 1.019;
}


#define toLum(color) dot(color, vec3(.2125, .7154, .0721) )
#define lightAjust(a,b) ((1.-b)*(pow(1.-a,vec3(b+1.))-1.)+a)/b
#define reinhard(c,l) c * (l / (1. + l) / l)
vec3 jt_toneMap(vec3 x){
    float l = toLum(x);
    x = reinhard(x,l);
    float m = max(x.r,max(x.g,x.b));
    return min(lightAjust(x/m,m),x);
}
#undef toLum
#undef lightAjust
#undef reinhard


vec3 robobo1221sTonemap(vec3 x){
	return sqrt(x / (x + 1.0f / x)) - abs(x) + x;
}

vec3 roboTonemap(vec3 c){
    return c/sqrt(1.+c*c);
}

vec3 jodieRoboTonemap(vec3 c){
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 tc=c/sqrt(c*c+1.);
    return mix(c/sqrt(l*l+1.),tc,tc);
}

vec3 jodieRobo2ElectricBoogaloo(const vec3 color){
    float luma = dot(color, vec3(.2126, .7152, .0722));

    // tonemap curve goes on this line
    // (I used robo here)
    vec4 rgbl = vec4(color, luma) * inversesqrt(luma*luma + 1.);

    vec3 mappedColor = rgbl.rgb;
    float mappedLuma = rgbl.a;

    float channelMax = max(max(max(
    	mappedColor.r,
    	mappedColor.g),
    	mappedColor.b),
    	1.);

    // this is just the simplified/optimised math
    // of the more human readable version below
    return (
        (mappedLuma*mappedColor-mappedColor)-
        (channelMax*mappedLuma-mappedLuma)
    )/(mappedLuma-channelMax);

    const vec3 white = vec3(1);

    // prevent clipping
    vec3 clampedColor = mappedColor/channelMax;

    // x is how much white needs to be mixed with
    // clampedColor so that its luma equals the
    // mapped luma
    //
    // mix(mappedLuma/channelMax,1.,x) = mappedLuma;
    //
    // mix is defined as
    // x*(1-a)+y*a
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/mix.xhtml
    //
    // (mappedLuma/channelMax)*(1.-x)+1.*x = mappedLuma

    float x = (mappedLuma - mappedLuma*channelMax)
        /(mappedLuma - channelMax);
    return mix(clampedColor, white, x);
}

vec3 jodieReinhardTonemap(vec3 c){
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    vec3 tc=c/(c+1.);
    return mix(c/(l+1.),tc,tc);
}

vec3 jodieReinhard2ElectricBoogaloo(const vec3 color){
    float luma = dot(color, vec3(.2126, .7152, .0722));

    // tonemap curve goes on this line
    // (I used reinhard here)
    vec4 rgbl = vec4(color, luma) / (luma + 1.);

    vec3 mappedColor = rgbl.rgb;
    float mappedLuma = rgbl.a;

    float channelMax = max(max(max(
    	mappedColor.r,
    	mappedColor.g),
    	mappedColor.b),
    	1.);

    // this is just the simplified/optimised math
    // of the more human readable version below
    return ((mappedLuma*mappedColor-mappedColor)-(channelMax*mappedLuma-mappedLuma))/(mappedLuma-channelMax);

    const vec3 white = vec3(1);

    // prevent clipping
    vec3 clampedColor = mappedColor/channelMax;

    // x is how much white needs to be mixed with
    // clampedColor so that its luma equals the
    // mapped luma
    //
    // mix(mappedLuma/channelMax,1.,x) = mappedLuma;
    //
    // mix is defined as
    // x*(1-a)+y*a
    // https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/mix.xhtml
    //
    // (mappedLuma/channelMax)*(1.-x)+1.*x = mappedLuma

    float x = (mappedLuma - mappedLuma*channelMax)
        /(mappedLuma - channelMax);
    return mix(clampedColor, white, x);
}





//  ╔╦╗┬ ┬┬╔═╗╦    ╦ ╦┌┬┐┬┬  ┬┌┬┐┬┌─┐┌─┐
//   ║ ││││║ ╦║    ║ ║ │ ││  │ │ │├┤ └─┐
//   ╩ └┴┘┴╚═╝╩═╝  ╚═╝ ┴ ┴┴─┘┴ ┴ ┴└─┘└─┘
//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : stegu
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//               https://github.com/stegu/webgl-noise
//

// (sqrt(5) - 1)/4 = F4, used once below
#define F4 0.309016994374947451
float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec2  mod289(vec2 x) {return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec3  mod289(vec3 x) {return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4  mod289(vec4 x) {return x - floor(x * (1.0 / 289.0)) * 289.0;}
float permute(float x){return mod289(((x*34.0)+1.0)*x);}
vec3  permute(vec3 x) {return mod289(((x*34.0)+1.0)*x);}
vec4  permute(vec4 x) {return mod289(((x*34.0)+1.0)*x);}
float taylorInvSqrt(float r){return 1.79284291400159 - 0.85373472095314 * r;}
vec4  taylorInvSqrt(vec4 r) {return 1.79284291400159 - 0.85373472095314 * r;}
float snoise2D(vec2 v){
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
  // First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

  // Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

  // Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m * m;
  m = m * m;

  // Gradients: 41 points uniformly over a line, mapped onto a diamond.
  // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

  // Normalise gradients implicitly by scaling m
  // Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

  // Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float snoise3D(vec3 v){
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

  // Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

  // Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

  //Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
}
vec4 grad4(float j, vec4 ip){
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;

  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;

  return p;
}
float snoise4D(vec4 v){
  const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
                        0.276393202250021,  // 2 * G4
                        0.414589803375032,  // 3 * G4
                       -0.447213595499958); // -1 + 4 * G4

  // First corner
  vec4 i  = floor(v + dot(v, vec4(F4)) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

  // Other corners

  // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;
  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
  //  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
  //  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C.xxxx
  //  x1 = x0 - i1  + 1.0 * C.xxxx
  //  x2 = x0 - i2  + 2.0 * C.xxxx
  //  x3 = x0 - i3  + 3.0 * C.xxxx
  //  x4 = x0 - 1.0 + 4.0 * C.xxxx
  vec4 x1 = x0 - i1 + C.xxxx;
  vec4 x2 = x0 - i2 + C.yyyy;
  vec4 x3 = x0 - i3 + C.zzzz;
  vec4 x4 = x0 + C.wwww;

  // Permutations
  i = mod289(i);
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

  // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
  // 7*7*6 = 294, which is close to the ring size 17*17 = 289.
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

  // Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

  // Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
                + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;
}
float fsnoise      (vec2 c){return fract(sin(dot(c, vec2(12.9898, 78.233))) * 43758.5453);}
float fsnoiseDigits(vec2 c){return fract(sin(dot(c, vec2(0.129898, 0.78233))) * 437.585453);}
vec3 hsv(float h, float s, float v){
    vec4 t = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + t.xyz) * 6.0 - vec3(t.w));
    return v * mix(vec3(t.x), clamp(p - vec3(t.x), 0.0, 1.0), s);
}
mat2 rotate2D(float r){
    return mat2(cos(r), sin(r), -sin(r), cos(r));
}
mat3 rotate3D(float angle, vec3 axis){
    vec3 a = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float r = 1.0 - c;
    return mat3(
        a.x * a.x * r + c,
        a.y * a.x * r + a.z * s,
        a.z * a.x * r - a.y * s,
        a.x * a.y * r - a.z * s,
        a.y * a.y * r + c,
        a.z * a.y * r + a.x * s,
        a.x * a.z * r + a.y * s,
        a.y * a.z * r - a.x * s,
        a.z * a.z * r + c
    );
}






////////////////////////////////////////////////////////////////
//
//                           HG_SDF
//
//     GLSL LIBRARY FOR BUILDING SIGNED DISTANCE BOUNDS
//
//     version 2021-01-29
//
//     Check https://mercury.sexy/hg_sdf for updates
//     and usage examples. Send feedback to spheretracing@mercury.sexy.
//
//     Brought to you by MERCURY https://mercury.sexy
//
//
//
// Released as Creative Commons Attribution-NonCommercial (CC BY-NC)
//
////////////////////////////////////////////////////////////////
//
// How to use this:
//
// 1. Build some system to #include glsl files in each other.
//   Include this one at the very start. Or just paste everywhere.
// 2. Build a sphere tracer. See those papers:
//   * "Sphere Tracing" https://link.springer.com/article/10.1007%2Fs003710050084
//   * "Enhanced Sphere Tracing" http://diglib.eg.org/handle/10.2312/stag.20141233.001-008
//   * "Improved Ray Casting of Procedural Distance Bounds" https://www.bibsonomy.org/bibtex/258e85442234c3ace18ba4d89de94e57d
//   The Raymnarching Toolbox Thread on pouet can be helpful as well
//   http://www.pouet.net/topic.php?which=7931&page=1
//   and contains links to many more resources.
// 3. Use the tools in this library to build your distance bound f().
// 4. ???
// 5. Win a compo.
//
// (6. Buy us a beer or a good vodka or something, if you like.)
//
////////////////////////////////////////////////////////////////
//
// Table of Contents:
//
// * Helper functions and macros
// * Collection of some primitive objects
// * Domain Manipulation operators
// * Object combination operators
//
////////////////////////////////////////////////////////////////
//
// Why use this?
//
// The point of this lib is that everything is structured according
// to patterns that we ended up using when building geometry.
// It makes it more easy to write code that is reusable and that somebody
// else can actually understand. Especially code on Shadertoy (which seems
// to be what everybody else is looking at for "inspiration") tends to be
// really ugly. So we were forced to do something about the situation and
// release this lib ;)
//
// Everything in here can probably be done in some better way.
// Please experiment. We'd love some feedback, especially if you
// use it in a scene production.
//
// The main patterns for building geometry this way are:
// * Stay Lipschitz continuous. That means: don't have any distance
//   gradient larger than 1. Try to be as close to 1 as possible -
//   Distances are euclidean distances, don't fudge around.
//   Underestimating distances will happen. That's why calling
//   it a "distance bound" is more correct. Don't ever multiply
//   distances by some value to "fix" a Lipschitz continuity
//   violation. The invariant is: each fSomething() function returns
//   a correct distance bound.
// * Use very few primitives and combine them as building blocks
//   using combine opertors that preserve the invariant.
// * Multiply objects by repeating the domain (space).
//   If you are using a loop inside your distance function, you are
//   probably doing it wrong (or you are building boring fractals).
// * At right-angle intersections between objects, build a new local
//   coordinate system from the two distances to combine them in
//   interesting ways.
// * As usual, there are always times when it is best to not follow
//   specific patterns.
//
////////////////////////////////////////////////////////////////
//
// FAQ
//
// Q: Why is there no sphere tracing code in this lib?
// A: Because our system is way too complex and always changing.
//    This is the constant part. Also we'd like everyone to
//    explore for themselves.
//
// Q: This does not work when I paste it into Shadertoy!!!!
// A: Yes. It is GLSL, not GLSL ES. We like real OpenGL
//    because it has way more features and is more likely
//    to work compared to browser-based WebGL. We recommend
//    you consider using OpenGL for your productions. Most
//    of this can be ported easily though.
//
// Q: How do I material?
// A: We recommend something like this:
//    Write a material ID, the distance and the local coordinate
//    p into some global variables whenever an object's distance is
//    smaller than the stored distance. Then, at the end, evaluate
//    the material to get color, roughness, etc., and do the shading.
//
// Q: I found an error. Or I made some function that would fit in
//    in this lib. Or I have some suggestion.
// A: Awesome! Drop us a mail at spheretracing@mercury.sexy.
//
// Q: Why is this not on github?
// A: Because we were too lazy. If we get bugged about it enough,
//    we'll do it.
//
// Q: Your license sucks for me.
// A: Oh. What should we change it to?
//
// Q: I have trouble understanding what is going on with my distances.
// A: Some visualization of the distance field helps. Try drawing a
//    plane that you can sweep through your scene with some color
//    representation of the distance field at each point and/or iso
//    lines at regular intervals. Visualizing the length of the
//    gradient (or better: how much it deviates from being equal to 1)
//    is immensely helpful for understanding which parts of the
//    distance field are broken.
//
////////////////////////////////////////////////////////////////






////////////////////////////////////////////////////////////////
//
//             HELPER FUNCTIONS/MACROS
//
////////////////////////////////////////////////////////////////

#define PI 3.14159265
#define TAU (2*PI)
#define PHI (sqrt(5)*0.5 + 0.5)

// Clamp to [0,1] - this operation is free under certain circumstances.
// For further information see
// http://www.humus.name/Articles/Persson_LowLevelThinking.pdf and
// http://www.humus.name/Articles/Persson_LowlevelShaderOptimization.pdf
#define saturate(x) clamp(x, 0, 1)

// Sign function that doesn't return 0
float sgn(float x) {
	return (x<0)?-1:1;
}

vec2 sgn(vec2 v) {
	return vec2((v.x<0)?-1:1, (v.y<0)?-1:1);
}

float square (float x) {
	return x*x;
}

vec2 square (vec2 x) {
	return x*x;
}

vec3 square (vec3 x) {
	return x*x;
}

float lengthSqr(vec3 x) {
	return dot(x, x);
}


// Maximum/minumum elements of a vector
float vmax(vec2 v) {
	return max(v.x, v.y);
}

float vmax(vec3 v) {
	return max(max(v.x, v.y), v.z);
}

float vmax(vec4 v) {
	return max(max(v.x, v.y), max(v.z, v.w));
}

float vmin(vec2 v) {
	return min(v.x, v.y);
}

float vmin(vec3 v) {
	return min(min(v.x, v.y), v.z);
}

float vmin(vec4 v) {
	return min(min(v.x, v.y), min(v.z, v.w));
}




////////////////////////////////////////////////////////////////
//
//             PRIMITIVE DISTANCE FUNCTIONS
//
////////////////////////////////////////////////////////////////
//
// Conventions:
//
// Everything that is a distance function is called fSomething.
// The first argument is always a point in 2 or 3-space called <p>.
// Unless otherwise noted, (if the object has an intrinsic "up"
// side or direction) the y axis is "up" and the object is
// centered at the origin.
//
////////////////////////////////////////////////////////////////

float fSphere(vec3 p, float r) {
	return length(p) - r;
}

// Plane with normal n (n is normalized) at some distance from the origin
float fPlane(vec3 p, vec3 n, float distanceFromOrigin) {
	return dot(p, n) + distanceFromOrigin;
}

// Cheap Box: distance to corners is overestimated
float fBoxCheap(vec3 p, vec3 b) { //cheap box
	return vmax(abs(p) - b);
}

// Box: correct distance to corners
float fBox(vec3 p, vec3 b) {
	vec3 d = abs(p) - b;
	return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}

// Same as above, but in two dimensions (an endless box)
float fBox2Cheap(vec2 p, vec2 b) {
	return vmax(abs(p)-b);
}

float fBox2(vec2 p, vec2 b) {
	vec2 d = abs(p) - b;
	return length(max(d, vec2(0))) + vmax(min(d, vec2(0)));
}


// Endless "corner"
float fCorner (vec2 p) {
	return length(max(p, vec2(0))) + vmax(min(p, vec2(0)));
}

// Blobby ball object. You've probably seen it somewhere. This is not a correct distance bound, beware.
float fBlob(vec3 p) {
	p = abs(p);
	if (p.x < max(p.y, p.z)) p = p.yzx;
	if (p.x < max(p.y, p.z)) p = p.yzx;
	float b = max(max(max(
		dot(p, normalize(vec3(1, 1, 1))),
		dot(p.xz, normalize(vec2(PHI+1, 1)))),
		dot(p.yx, normalize(vec2(1, PHI)))),
		dot(p.xz, normalize(vec2(1, PHI))));
	float l = length(p);
	return l - 1.5 - 0.2 * (1.5 / 2)* cos(min(sqrt(1.01 - b / l)*(PI / 0.25), PI));
}

// Cylinder standing upright on the xz plane
float fCylinder(vec3 p, float r, float height) {
	float d = length(p.xz) - r;
	d = max(d, abs(p.y) - height);
	return d;
}

// Capsule: A Cylinder with round caps on both sides
float fCapsule(vec3 p, float r, float c) {
	return mix(length(p.xz) - r, length(vec3(p.x, abs(p.y) - c, p.z)) - r, step(c, abs(p.y)));
}

// Distance to line segment between <a> and <b>, used for fCapsule() version 2below
float fLineSegment(vec3 p, vec3 a, vec3 b) {
	vec3 ab = b - a;
	float t = saturate(dot(p - a, ab) / dot(ab, ab));
	return length((ab*t + a) - p);
}

// Capsule version 2: between two end points <a> and <b> with radius r
float fCapsule(vec3 p, vec3 a, vec3 b, float r) {
	return fLineSegment(p, a, b) - r;
}

// Torus in the XZ-plane
float fTorus(vec3 p, float smallRadius, float largeRadius) {
	return length(vec2(length(p.xz) - largeRadius, p.y)) - smallRadius;
}

// A circle line. Can also be used to make a torus by subtracting the smaller radius of the torus.
float fCircle(vec3 p, float r) {
	float l = length(p.xz) - r;
	return length(vec2(p.y, l));
}

// A circular disc with no thickness (i.e. a cylinder with no height).
// Subtract some value to make a flat disc with rounded edge.
float fDisc(vec3 p, float r) {
	float l = length(p.xz) - r;
	return l < 0 ? abs(p.y) : length(vec2(p.y, l));
}

// Hexagonal prism, circumcircle variant
float fHexagonCircumcircle(vec3 p, vec2 h) {
	vec3 q = abs(p);
	return max(q.y - h.y, max(q.x*sqrt(3)*0.5 + q.z*0.5, q.z) - h.x);
	//this is mathematically equivalent to this line, but less efficient:
	//return max(q.y - h.y, max(dot(vec2(cos(PI/3), sin(PI/3)), q.zx), q.z) - h.x);
}

// Hexagonal prism, incircle variant
float fHexagonIncircle(vec3 p, vec2 h) {
	return fHexagonCircumcircle(p, vec2(h.x*sqrt(3)*0.5, h.y));
}

// Cone with correct distances to tip and base circle. Y is up, 0 is in the middle of the base.
float fCone(vec3 p, float radius, float height) {
	vec2 q = vec2(length(p.xz), p.y);
	vec2 tip = q - vec2(0, height);
	vec2 mantleDir = normalize(vec2(height, radius));
	float mantle = dot(tip, mantleDir);
	float d = max(mantle, -q.y);
	float projected = dot(tip, vec2(mantleDir.y, -mantleDir.x));

	// distance to tip
	if ((q.y > height) && (projected < 0)) {
		d = max(d, length(tip));
	}

	// distance to base ring
	if ((q.x > radius) && (projected > length(vec2(height, radius)))) {
		d = max(d, length(q - vec2(radius, 0)));
	}
	return d;
}

//
// "Generalized Distance Functions" by Akleman and Chen.
// see the Paper at https://www.viz.tamu.edu/faculty/ergun/research/implicitmodeling/papers/sm99.pdf
//
// This set of constants is used to construct a large variety of geometric primitives.
// Indices are shifted by 1 compared to the paper because we start counting at Zero.
// Some of those are slow whenever a driver decides to not unroll the loop,
// which seems to happen for fIcosahedron und fTruncatedIcosahedron on nvidia 350.12 at least.
// Specialized implementations can well be faster in all cases.
//

const vec3 GDFVectors[19] = vec3[](
	normalize(vec3(1, 0, 0)),
	normalize(vec3(0, 1, 0)),
	normalize(vec3(0, 0, 1)),

	normalize(vec3(1, 1, 1 )),
	normalize(vec3(-1, 1, 1)),
	normalize(vec3(1, -1, 1)),
	normalize(vec3(1, 1, -1)),

	normalize(vec3(0, 1, PHI+1)),
	normalize(vec3(0, -1, PHI+1)),
	normalize(vec3(PHI+1, 0, 1)),
	normalize(vec3(-PHI-1, 0, 1)),
	normalize(vec3(1, PHI+1, 0)),
	normalize(vec3(-1, PHI+1, 0)),

	normalize(vec3(0, PHI, 1)),
	normalize(vec3(0, -PHI, 1)),
	normalize(vec3(1, 0, PHI)),
	normalize(vec3(-1, 0, PHI)),
	normalize(vec3(PHI, 1, 0)),
	normalize(vec3(-PHI, 1, 0))
);

// Version with variable exponent.
// This is slow and does not produce correct distances, but allows for bulging of objects.
float fGDF(vec3 p, float r, float e, int begin, int end) {
	float d = 0;
	for (int i = begin; i <= end; ++i)
		d += pow(abs(dot(p, GDFVectors[i])), e);
	return pow(d, 1/e) - r;
}

// Version with without exponent, creates objects with sharp edges and flat faces
float fGDF(vec3 p, float r, int begin, int end) {
	float d = 0;
	for (int i = begin; i <= end; ++i)
		d = max(d, abs(dot(p, GDFVectors[i])));
	return d - r;
}

// Primitives follow:

float fOctahedron(vec3 p, float r, float e) {
	return fGDF(p, r, e, 3, 6);
}

float fDodecahedron(vec3 p, float r, float e) {
	return fGDF(p, r, e, 13, 18);
}

float fIcosahedron(vec3 p, float r, float e) {
	return fGDF(p, r, e, 3, 12);
}

float fTruncatedOctahedron(vec3 p, float r, float e) {
	return fGDF(p, r, e, 0, 6);
}

float fTruncatedIcosahedron(vec3 p, float r, float e) {
	return fGDF(p, r, e, 3, 18);
}

float fOctahedron(vec3 p, float r) {
	return fGDF(p, r, 3, 6);
}

float fDodecahedron(vec3 p, float r) {
	return fGDF(p, r, 13, 18);
}

float fIcosahedron(vec3 p, float r) {
	return fGDF(p, r, 3, 12);
}

float fTruncatedOctahedron(vec3 p, float r) {
	return fGDF(p, r, 0, 6);
}

float fTruncatedIcosahedron(vec3 p, float r) {
	return fGDF(p, r, 3, 18);
}


////////////////////////////////////////////////////////////////
//
//                DOMAIN MANIPULATION OPERATORS
//
////////////////////////////////////////////////////////////////
//
// Conventions:
//
// Everything that modifies the domain is named pSomething.
//
// Many operate only on a subset of the three dimensions. For those,
// you must choose the dimensions that you want manipulated
// by supplying e.g. <p.x> or <p.zx>
//
// <inout p> is always the first argument and modified in place.
//
// Many of the operators partition space into cells. An identifier
// or cell index is returned, if possible. This return value is
// intended to be optionally used e.g. as a random seed to change
// parameters of the distance functions inside the cells.
//
// Unless stated otherwise, for cell index 0, <p> is unchanged and cells
// are centered on the origin so objects don't have to be moved to fit.
//
//
////////////////////////////////////////////////////////////////



// Rotate around a coordinate axis (i.e. in a plane perpendicular to that axis) by angle <a>.
// Read like this: R(p.xz, a) rotates "x towards z".
// This is fast if <a> is a compile-time constant and slower (but still practical) if not.
void pR(inout vec2 p, float a) {
	p = cos(a)*p + sin(a)*vec2(p.y, -p.x);
}

// Shortcut for 45-degrees rotation
void pR45(inout vec2 p) {
	p = (p + vec2(p.y, -p.x))*sqrt(0.5);
}

// Repeat space along one axis. Use like this to repeat along the x axis:
// <float cell = pMod1(p.x,5);> - using the return value is optional.
float pMod1(inout float p, float size) {
	float halfsize = size*0.5;
	float c = floor((p + halfsize)/size);
	p = mod(p + halfsize, size) - halfsize;
	return c;
}

// Same, but mirror every second cell so they match at the boundaries
float pModMirror1(inout float p, float size) {
	float halfsize = size*0.5;
	float c = floor((p + halfsize)/size);
	p = mod(p + halfsize,size) - halfsize;
	p *= mod(c, 2.0)*2 - 1;
	return c;
}

// Repeat the domain only in positive direction. Everything in the negative half-space is unchanged.
float pModSingle1(inout float p, float size) {
	float halfsize = size*0.5;
	float c = floor((p + halfsize)/size);
	if (p >= 0)
		p = mod(p + halfsize, size) - halfsize;
	return c;
}

// Repeat only a few times: from indices <start> to <stop> (similar to above, but more flexible)
float pModInterval1(inout float p, float size, float start, float stop) {
	float halfsize = size*0.5;
	float c = floor((p + halfsize)/size);
	p = mod(p+halfsize, size) - halfsize;
	if (c > stop) { //yes, this might not be the best thing numerically.
		p += size*(c - stop);
		c = stop;
	}
	if (c <start) {
		p += size*(c - start);
		c = start;
	}
	return c;
}


// Repeat around the origin by a fixed angle.
// For easier use, num of repetitions is use to specify the angle.
float pModPolar(inout vec2 p, float repetitions) {
	float angle = 2*PI/repetitions;
	float a = atan(p.y, p.x) + angle/2.;
	float r = length(p);
	float c = floor(a/angle);
	a = mod(a,angle) - angle/2.;
	p = vec2(cos(a), sin(a))*r;
	// For an odd number of repetitions, fix cell index of the cell in -x direction
	// (cell index would be e.g. -5 and 5 in the two halves of the cell):
	if (abs(c) >= (repetitions/2)) c = abs(c);
	return c;
}

// Repeat in two dimensions
vec2 pMod2(inout vec2 p, vec2 size) {
	vec2 c = floor((p + size*0.5)/size);
	p = mod(p + size*0.5,size) - size*0.5;
	return c;
}

// Same, but mirror every second cell so all boundaries match
vec2 pModMirror2(inout vec2 p, vec2 size) {
	vec2 halfsize = size*0.5;
	vec2 c = floor((p + halfsize)/size);
	p = mod(p + halfsize, size) - halfsize;
	p *= mod(c,vec2(2))*2 - vec2(1);
	return c;
}

// Same, but mirror every second cell at the diagonal as well
vec2 pModGrid2(inout vec2 p, vec2 size) {
	vec2 c = floor((p + size*0.5)/size);
	p = mod(p + size*0.5, size) - size*0.5;
	p *= mod(c,vec2(2))*2 - vec2(1);
	p -= size/2;
	if (p.x > p.y) p.xy = p.yx;
	return floor(c/2);
}

// Repeat in three dimensions
vec3 pMod3(inout vec3 p, vec3 size) {
	vec3 c = floor((p + size*0.5)/size);
	p = mod(p + size*0.5, size) - size*0.5;
	return c;
}

// Mirror at an axis-aligned plane which is at a specified distance <dist> from the origin.
float pMirror (inout float p, float dist) {
	float s = sgn(p);
	p = abs(p)-dist;
	return s;
}

// Mirror in both dimensions and at the diagonal, yielding one eighth of the space.
// translate by dist before mirroring.
vec2 pMirrorOctant (inout vec2 p, vec2 dist) {
	vec2 s = sgn(p);
	pMirror(p.x, dist.x);
	pMirror(p.y, dist.y);
	if (p.y > p.x)
		p.xy = p.yx;
	return s;
}

// Reflect space at a plane
float pReflect(inout vec3 p, vec3 planeNormal, float offset) {
	float t = dot(p, planeNormal)+offset;
	if (t < 0) {
		p = p - (2*t)*planeNormal;
	}
	return sgn(t);
}


////////////////////////////////////////////////////////////////
//
//             OBJECT COMBINATION OPERATORS
//
////////////////////////////////////////////////////////////////
//
// We usually need the following boolean operators to combine two objects:
// Union: OR(a,b)
// Intersection: AND(a,b)
// Difference: AND(a,!b)
// (a and b being the distances to the objects).
//
// The trivial implementations are min(a,b) for union, max(a,b) for intersection
// and max(a,-b) for difference. To combine objects in more interesting ways to
// produce rounded edges, chamfers, stairs, etc. instead of plain sharp edges we
// can use combination operators. It is common to use some kind of "smooth minimum"
// instead of min(), but we don't like that because it does not preserve Lipschitz
// continuity in many cases.
//
// Naming convention: since they return a distance, they are called fOpSomething.
// The different flavours usually implement all the boolean operators above
// and are called fOpUnionRound, fOpIntersectionRound, etc.
//
// The basic idea: Assume the object surfaces intersect at a right angle. The two
// distances <a> and <b> constitute a new local two-dimensional coordinate system
// with the actual intersection as the origin. In this coordinate system, we can
// evaluate any 2D distance function we want in order to shape the edge.
//
// The operators below are just those that we found useful or interesting and should
// be seen as examples. There are infinitely more possible operators.
//
// They are designed to actually produce correct distances or distance bounds, unlike
// popular "smooth minimum" operators, on the condition that the gradients of the two
// SDFs are at right angles. When they are off by more than 30 degrees or so, the
// Lipschitz condition will no longer hold (i.e. you might get artifacts). The worst
// case is parallel surfaces that are close to each other.
//
// Most have a float argument <r> to specify the radius of the feature they represent.
// This should be much smaller than the object size.
//
// Some of them have checks like "if ((-a < r) && (-b < r))" that restrict
// their influence (and computation cost) to a certain area. You might
// want to lift that restriction or enforce it. We have left it as comments
// in some cases.
//
// usage example:
//
// float fTwoBoxes(vec3 p) {
//   float box0 = fBox(p, vec3(1));
//   float box1 = fBox(p-vec3(1), vec3(1));
//   return fOpUnionChamfer(box0, box1, 0.2);
// }
//
////////////////////////////////////////////////////////////////


// The "Chamfer" flavour makes a 45-degree chamfered edge (the diagonal of a square of size <r>):
float fOpUnionChamfer(float a, float b, float r) {
	return min(min(a, b), (a - r + b)*sqrt(0.5));
}

// Intersection has to deal with what is normally the inside of the resulting object
// when using union, which we normally don't care about too much. Thus, intersection
// implementations sometimes differ from union implementations.
float fOpIntersectionChamfer(float a, float b, float r) {
	return max(max(a, b), (a + r + b)*sqrt(0.5));
}

// Difference can be built from Intersection or Union:
float fOpDifferenceChamfer (float a, float b, float r) {
	return fOpIntersectionChamfer(a, -b, r);
}

// The "Round" variant uses a quarter-circle to join the two objects smoothly:
float fOpUnionRound(float a, float b, float r) {
	vec2 u = max(vec2(r - a,r - b), vec2(0));
	return max(r, min (a, b)) - length(u);
}

float fOpIntersectionRound(float a, float b, float r) {
	vec2 u = max(vec2(r + a,r + b), vec2(0));
	return min(-r, max (a, b)) + length(u);
}

float fOpDifferenceRound (float a, float b, float r) {
	return fOpIntersectionRound(a, -b, r);
}


// The "Columns" flavour makes n-1 circular columns at a 45 degree angle:
float fOpUnionColumns(float a, float b, float r, float n) {
	if ((a < r) && (b < r)) {
		vec2 p = vec2(a, b);
		float columnradius = r*sqrt(2)/((n-1)*2+sqrt(2));
		pR45(p);
		p.x -= sqrt(2)/2*r;
		p.x += columnradius*sqrt(2);
		if (mod(n,2) == 1) {
			p.y += columnradius;
		}
		// At this point, we have turned 45 degrees and moved at a point on the
		// diagonal that we want to place the columns on.
		// Now, repeat the domain along this direction and place a circle.
		pMod1(p.y, columnradius*2);
		float result = length(p) - columnradius;
		result = min(result, p.x);
		result = min(result, a);
		return min(result, b);
	} else {
		return min(a, b);
	}
}

float fOpDifferenceColumns(float a, float b, float r, float n) {
	a = -a;
	float m = min(a, b);
	//avoid the expensive computation where not needed (produces discontinuity though)
	if ((a < r) && (b < r)) {
		vec2 p = vec2(a, b);
		float columnradius = r*sqrt(2)/n/2.0;
		columnradius = r*sqrt(2)/((n-1)*2+sqrt(2));

		pR45(p);
		p.y += columnradius;
		p.x -= sqrt(2)/2*r;
		p.x += -columnradius*sqrt(2)/2;

		if (mod(n,2) == 1) {
			p.y += columnradius;
		}
		pMod1(p.y,columnradius*2);

		float result = -length(p) + columnradius;
		result = max(result, p.x);
		result = min(result, a);
		return -min(result, b);
	} else {
		return -m;
	}
}

float fOpIntersectionColumns(float a, float b, float r, float n) {
	return fOpDifferenceColumns(a,-b,r, n);
}

// The "Stairs" flavour produces n-1 steps of a staircase:
// much less stupid version by paniq
float fOpUnionStairs(float a, float b, float r, float n) {
	float s = r/n;
	float u = b-r;
	return min(min(a,b), 0.5 * (u + a + abs ((mod (u - a + s, 2 * s)) - s)));
}

// We can just call Union since stairs are symmetric.
float fOpIntersectionStairs(float a, float b, float r, float n) {
	return -fOpUnionStairs(-a, -b, r, n);
}

float fOpDifferenceStairs(float a, float b, float r, float n) {
	return -fOpUnionStairs(-a, b, r, n);
}


// Similar to fOpUnionRound, but more lipschitz-y at acute angles
// (and less so at 90 degrees). Useful when fudging around too much
// by MediaMolecule, from Alex Evans' siggraph slides
float fOpUnionSoft(float a, float b, float r) {
	float e = max(r - abs(a - b), 0);
	return min(a, b) - e*e*0.25/r;
}


// produces a cylindical pipe that runs along the intersection.
// No objects remain, only the pipe. This is not a boolean operator.
float fOpPipe(float a, float b, float r) {
	return length(vec2(a, b)) - r;
}

// first object gets a v-shaped engraving where it intersect the second
float fOpEngrave(float a, float b, float r) {
	return max(a, (a + r - abs(b))*sqrt(0.5));
}

// first object gets a capenter-style groove cut out
float fOpGroove(float a, float b, float ra, float rb) {
	return max(a, min(a + ra, rb - abs(b)));
}

// first object gets a capenter-style tongue attached
float fOpTongue(float a, float b, float ra, float rb) {
	return min(a, max(a - ra, abs(b) - rb));
}

//  ╔═╗┌┐┌┌┬┐  ╦ ╦╔═╗    ╔═╗╔╦╗╔═╗  ╔═╗┌─┐┌┬┐┌─┐
//  ║╣ │││ ││  ╠═╣║ ╦    ╚═╗ ║║╠╣   ║  │ │ ││├┤
//  ╚═╝┘└┘─┴┘  ╩ ╩╚═╝────╚═╝═╩╝╚    ╚═╝└─┘─┴┘└─┘


// point rotation about an arbitrary axis, ax - from gaziya5
vec3 erot(vec3 p, vec3 ax, float ro) {
    return mix(dot(p,ax)*ax,p,cos(ro))+sin(ro)*cross(ax,p);
}

// from https://twitter.com/gaziya5/status/1340475834352631808
#define sabs(p) sqrt (p*p + 1e-2)
#define smin(a, b) (a + b-sabs (ab)) * .5
#define smax(a, b) (a + b + sabs (ab)) * .5

float opSmoothSubtraction( float d1, float d2, float k )
{
    float h = max(k-abs(-d1-d2),0.0);
    return max(-d1, d2) + h*h*0.25/k;
	//float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
	//return mix( d2, -d1, h ) + k*h*(1.0-h);
}

// smooth minimum
float smin_op(float a, float b, float k) {
    float h = max(0.,k-abs(b-a))/k;
    return min(a,b)-h*h*h*k/6.;
}

// from michael0884's marble marcher community edition
void planeFold(inout vec3 z, vec3 n, float d) {
    z.xyz -= 2.0 * min(0.0, dot(z.xyz, n) - d) * n;
}

void sierpinskiFold(inout vec3 z) {
    z.xy -= min(z.x + z.y, 0.0);
    z.xz -= min(z.x + z.z, 0.0);
    z.yz -= min(z.y + z.z, 0.0);
}

void mengerFold(inout vec3 z)
{
    z.xy += min(z.x - z.y, 0.0)*vec2(-1.,1.);
    z.xz += min(z.x - z.z, 0.0)*vec2(-1.,1.);
    z.yz += min(z.y - z.z, 0.0)*vec2(-1.,1.);
}

void boxFold(inout vec3 z, vec3 r) {
    z.xyz = clamp(z.xyz, -r, r) * 2.0 - z.xyz;
}

// from a distance estimated fractal by discord user Nameless#1608
// array repetition
#define pmod(p,a) mod(p - 0.5*a,a) - 0.5*a

// another fold
void sphereFold(inout vec3 z) {
    float minRadius2 = 1.;
    float fixedRadius2 = 5.;
    float r2 = dot(z,z);
    if (r2 < minRadius2) {
        float temp = (fixedRadius2/minRadius2);
        z*= temp;
    } else if (r2 < fixedRadius2) {
        float temp =(fixedRadius2/r2);
        z*=temp;
    }
}

// some geometric primitives
float sdSphere( vec3 p, float s ) {return length(p)-s;}
float sdTorus( vec3 p, vec2  t ) {return length( vec2(length(p.xz)-t.x,p.y) )-t.y;}
float sdCylinder( vec3 p, vec2  h ) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float old_de( vec3 porig ) { // distance estimator for the scene
    vec3 p = porig;

    // p = pmod(p, vec3(3.85,0.65,3.617));
		pModMirror2(p.xz, vec2(3.85, 3.617));
		pModMirror1(p.y, 0.685);

		// sphereFold(p);
		// mengerFold(p);

    float tfactor = abs(pow(abs(cos(time/2.)), 6.)) * 2 - 1;

		// float drings = sdTorus(p, vec2(1.182, 0.08 + 0.05 * cos(time/5.+0.5*porig.x+0.8*porig.z)));
		float drings = sdTorus(p, vec2(1.182, 0.08 ));

		// float dballz = sdSphere(p, 0.8 + 0.25*tfactor*(sin(time*2.1+porig.x*2.18+porig.z*2.7+porig.y*3.14)+1.));
		float dballz = sdSphere(p, 0.8 + 0.25*tfactor);

    float pillarz = smin_op(drings, dballz, 0.9);

		float dplane = fPlane(porig, vec3(0,1,0), 5.);

		// p = pmod(p*0.2, vec3(2.4,1.2,1.6));
		p = porig;

		pR(p.yz, time/3.14);
		// pR(p.xy, time*0.3);

    float dtorus = fTorus( p, 1.2, 6.6);

		float dfinal = smin_op(
											smin_op(
												max(pillarz, sdSphere(porig, 8.5)),
													dtorus, 0.385),
														dplane, 0.685);

		return dfinal;
}








// by gaz
float screw_de(vec3 p){
    float c=.2;
    p.z+=atan(p.y,p.x)/M_PI*c;
    p.z=mod(p.z,c*2.)-c;
    return length(vec2(length(p.xy)-.4,p.z))-.1;
}




float escape = 0.;

vec3 pal( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{
    return a + b*cos( 6.28318*(c*t+d) );
}






float fractal_de(vec3 p0){
   vec4 p = vec4(p0, 1.);
   escape = 0.;
   for(int i = 0; i < 8; i++){
       p.xyz = mod(p.xyz-1.,2.)-1.;
        p*=1.4/dot(p.xyz,p.xyz);
        escape += exp(-0.2*dot(p.xyz,p.xyz));
   }
   return (length(p.xz/p.w)*0.25);
}


// // hard crash on desktop - probably hardware related - looks like shit on laptop
// // not indexed in the DEC
// float torus(vec3 pos, vec3 p, vec2 s){    
//     vec2 a = normalize(p.xz-pos.xz);
//     pos.xz += a*s.x;
//     return length(pos-p)-s.y;
// }
// float fractal_de2(vec3 p0){
//     vec4 p = vec4(p0, 1.);
//     for(int i = 0; i < 8; i++){
//         p.xyz = mod(p.xyz-1., 2.)-1.;
//         p*=(1.8/dot(p.xyz,p.xyz));
//         escape += exp(-0.2*dot(p.xyz,p.xyz));
//     }
//     p.xyz /= p.w;
//     return 0.25*torus(p0, p.xyz, vec2(5.,0.7));
// }

float fractal_de3(vec3 p0){
    vec4 p = vec4(p0, 1.);
    for(int i = 0; i < 8; i++){
        p.xyz = mod(p.xyz-1., 2.)-1.;
        p*=(1.2/dot(p.xyz,p.xyz));
        escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    p/=p.w;
    return abs(p.x)*0.25;
}

float fractal_de4(vec3 p0){
    vec4 p = vec4(p0, 1.);
    for(int i = 0; i < 8; i++){
        
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        p = abs(p);
        p.xyz = mod(p.xyz-1., 2.)-1.;

        p*=1.23;
        escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    p/=p.w;
    return abs(p.y)*0.25;
}

float fractal_de5(vec3 pos) 
{
#define SCALE 2.8
#define MINRAD2 .25
    float minRad2 = clamp(MINRAD2, 1.0e-9, 1.0);
#define scale (vec4(SCALE, SCALE, SCALE, abs(SCALE)) / minRad2)
    float absScalem1 = abs(SCALE - 1.0);
    float AbsScaleRaisedTo1mIters = pow(abs(SCALE), float(1-10));
	vec4 p = vec4(pos,1);
	vec4 p0 = p;  // p.w is the distance estimate

	for (int i = 0; i < 9; i++)
	{
		p.xyz = clamp(p.xyz, -1.0, 1.0) * 2.0 - p.xyz;

		float r2 = dot(p.xyz, p.xyz);
		p *= clamp(max(minRad2/r2, minRad2), 0.0, 1.0);

		// scale, translate
		p = p*scale + p0;
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
	}
	return ((length(p.xyz) - absScalem1) / p.w - AbsScaleRaisedTo1mIters);
#undef MINRAD2
#undef SCALE
#undef scale
}

// highly varied domain - take a look around
float fractal_de6( vec3 p )
{
	p = p.xzy;
    vec3 cSize = vec3(1., 1., 1.3);
	float scale = 1.;
	for( int i=0; i < 12;i++ )
	{
		p = 2.0*clamp(p, -cSize, cSize) - p;
		float r2 = dot(p,p);
		float k = max((2.)/(r2), .027);
		p     *= k;
		scale *= k;
        escape += exp(-0.00002*dot(p.xyz,p.xyz));
	}
	float l = length(p.xy);
	float rxy = l - 4.0;
	float n = l * p.z;
	rxy = max(rxy, -(n) / 4.);
	return (rxy) / abs(scale);
}

float fractal_de7( vec3 p )
{
	p = p.xzy;
    vec3 cSize = vec3(1., 1., 1.3);
	float scale = 1.;
	for( int i=0; i < 12;i++ )
	{
		p = 2.0*clamp(p, -cSize, cSize) - p;
        float r2 = dot(p,p+sin(p.z*.3));
		float k = max((2.)/(r2), .027);
		p     *= k;
		scale *= k;
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
	}
	float l = length(p.xy);
	float rxy = l - 4.0;
	float n = l * p.z;
	rxy = max(rxy, -(n) / 4.);
	return (rxy) / abs(scale);
}

float fractal_de8( vec3 p )
{
	float scale = 1.0;
    
    float orb = 10000.0;

    for( int i=0; i<6; i++ )
	{
		p = -1.0 + 2.0*fract(0.5*p+0.5);

        p -= sign(p)*0.04; // trick
        
        float r2 = dot(p,p);
		float k = 0.95/r2;
		p     *= k;
		scale *= k;

        orb = min( orb, r2);
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
	}

    float d1 = sqrt( min( min( dot(p.xy,p.xy), dot(p.yz,p.yz) ), dot(p.zx,p.zx) ) ) - 0.02;
    float d2 = abs(p.y);
    float dmi = d2;
    float adr = 0.7*floor((0.5*p.y+0.5)*8.0);
    if( d1<d2 )
    {
        dmi = d1;
        adr = 0.0;
    }
    return 0.5*dmi/scale;
}

float fractal_de9( vec3 p )
{
    vec3 CSize = vec3(1., 1.7, 1.);
	p = p.xzy;
	float scale = 1.1;
	for( int i=0; i < 8;i++ )
	{
		p = 2.0*clamp(p, -CSize, CSize) - p;
		float r2 = dot(p,p);
		float k = max((2.)/(r2), .5);
		p     *= k;
		scale *= k;
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
	}
	float l = length(p.xy);
	float rxy = l - 1.0;
	float n = l * p.z;
	rxy = max(rxy, (n) / 8.);
	return (rxy) / abs(scale);
}

float fractal_de10( vec3 p )
{
    vec3 CSize = vec3(1., 1.7, 1.);
	p = p.xzy;
	float scale = 1.1;
	for( int i=0; i < 8;i++ )
	{
		p = 2.0*clamp(p, -CSize, CSize) - p;
        float r2 = dot(p,p+sin(p.z*.3)); //Alternate fractal
		float k = max((2.)/(r2), .5);
		p     *= k;
		scale *= k;
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
	}
	float l = length(p.xy);
	float rxy = l - 1.0;
	float n = l * p.z;
	rxy = max(rxy, (n) / 8.);
	return (rxy) / abs(scale);
}

// hard crash on desktop
float fractal_de11(vec3 p0){
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    for(int i = 0; i < 8; i++){
        p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
        p*=(0.9/dot(p.xyz,p.xyz));
        escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    p/=p.w;
    return abs(p.y)*0.25;
}

vec3 fold(vec3 p0){
    vec3 p = p0;
    if(length(p) > 1.2) return p;
    p = mod(p,2.)-1.;
    return p;
}
float fractal_de12(vec3 p0){
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    for(int i = 0; i < 12; i++){
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        p = abs(p);
        p.xyz = fold(p.xyz);
        p.xyz = mod(p.xyz-1., 2.)-1.;
        p*=(1.2/dot(p.xyz,p.xyz));
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    p/=p.w;
    return abs(p.x)*0.25;
}


// hard crash - maybe better with tile based renderer?
vec3 fold2(vec3 p0){
    vec3 p = p0;
    if(length(p) > 2.)return p;
        p = mod(p,2.)-1.;
    return p;
}
float fractal_de13(vec3 p0){
    vec4 p = vec4(p0*10., 1.);
    escape = 0.;
    for(int i = 0; i < 12; i++){
        //p.xyz = clamp(p.xyz, vec3(-2.3), vec3(2.3))-p.xyz;
        //p.xyz += sin(float(i+1));
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        p = abs(p);
        p.xyz = fold2(p.xyz);

        //p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
        p.xyz = mod(p.xyz-1., 2.)-1.;
        p*=(1.1/dot(p.xyz,p.xyz));
        //p*=1.2;
        escape += exp(-0.2*dot(p.xyz,p.xyz));

    }
    p/=p.w;
    return (abs(p.x)*0.25)/10.;
}


float fractal_de14(vec3 pos) {
	vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
    int iterations = 10;
    float Power = 2.2;
	for (int i = 0; i < iterations ; i++) {
		r = length(z);
		if (r>EPSILON) break;
		
		// convert to polar coordinates
		float theta = acos(z.z/r);
		float phi = atan(z.y,z.x);
		dr =  pow( r, Power-1.0)*Power*dr + 1.0;
		// scale and rotate the point
		float zr = pow( r,Power);
		theta = theta*Power;
		phi = phi*Power;
		
		// convert back to cartesian coordinates
		z = zr*vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
		z+=pos;
	}
	return 0.5*log(r)*r/dr;
}

float fractal_de15(vec3 p){
    p=abs(p)-1.2;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    if(p.x<p.y)p.xy=p.yx;

    float s=1.;
    for(int i=0;i<6;i++)
    {
      p=abs(p);
      float r=2./clamp(dot(p,p),.1,1.);
      s*=r;
      p*=r;
      p-=vec3(.6,.6,3.5);
    }
    float a=1.5;
    p-=clamp(p,-a,a);
    return length(p)/s;
}


float fractal_de16(vec3 p){
    // box fold
    p=abs(p)-15.;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    if(p.x<p.y)p.xy=p.yx;
    float s=2.;
    for (int i=0; i<8; i++)
    {
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.);
        float r=-1.55/max(.41,dot(p,p));
        s*=r;
        p*=r;
        p-=.5;
    }
    s=abs(s);
    return dot(p,normalize(vec3(1,2,3)))/s;
}

void sFold90(inout vec2 p)
{
    vec2 v=normalize(vec2(1,-1));
    float g=dot(p,v);
    p-=(g-sqrt(g*g+1e-1))*v;
}
float fractal_de17(vec3 p){
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
    p=abs(p)-1.8;
    sFold90(p.zy);
    sFold90(p.xy);
    sFold90(p.zx);
    float s=2.;
    vec3  offset=p*.5;
    for(int i=0;i<8;i++){
        p=1.-abs(p-1.);
        float r=-1.3*max(1.5/dot(p,p),1.5);
        s*=r;
        p*=r;
        p+=offset;
        p.zx*=rot(-1.2);
    }
    s=abs(s);
    float a=8.5;
    p-=clamp(p,-a,a);
    return length(p)/s;
#undef rot
}


float lpNorm(vec3 p, float n)
{
    p = pow(abs(p), vec3(n));
    return pow(p.x+p.y+p.z, 1.0/n);
}
float fractal_de18(vec3 p){
    vec3 offset=p*.5;
    float s=2.;
    for (int i=0; i<5; i++)
    {
        p=clamp(p,-1.,1.)*2.-p;
        float r=-10.*clamp(max(.3/pow(lpNorm(p,5.),2.),.3),.0,.6);
        s*=r;
        p*=r;
        p+=offset;
    }
    s=abs(s);
    float a=10.;
    p-=clamp(p,-a,a);
    return length(p)/s;
}


#define sabs1(p)sqrt((p)*(p)+1e-1)
#define sabs2(p)sqrt((p)*(p)+1e-3)

float fractal_de19(vec3 p){
    float s=2.;
    p=abs(p);
    for (int i=0; i<4; i++) 
    {
        p=1.-sabs2(p-1.);
        float r=-9.*clamp(max(.2/pow(min(min(sabs1(p.x),sabs1(p.y)),sabs1(p.z)),.5), .1), 0., .5);
        s*=r;
        p*=r;
        p+=1.;
    }
    s=abs(s);
    float a=2.;
    p-=clamp(p,-a,a);
    return length(p)/s-.01;
}

float fractal_de20(vec3 p)
{
    float s=3.;
    for(int i = 0; i < 4; i++) {
        p=mod(p-1.,2.)-1.;
        float r=1.2/dot(p,p);
        p*=r;
        s*=r;
    }
    p = abs(p)-0.8;
    if (p.x < p.z) p.xz = p.zx;
    if (p.y < p.z) p.yz = p.zy;
    if (p.x < p.y) p.xy = p.yx;
    return length(cross(p,normalize(vec3(0,1,1))))/s-.001;
}


float fractal_de21(vec3 p){
    float s=3.;
    p=abs(p);
    for (float i=0.; i<9.; i++){
        p-=clamp(p,-1.,1.)*2.;
        float r=6.62*clamp(.12/min(dot(p,p),1.),0.,1.);
        s*=r;
        p*=r;
        p+=1.5;
    }
    s=abs(s);
    float a=.8;
    p-=clamp(p,-a,a);
    return length(p)/s;
}


float fractal_de22(vec3 p){
    float s=12.;
    p=abs(p);
    vec3 offset=p*3.;
    for (float i=0.; i<5.; i++){
        p=1.-abs(p-1.);
        float r=-5.5*clamp(.3*max(2.5/dot(p,p),.8),0.,1.5);
        p*=r;
        p+=offset;
        s*=r;
    }
    s=abs(s);
    p=abs(p)-3.;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    if(p.x<p.y)p.xy=p.yx;
    float a=3.;
    p-=clamp(p,-a,a);
    return length(p.xz)/s;
}


float fractal_de23(vec3 p){
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
    p=abs(p)-3.;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    if(p.x<p.y)p.xy=p.yx;
    float s=3.;
    vec3  offset=p*1.2;
    for (float i=0.;i<8.;i++){
        p=1.-abs(p-1.);
        float r=-6.5*clamp(.41*max(1.1/dot(p,p),.8),.0,1.8);
        s*=r;
        p*=r;
        p+=offset;
        p.yz*=rot(-1.2);
    }
    s=abs(s);
    float a=20.;
    p-=clamp(p,-a,a);
    return length(p)/s;
#undef rot
}


float fractal_de24(vec3 p){
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
    p=abs(p)-2.;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    if(p.x<p.y)p.xy=p.yx;
    float s=2.5;
    vec3 off=p*2.8;
    for (float i=0.;i<6.;i++)
    {
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.);
        float r=-11.*clamp(.8*max(2.5/dot(p,p),.2),.3,.6);
        s*=r;
        p*=r;
        p+=off;
        p.yz*=rot(2.1);
    }
    s=abs(s);
    float a=30.;
    p-=clamp(p,-a,a);
    return length(p)/s;
#undef rot
}

float fractal_de25(vec3 p){
    p=abs(p);
    float s=3.;
    vec3  offset = p*.5;
    for (float i=0.; i<5.; i++){
        p=1.-abs(p-1.);
        float r=-3.*clamp(.57*max(3./dot(p,p),.9),0.,1.);
        s*=r;
        p*=r;
        p+=offset;
    }
    s=abs(s);
    return length(cross(p,normalize(vec3(1))))/s-.008;
}


float fractal_de26(vec3 p){
    p.xy=abs(p.xy)-2.;
    if(p.x<p.y)p.xy=p.yx;
    p.z=mod(p.z,4.)-2.;

    p.x-=3.2;
    p=abs(p);
    float s=2.;
    vec3 offset= p*1.5;
    for (float i=0.; i<5.; i++){
        p=1.-abs(p-1.);
        float r=-7.5*clamp(.38*max(1.2/dot(p,p),1.),0.,1.);
        s*=r;
        p*=r;
        p+=offset;
    }
    s=abs(s);
    float a=100.;
    p-=clamp(p,-a,a);
    return length(p)/s;
}

float fractal_de27(vec3 p){
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
  float s=1.;
  for(int i=0;i<3;i++){
    p=abs(p)-.3;
    if(p.x<p.y)p.xy=p.yx;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    p.xy=abs(p.xy)-.2;
    p.xy*=rot(.3);
    p.yz*=rot(.3);
    p*=2.;
    s*=2.;
  }
  p/=s;
  float h=.5;
  p.x-=clamp(p.x,-h,h);
  // torus SDF
  return length(vec2(length(p.xy)-.5,p.z))-.05;
#undef rot
}



float fractal_de28(vec3 p){
  float s=1.;
  for(int i=0;i<3;i++){
    p=abs(p)-.3;
    if(p.x<p.y)p.xy=p.yx;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    p.xy-=.2;
    p*=2.;
    s*=2.;
  }
  p/=s;
  float h=.5;
  p.x-=clamp(p.x,-h,h);
  // torus SDF
  return length(vec2(length(p.xy)-.5,p.z))-.05;
}


float fractal_de29(vec3 p){
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
  for(int i=0;i<3;i++){
    p=abs(p)-.3;
    if(p.x<p.y)p.xy=p.yx;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    p.xy-=.2;
    p.xy*=rot(.5);
    p.yz*=rot(.5);
  }
  float h=.5;
  p.x-=clamp(p.x,-h,h);
  // torus SDF
  return length(vec2(length(p.xy)-.5,p.z))-.05;
#undef rot
}




#define TAUg atan(1.)*8.

vec2 pmodg(vec2 p, float n)
{
  float a=mod(atan(p.y, p.x),TAUg/n)-.5 *TAUg/n;
  return length(p)*vec2(sin(a),cos(a));
}

float fractal_de30(vec3 p)
{
    for(int i=0;i<4;i++)
    {
        p.xy = pmodg(p.xy,10.);
        p.y-=2.;
        p.yz = pmodg(p.yz, 12.);
        p.z-=10.;
    }
    return dot(abs(p),normalize(vec3(13,1,7)))-.7;
}

float fractal_de31(vec3 p)
{
  p.x-=4.;
  p=mod(p,8.)-4.;
  for(int j=0;j<3;j++)
  {
     p.xy=abs(p.xy)-.3;
     // p.yz=abs(p.yz)-sin(time*2.)*.3+.1,
     p.yz=abs(p.yz)-.3+.1,
     p.xz=abs(p.xz)-.2;
  }
   return length(cross(p,vec3(.5)))-.1;
}




vec3 fold32(vec3 p0){
    vec3 p = p0;
    if(length(p) > 2.)return p;
    p = mod(p,2.)-1.;
    return p;
}
float fractal_de32(vec3 p0){
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    if(p.x > p.z)p.xz = p.zx;
    if(p.z > p.y)p.zy = p.yz;
    if(p.y > p.x)p.yx = p.xy;
    p = abs(p);
    for(int i = 0; i < 8; i++){
        p.xyz = fold32(p.xyz);
        p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
        p*=(1.1/clamp(dot(p.xyz,p.xyz),-0.1,1.));
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    p/=p.w;
    return abs(p.x)*0.25;
}



float fractal_de33(vec3 p0){ // hard crash on desktop
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    
    for(int i = 0; i < 8; i++){
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        if(p.y > p.x)p.yx = p.xy;
        p = abs(p);
        p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
        p*=(1.0/clamp(dot(p.xyz,p.xyz),-0.1,2.));
        p.xyz-=vec3(0.1,0.4,0.2);
        escape += exp(-0.2*dot(p.xyz,p.xyz));

    }
    p/=p.w;
    return abs(p.x)*0.25;
}


float fractal_de34(vec3 p0){ // hard crash on desktop
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    
    for(int i = 0; i < 8; i++){
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        if(p.y > p.x)p.yx = p.xy;
        p = abs(p);
        p.xyz = fract(p.xyz*0.5 - 1.)*2.-1.0;
        p*=(1.0/clamp(dot(p.xyz,p.xyz),-0.1,1.));
        p.xyz-=vec3(0.1,0.4,0.2);
        escape += exp(-0.2*dot(p.xyz,p.xyz));

    }
    p/=p.w;
    return abs(p.x)*0.25;
}

float fractal_de35(vec3 p){
    p=mod(p,2.)-1.;
    p=abs(p)-1.;
    if(p.x<p.z)p.xz=p.zx;
    if(p.y<p.z)p.yz=p.zy;
    if(p.x<p.y)p.xy=p.yx;
    float s=1.;
    for(int i=0;i<10;i++)
    {
      float r2=2./clamp(dot(p,p),.1,1.);
      p=abs(p)*r2-vec3(.6,.6,3.5);
      s*=r2;
    }
    return length(p)/s;
}


float fractal_de36(vec3 p)
{
	float itr=10.,r=0.1;

	p=mod(p-1.5,3.)-1.5;
	p=abs(p)-1.3;
	if(p.x<p.z)p.xz=p.zx;
	if(p.y<p.z)p.yz=p.zy;
 	if(p.x<p.y)p.xy=p.yx;
	float s=1.;
	p-=vec3(.5,-.3,1.5);
	for(float i=0.;i++<itr;)
    {
		float r2=2./clamp(dot(p,p),.1,1.);
		p=abs(p)*r2;
		p-=vec3(.7,.3,5.5);
		s*=r2;
	}
    return length(p.xy)/(s-r);
}


float fractal_de37(vec3 p)
{
	float s=2.,r2;
	p=abs(p);
    for(int i=0; i<12;i++){
		p=1.-abs(p-1.);
        r2=1.2/dot(p,p);
    	p*=r2;
    	s*=r2;
	}
	return length(cross(p,normalize(vec3(1))))/s-0.003;
}


float fractal_de38(vec3 p)
{
	float s=2.,r2;
	p=abs(p);
    for(int i=0; i<12;i++){
		p=1.-abs(p-1.);
        r2=(i%3==1)?1.3:1.3/dot(p,p);
    	p*=r2;
    	s*=r2;
	}
	return length(cross(p,normalize(vec3(1))))/s-0.003;
}



// this is the one that looks like chains
#define sabs39(a) sqrt(a * a + 0.005)
#define smin39(a,b) SMin1(a,b,0.0003)
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
float SMin1(float a, float b, float k)
{
    return a + 0.5 * ((b-a) - sqrt((b-a) * (b-a) + k));
}
vec2 fold39(vec2 p, int n)
{
    p.x=sabs39(p.x);
    vec2 v=vec2(0,1);
    for(int i=0;i<n;i++)
    {
        //p-=2.0*min(0.0,dot(p,v))*v;
        p-=2.0*smin39(0.0,dot(p,v))*v;
        v=normalize(vec2(v.x-1.0,v.y));
    }
    return p;    
}
float sdTorus39( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float fractal_de39(vec3 p)
{
#if 1
	float A=5.566;
    float c=7.0;
    p=mod(p,c)-c*0.5;
    p.xz=fold39(p.xz,5);
    for(int i=0;i<5;i++)
    {
    	p.xy=abs(p.xy)-2.0;
    	p.yz=abs(p.yz)-2.5;
    	p.xy*=rot(A);
    	p.yz*=rot(A*0.5);
    	p=abs(p)-2.0;
    }
#endif
    vec2 s=vec2(0.05,0.02);
    float h=0.08;
    float de=1.0;
    vec3 q=p;
    q.xy=fold39(q.xy,5);
    q.y-=2.;
    q.x-=clamp(q.x,-h,h);
    de=min(de,sdTorus39(q,s));
    q=p;
    q.xy*=rot(M_PI/exp2(5.0));
    q.xy=fold39(q.xy,5);
    q.y-=2.0;
    q.x-=clamp(q.x,-h,h);
    de=min(de,sdTorus39(q.xzy,s));
    return de;
}

#undef rot
#undef smin39
#undef sabs39



#define fold45(p)(p.y>p.x)?p.yx:p
float fractal_de40(vec3 p)
{
    float scale = 2.1,
           off0 = .8,
           off1 = .3,
           off2 = .83;
    vec3 off =vec3(2.,.2,.1);
	float s=1.0;
	for(int i = 0;++i<20;)
	{
		p.xy = abs(p.xy);
		p.xy = fold45(p.xy);
		p.y -= off0;
		p.y = -abs(p.y);
		p.y += off0;
		p.x += off1;
		p.xz = fold45(p.xz);
		p.x -= off2;
		p.xz = fold45(p.xz);
		p.x += off1;
		p -= off;
		p *= scale;
		p += off;
		s *= scale;
	}
	return length(p)/s;
}



float fractal_de41(vec3 p){
	float s=4.;
	p=abs(p);
	vec3 off=p*4.6;
	for (float i=0.; i<8.; i++){
        p=1.-abs(abs(p-2.)-1.); 
    	float r=-13.*clamp(.38*max(1.3/dot(p,p),.7),0.,3.3);
        s*=r;
		p*=r;
        p+=off;
    }
	return length(cross(p,normalize(vec3(1,3,3))))/s-.006;
}



//spiky forest - crash on desktop
#define rot(a)mat2(cos(a),sin(a),-sin(a),cos(a))
float lpNorm42(vec3 p, float n)
{
	p = pow(abs(p), vec3(n));
	return pow(p.x+p.y+p.z, 1.0/n);
}

float fractal_de42(vec3 p){
    vec3 q=p;
	float s = 2.5;
	for(int i = 0; i < 10; i++) {
        p=mod(p-1.,2.)-1.;
		float r2=1.1/pow(lpNorm42(abs(p),2.+q.y*10.),1.75);
    	p*=r2;
    	s*=r2;
        p.xy*=rot(.001);
    }
    return q.y>1.3?length(p)/s:abs(p.y)/s;
}
#undef rot



vec3 rotate43(vec3 p,vec3 axis,float theta)
{
    vec3 v = cross(axis,p), u = cross(v, axis);
    return u * cos(theta) + v * sin(theta) + axis * dot(p, axis);   
}
vec2 pmod43(vec2 p, float r)
{
    float a = mod(atan(p.y, p.x), (M_PI*2) / r) - 0.5 * (M_PI*2) / r;
    return length(p) * vec2(-sin(a), cos(a));
}
float fractal_de43(vec3 p)
{
    for(int i=0;i<5;i++)
    {
        p.xy = pmod43(p.xy,12.0);
        p.y-=4.0;
        p.yz = pmod43(p.yz,16.0);
        p.z-=6.8;
    }
    return dot(abs(p),rotate43(normalize(vec3(2,1,3)),normalize(vec3(7,1,2)),1.8))-0.3;
}



#define pmod44(p,n)length(p)*sin(vec2(0.,M_PI/2.)+mod(atan(p.y,p.x),2.*M_PI/n)-M_PI/n)
#define fold44(p,v)p-2.*min(0.,dot(p,v))*v;

float fractal_de44(vec3 p)
{
	float s = 1.0;
    p.z=fract(p.z)-.5;
	for(int i=0;i<20;i++) // very, very expensive de
	{
	    p.y += .15;
	    p.xz = abs(p.xz);
	    for(int j=0;j<2;j++)
        {
	        p.xy = pmod44(p.xy,8.);
	        p.y -= .18;
	    }
	    p.xy = fold44(p.xy,normalize(vec2(1,-.8)));
		p.y = -abs(p.y);
		p.y += .4;
	    p.yz = fold44(p.yz,normalize(vec2(3,-1)));
		p.x -= .47;
		p.yz = fold44(p.yz,normalize(vec2(2,-7)));
		p -= vec3(1.7,.4,0);
        float r2= 3.58/dot(p,p);
        p *= r2;
		p += vec3(1.8,.7,.0);
		s *= r2;
	}
	return length(p)/s;
}



float fractal_de45(vec3 p){
        p.z-=2.5;
        float s = 3.;
        float e = 0.;
        for(int j=0;j++<8;)
            s*=e=3.8/clamp(dot(p,p),0.,2.),
            p=abs(p)*e-vec3(1,15,1);
        return length(cross(p,vec3(1,1,-1)*.577))/s;
}


float fractal_de46(vec3 p){
    float s = 2.;
    float e = 0.;
    for(int j=0;++j<7;)
        p.xz=abs(p.xz)-2.3,
        p.z>p.x?p=p.zyx:p,
        p.z=1.5-abs(p.z-1.3+sin(p.z)*.2),
        p.y>p.x?p=p.yxz:p,
        p.x=3.-abs(p.x-5.+sin(p.x*3.)*.2),
        p.y>p.x?p=p.yxz:p,
        p.y=.9-abs(p.y-.4),
        e=12.*clamp(.3/min(dot(p,p),1.),.0,1.)+
        2.*clamp(.1/min(dot(p,p),1.),.0,1.),
        p=e*p-vec3(7,1,1),
        s*=e;
    return length(p)/s;
}



float fractal_de47(vec3 p)
{
  	float s = 4.;
	for(int i = 0; i < 8; i++)
    {
		p=mod(p-1.,2.)-1.;
		float r2=(i%3==0)?1.5:1.2/dot(p,p);
        p*=r2;
        s*=r2;
	}
	vec3 q=p/s;
	q.xz=mod(q.xz-.002,.004)-.002;
	return min(length(q.yx)-.0003,length(q.yz)-.0003);
}


float fractal_de48(vec3 p){
    p.z-=-1.;
    #define fold48(p,v)p-2.*min(0.,dot(p,v))*v;
    float s=3.;
    float l=0.;
    
    for(int i = 0;++i<15;)
    {
        p.xy=fold48(p.xy,normalize(vec2(1,-1.3)));
        p.y=-abs(p.y);
        p.y+=.5;
        p.xz=abs(p.xz);
        p.yz=fold48(p.yz,normalize(vec2(8,-1)));
        p.x-=.5;
        p.yz=fold48(p.yz,normalize(vec2(1,-2)));
        p-=vec3(1.8,.4,.1);
        l = 2.6/dot(p,p);
        p*=l;
        p+=vec3(1.8,.7,.2);
        s*=l;
    }
    return length(p.xy)/s;
    #undef fold48
}


float lpNorm49(vec3 p, float n)
{
	p = pow(abs(p), vec3(n));
	return pow(p.x+p.y+p.z, 1.0/n);
}

float fractal_de49(vec3 p){
	float s = 1.;
	for(int i = 0; i < 9; i++) {
		p=p-2.*round(p/2.);
		float r2=1.1/max(pow(lpNorm49(p.xyz, 4.5),1.6),.15);
    	p*=r2;
    	s*=r2;
	}
	return length(p)/s-.001;
}



#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
float hash50(float x){
    return fract(sin(x*234.123+156.2));
}
float lpNorm50(vec3 p, float n)
{
	p = pow(abs(p), vec3(n));
	return pow(p.x+p.y+p.z, 1.0/n);
}
float fractal_de50(vec3 p){
    vec2 id=floor(p.xz);
    p.xz=mod(p.xz,1.)-.5;
    p.y=abs(p.y)-.5;
    p.y=abs(p.y)-.5;
    p.xy*=rot(hash50(dot(id,vec2(12.3,46.7))));
    p.yz*=rot(hash50(dot(id,vec2(32.9,76.2))));
    float s = 1.;
	for(int i = 0; i < 6; i++) {
		float r2=1.2/pow(lpNorm50(p.xyz, 5.0),1.5);
		p-=.1;
    	p*=r2;
    	s*=r2;
		p=p-2.*round(p/2.);
	}
	return .6*dot(abs(p),normalize(vec3(1,2,3)))/s-.002;
}
#undef rot



float fractal_de51(vec3 p){
    for(int j=0;++j<8;)
        p.z-=.3,
        p.xz=abs(p.xz),
        p.xz=(p.z>p.x)?p.zx:p.xz,
        p.xy=(p.y>p.x)?p.yx:p.xy,
        p.z=1.-abs(p.z-1.),
        p=p*3.-vec3(10,4,2);

    return length(p)/6e3-.001;
}




float lpNorm52(vec3 p, float n)
{
	p = pow(abs(p),vec3(n));
	return pow(p.x+p.y+p.z,1./n);
}
float fractal_de52(vec3 p){
    float scale=4.5;
    float mr2=.5;
    float off=.5;
    float s=1.;
    vec3 p0 = p;
    for (int i=0; i<16; i++) {
        if(i%3==0)p=p.yzx;
        if(i%2==1)p=p.yxz;
        p -= clamp(p,-1.,1.)*2.;
        float r2=pow(lpNorm52(p.xyz,5.),2.);
        float g=clamp(mr2*max(1./r2,1.),0.,1.);
        p=p*scale*g+p0*off;
        s=s*scale*g+off;
    }
    return length(p)/s-.01;
}


// hard crash on desktop
#define sabs53(p) sqrt((p)*(p)+.8)
void sfold90_53(inout vec2 p)
{
    p=(p.x+p.y+vec2(1,-1)*sabs(p.x-p.y))*.5;
}
float fractal_de53(vec3 p)
{
	p=mod(p-1.5,3.)-1.5;
	p=abs(p)-1.3;
	sfold90_53(p.xz);
	sfold90_53(p.xz);
	sfold90_53(p.xz);
	
	float s=1.;
	p-=vec3(.5,-.3,1.5);
	for(float i=0.;i++<7.;)
    {
		float r2=2.1/clamp(dot(p,p),.0,1.);
		p=abs(p)*r2;
		p-=vec3(.1,.5,7.);
		s*=r2;
	}
    float a=3.;
    p-=clamp(p,-a,a);
    return length(p)/s-.005;
}
#undef sabs53




// hard crash on desktop
#define sabs_54(x)sqrt((x)*(x)+.005)
#define sabs2_54(x)sqrt((x)*(x)+1e-4)
#define smax_54(a,b) (a+b+sabs2_54(a-b))*.5

void sfold90_54(inout vec2 p)
{
    p=(p.x+p.y+vec2(1,-1)*sabs(p.x-p.y))*.5;
}

float fractal_de54(vec3 p){
    vec3 q=p;
    p=abs(p)-4.;
    sfold90_54(p.xy);
    sfold90_54(p.yz);
    sfold90_54(p.zx);
    
	float s=2.5;
	p=sabs_54(p);
	vec3  p0 = p*1.5;
	for (float i=0.; i<4.; i++){
    	p=1.-sabs2_54(sabs2_54(p-2.)-1.); 
    	float g=-5.5*clamp(.7*smax_54(1.6/dot(p,p),.7),.0,5.5);
    	p*=g;
    	p+=p0+normalize(vec3(1,5,12))*(5.-.8*i);
        s*=g;
	}
	s=sabs_54(s);
	float a=25.;
	p-=clamp(p,-a,a);
	
	q=abs(q)-vec3(3.7);
    sfold90_54(q.xy);
    sfold90_54(q.yz);
    sfold90_54(q.zx);
  	return smax_54(max(abs(q.y),abs(q.z))-1.3,length(p)/s-.00);
}

#undef sabs_54
#undef sabs2_54
#undef smax_54


float fractal_de55(vec3 p){
    float s=2.;
    float e=0.;
    vec3 q=vec3(3,3,.0);
    for(int i=0;
        i++<7;
        p=q-abs(p-q*.4)
    )
        s*=e=15./min(dot(p,p),15.),
        p=abs(p)*e-2.;
    return (length(p.xz)-.5)/s;
}


float fractal_de56(vec3 p){
    vec3 q;
    p-=vec3(1.,.1,.1);
    q=p;
    float s=1.5;
    float e=0.;
    for(int j=0;j++<15;s*=e)
        p=sign(p)*(1.2-abs(p-1.2)),
        p=p*(e=8./clamp(dot(p,p),.3,5.5))+q*2.;
    return length(p)/s;
}



float fractal_de57(vec3 p){
    p.xz=fract(p.xz)-.5;
    float k=1.;
    float s=0.;
    for(int i=0;i++<9;)
        s=2./clamp(dot(p,p),.1,1.),
        p=abs(p)*s-vec3(.5,3,.5),
        k*=s;
    return length(p)/k-.001;
}

// this one is very cool, but hard crash on desktop
float fractal_de58(vec3 p){
    float s=2.;
    float k=0.;
    p=abs(mod(p-1.,2.)-1.)-1.;
    for(int j=0;++j<9;)
        p=1.-abs(p-1.),
        p=p*(k=-1./dot(p,p))-vec3(.1,.3,.1),
        s*=abs(k);
    return length(p.xz)/s;
}




// SDF sphere
vec4 sphere59 (vec4 z) {
  float r2 = dot (z.xyz, z.xyz);
  if (r2 < 2.0)
    z *= (1.0 / r2);
  else z *= 0.5;
  return z;
}
// SDF box
vec3 box59 (vec3 z) {
  return clamp (z, -1.0, 1.0) * 2.0 - z;
}
float DE0_59 (vec3 pos) {
  vec3 from = vec3 (0.0);
  vec3 z = pos - from;
  float r = dot (pos - from, pos - from) * pow (length (z), 2.0);
  return (1.0 - smoothstep (0.0, 0.01, r)) * 0.01;
}
float DE2_59 (vec3 pos) {
  vec3 params = vec3 (0.5, 0.5, 0.5);
  vec4 scale = vec4 (-20.0 * 0.272321);
  vec4 p = vec4 (pos, 1.0), p0 = p;
  vec4 c = vec4 (params, 0.5) - 0.5; // param = 0..1

  for (float i = 0.0; i < 10.0; i++) {
    p.xyz = box59(p.xyz);
    p = sphere59(p);
    p = p * scale + c;
  }
  return length(p.xyz) / p.w;
}
float fractal_de59 (vec3 pos) {
  return max (DE0_59(pos), DE2_59(pos));
}

// by jorge2017a1
float de2_60(vec3 p) {
    vec3 op = p;
    p = abs(1.0 - mod(p, 2.));
    float r = 0., power = 8., dr = 1.;
    vec3 z = p;
    
    for (int i = 0; i < 7; i++) {
        op = -1.0 + 2.0 * fract(0.5 * op + 0.5);
        float r2 = dot(op, op);
        r = length(z);

        if (r > 1.616) break;
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);

        dr = pow(r, power - 1.) * power * dr + 1.;
        float zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;
        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z += p;
    }
    return (.5 * log(r) * r / dr);
}
float de1_60(vec3 p) {
    float s = 1.;
    float d = 0.;
    vec3 r,q;
        r = p;
      q = r;
    
    for (int j = 0; j < 6; j++) {
        r = abs(mod(q * s + 1.5, 2.) - 1.);	
        r = max(r, r.yzx);
        d = max(d, (.3 - length(r *0.985) * .3) / s);
        s *= 2.1;
    }
    return d;
}
float fractal_de60(vec3 p) {
    return min(de1_60(p), de2_60(p));
}


// by gaz
#define rot61(a) mat2(cos(a),sin(a),-sin(a),cos(a))
float fractal_de61(vec3 p){
	p=abs(p)-3.;
	if(p.x<p.z)p.xz=p.zx;
	if(p.y<p.z)p.yz=p.zy;
 	if(p.x<p.y)p.xy=p.yx;
 	float s=2.;
	vec3  off=p*.5;
	for(int i=0;i<12;i++){
		p=1.-abs(p-1.);
  		float k=-1.1*max(1.5/dot(p,p),1.5);
    	s*=abs(k);
   		p*=k;
		p+=off;
    	p.zx*=rot61(-1.2);
    }
	// orbit=log2(s);
	float a=2.5;
	p-=clamp(p,-a,a);
	return length(p)/s;
}
#undef rot61



// by mrange
void sphere_fold62(inout vec3 z, inout float dz) {
const float fixed_radius2 = 1.9;
const float min_radius2   = 0.5;
    float r2 = dot(z, z);
    if(r2 < min_radius2) {
        float temp = (fixed_radius2 / min_radius2);
        z *= temp;
        dz *= temp;
    } else if(r2 < fixed_radius2) {
        float temp = (fixed_radius2 / r2);
        z *= temp;
        dz *= temp;
    }
}
vec3 pmin62(vec3 a, vec3 b, vec3 k) {
  vec3 h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0);
  return mix(b, a, h) - k*h*(1.0-h);
}
void box_fold62(float k, inout vec3 z, inout float dz) {
  // soft clamp after suggestion from ollij
  const vec3  folding_limit = vec3(1.0);
  vec3 zz = sign(z)*pmin62(abs(z), folding_limit, vec3(k));
  z = zz * 2.0 - z;
}
float sphere62(vec3 p, float t) {
  return length(p)-t;
}
float torus62(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float fractal_de62(vec3 z) {
    const float scale = -2.8;
    vec3 offset = z;
    float dr = 1.0;
    float fd = 0.0;
    const float k = 0.05;
    for(int n = 0; n < 5; ++n) {
        box_fold62(k/dr, z, dr);
        sphere_fold62(z, dr);
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0;        
        float r1 = sphere62(z, 5.0);
        float r2 = torus62(z, vec2(8.0, 1));        
        float r = n < 4 ? r2 : r1;        
        float dd = r / abs(dr);
        if (n < 3 || dd < fd) {
          fd = dd;
        }
    }
    return fd;
}


// by gaz - hard crash on desktop
float fractal_de63(vec3 p){
    p.x<p.z?p=p.zyx:p;
    p.y<p.z?p=p.xzy:p;
    float s=3.;
    float l=0.;
    for(int j=0;j++<6;)
        s*=l=2./min(dot(p,p),1.),
        p=abs(p)*l-vec3(.5,.5,7);
    return length(cross(p,p/p))/s;
}


// by evilryu
void sphere_fold64(inout vec3 z, inout float dz) {
    float fixed_radius2 = 1.9;
    float min_radius2 = 0.1;
    float r2 = dot(z, z);
    if(r2 < min_radius2) {
        float temp = (fixed_radius2 / min_radius2);
        z *= temp;
        dz *= temp;
    }else if(r2 < fixed_radius2) {
        float temp = (fixed_radius2 / r2);
        z *= temp;
        dz *= temp;
    }
}
void box_fold64(inout vec3 z, inout float dz) {
    float folding_limit = 1.0;
    z = clamp(z, -folding_limit, folding_limit) * 2.0 - z;
}
float fractal_de64(vec3 z) {
    vec3 offset = z;
    float scale = -2.8;
    float dr = 1.0;
    for(int n = 0; n < 15; ++n) {
        box_fold64(z, dr);
        sphere_fold64(z, dr);
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0;
    }
    float r = length(z);
    return r / abs(dr);
}


//by mrange
vec3 mod3_65(inout vec3 p, vec3 size) {
  vec3 c = floor((p + size*0.5)/size);
  p = mod(p + size*0.5, size) - size*0.5;
  return c;
}
void sphere_fold65(float fr, inout vec3 z, inout float dz) {
const float fixed_radius2 = 4.5;
const float min_radius2   = 0.5;
  float r2 = dot(z, z);
  if(r2 < min_radius2) {
    float temp = (fr / min_radius2);
    z *= temp;
    dz *= temp;
  } else if(r2 < fr) {
    float temp = (fr / r2);
    z *= temp;
    dz *= temp;
  }
}
void box_fold65(float fl, inout vec3 z, inout float dz) {
  z = clamp(z, -fl, fl) * 2.0 - z;
}
float sphere65(vec3 p, float t) {
  return length(p)-t;
}
float torus65(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float mb65(float fl, float fr, vec3 z) {
  vec3 offset = z;
  const float scale = -3.0;
  float dr = 1.0;
  float fd = 0.0;
  for(int n = 0; n < 5; ++n) {
    box_fold65(fl, z, dr);
    sphere_fold65(fr, z, dr);
    z = scale * z + offset;
    dr = dr * abs(scale) + 1.0;        
    float r1 = sphere65(z, 5.0);
    float r2 = torus65(z, vec2(8.0, 1));        
    float r = n < 4 ? r2 : r1;        
    float dd = r / abs(dr);
    if (n < 3 || dd < fd) {
      fd = dd;
    }
  }
  return fd;
}
#define PATHA 0.4*vec2(0.11, 0.21)
#define PATHB 0.7*vec2(13.0, 3.0)
float fractal_de65(vec3 p) { 
  float tm = p.z;
  const float folding_limit = 2.3;
  const vec3  rep = vec3(10.0);
  
  vec3 wrap = vec3(sin(tm*PATHA)*PATHB, tm);
  vec3 wrapDeriv = normalize(vec3(PATHA*PATHB*cos(PATHA*tm), 1.0));
  p.xy -= wrap.xy;
  p -= wrapDeriv*dot(vec3(p.xy, 0), wrapDeriv)*0.5*vec3(1,1,-1);

  p -= rep*vec3(0.5, 0.0, 0.0);
  p.y *= (1.0 + 0.1*abs(p.y));
  vec3 i = mod3_65(p, rep);
  
  const float fixed_radius2 = 4.5;
  float fl = folding_limit + 0.3*sin(0.025*p.z+1.0)- 0.3; 
  float fr = fixed_radius2 - 3.0*cos(0.025*sqrt(0.5)*p.z-1.0);

  return mb65(fl, fr, p);
} 




// by xem
float fractal_de66(vec3 p){
    vec4 o=vec4(p,1);
    vec4 q=o;
    for(float i=0.;i<9.;i++){
      o.xyz=clamp(o.xyz,-1.,1.)*2.-o.xyz;
      o=o*clamp(max(.25/dot(o.xyz,o.xyz),.25),0.,1.)*vec4(11.2)+q;
    }
    return (length(o.xyz)-1.)/o.w-5e-4;
}



// by WAHa_06x36
float periodic67(float x,float period,float dutycycle)
{
	x/=period;
	x=abs(x-floor(x)-0.5)-dutycycle*0.5;
	return x*period;
}
float fractal_de67(vec3 pos)
{
	vec3 gridpos=pos-floor(pos)-0.5;
	float r=length(pos.xy);
	float a=atan(pos.y,pos.x);
	a+=12.*0.3*sin(floor(r/3.0)+1.0)*sin(floor(pos.z)*13.73);
	return min(max(max(
	periodic67(r,3.0,0.2),
	periodic67(pos.z,1.0,0.7+0.3*cos(4.))),
	periodic67(a*r,3.141592*2.0/6.0*r,0.7+0.3*cos(4.))),
	0.25);
}



// by dr2
vec2 Rot2D_68 (vec2 q, float a)
{
  vec2 cs;
  cs = sin (a + vec2 (0.5 * M_PI, 0.));
  return vec2 (dot (q, vec2 (cs.x, - cs.y)), dot (q.yx, cs));
}
float PrBoxDf_68 (vec3 p, vec3 b)
{
  vec3 d;
  d = abs (p) - b;
  return min (max (d.x, max (d.y, d.z)), 0.) + length (max (d, 0.));
}
float fractal_de68(vec3 p)
{
  vec3 b;
  float r, a;
  const float nIt = 5., sclFac = 2.4;
  b = (sclFac - 1.) * vec3 (1., 1.125, 0.625);
  r = length (p.xz);
  a = (r > 0.) ? atan (p.z, - p.x) / (2. * M_PI) : 0.;
  p.y = mod (p.y - 4. * a + 2., 4.) - 2.;
  p.x = mod (16. * a + 1., 2.) - 1.;
  p.z = r - 32. / (2. * M_PI);
  p.yz = Rot2D_68 (p.yz, 2. * M_PI * a);
  for (float n = 0.; n < nIt; n ++) {
    p = abs (p);
    p.xy = (p.x > p.y) ? p.xy : p.yx;
    p.xz = (p.x > p.z) ? p.xz : p.zx;
    p.yz = (p.y > p.z) ? p.yz : p.zy;
    p = sclFac * p - b;
    p.z += b.z * step (p.z, -0.5 * b.z);
  }
  return 0.8 * PrBoxDf_68 (p, vec3 (1.)) / pow (sclFac, nIt);
}



// by dr2
vec2 Rot2D_69 (vec2 q, float a)
{
  vec2 cs;
  cs = sin (a + vec2 (0.5 * M_PI, 0.));
  return vec2 (dot (q, vec2 (cs.x, - cs.y)), dot (q.yx, cs));
}
float PrBoxDf_69 (vec3 p, vec3 b)
{
  vec3 d;
  d = abs (p) - b;
  return min (max (d.x, max (d.y, d.z)), 0.) + length (max (d, 0.));
}
float fractal_de69(vec3 p)
{
  vec3 b;
  float r, a;
  const float nIt = 5., sclFac = 2.4;
  b = (sclFac - 1.) * vec3 (1., 1.125, 0.625);
  r = length (p.xz);
  a = (r > 0.) ? atan (p.z, - p.x) / (2. * M_PI) : 0.;
  p.x = mod (16. * a + 1., 2.) - 1.;
  p.z = r - 32. / (2. * M_PI);
  p.yz = Rot2D_69 (p.yz, M_PI * a);
  for (float n = 0.; n < nIt; n ++) {
    p = abs (p);
    p.xy = (p.x > p.y) ? p.xy : p.yx;
    p.xz = (p.x > p.z) ? p.xz : p.zx;
    p.yz = (p.y > p.z) ? p.yz : p.zy;
    p = sclFac * p - b;
    p.z += b.z * step (p.z, -0.5 * b.z);
  }
  return 0.8 * PrBoxDf_69 (p, vec3 (1.)) / pow (sclFac, nIt);
}




// by Kali
mat2 rot70(float a) {
    return mat2(cos(a),sin(a),-sin(a),cos(a));	
}
vec4 formula70(vec4 p) {
    p.xz = abs(p.xz+1.)-abs(p.xz-1.)-p.xz;
    p=p*2./clamp(dot(p.xyz,p.xyz),.15,1.)-vec4(0.5,0.5,0.8,0.);
    p.xy*=rot70(.5);
    return p;
}
float screen70(vec3 p) {
    float d1=length(p.yz-vec2(.25,0.))-.5;	
    float d2=length(p.yz-vec2(.25,2.))-.5;	
    return min(max(d1,abs(p.x-.3)-.01),max(d2,abs(p.x+2.3)-.01));
}
float fractal_de70(vec3 pos) {
    vec3 tpos=pos;
    tpos.z=abs(2.-mod(tpos.z,4.));
    vec4 p=vec4(tpos,1.5);
    float y=max(0.,.35-abs(pos.y-3.35))/.35;

    for (int i=0; i<8; i++) {p=formula70(p);}
    float fr=max(-tpos.x-4.,(length(max(vec2(0.),p.yz-3.)))/p.w);

    float sc=screen70(tpos);
    return min(sc,fr);	
}



// adapted from glkt
float smin71( float a, float b, float k ){
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}
float noise71(vec3 p){
    vec3 np = normalize(p);
    
    // previously some kind of bi-planar mapping to a texture
    // float a = sin(4.*np.x)*np.y*np.x; 
    // float b = sin(3.*np.y)*np.z*np.y; 
    // previously a = texture(iChannel0,iTime/20.+np.xy).x;      
    // previously b = texture(iChannel0,iTime/20.+.77+np.yz).x;

    // trying this, with twigl noise functions
    float a = 0.1*snoise2D(np.xy*10.);
    float b = 0.1*snoise2D(0.77+np.yz*10.);
    
    a = mix(a,.5,abs(np.x));
    b = mix(b,.5,abs(np.z));
    return mix(a+b-.4,.5,abs(np.y)/2.);
}
float fractal_de71(vec3 p){
    // spheres
    float d = (-1.*length(p)+3.)+1.5*noise71(p);    
    d = min(d, (length(p)-1.5)+1.5*noise71(p) );  
    // links
    float m = 1.5; float s = .03;    
    d = smin71(d, max( abs(p.x)-s, abs(p.y+p.z*.2)-.07 ) , m);          
    d = smin71(d, max( abs(p.z)-s, abs(p.x+p.y/2.)-.07 ), m );    
    d = smin71(d, max( abs(p.z-p.y*.4)-s, abs(p.x-p.y*.2)-.07 ), m );    
    d = smin71(d, max( abs(p.z*.2-p.y)-s, abs(p.x+p.z)-.07 ), m );    
    d = smin71(d, max( abs(p.z*-.2+p.y)-s, abs(-p.x+p.z)-.07 ), m );
    return d;
}


// by gaz
float fractal_de72(vec3 p){
    float g=1.;
    float e=0.;
    vec3 q=vec3(0);
    p.z-=1.;
    q=p;
    float s=2.;
    for(int j=0;j++<8;)
        p-=clamp(p,-.9,.9)*2.,
        p=p*(e=3./min(dot(p,p),1.))+q,
        s*=e;
    return length(p)/s;
}

// by unconed
vec4 fold1_73(vec4 z) {
    vec3 p = z.xyz;
    p = p - 2.0 * clamp(p, -1.0, 1.0);
    return vec4(p, z.w);
}
vec4 fold2_73(vec4 z) {
    vec3 p = z.xyz;
    p = p - 2.0 * clamp(p, -1.0, 1.0);
    return vec4(p * 2.0, 2.0 * z.w);
}
vec4 invertRadius_73(vec4 z, float radius2, float limit) {
  float r2 = dot(z.xyz, z.xyz);
  float f = clamp(radius2 / r2, 1., limit);
  return z * f;
}
vec4 affine_73(vec4 z, float factor, vec3 offset) {
  z.xyz *= factor;
  z.xyz += offset;
  z.w *= abs(factor);
  return z;
}
vec4 mandel_73(vec4 z, vec3 offset) {
  float x = z.x;
  float y = z.y;
  z.w = 2. * length(z.xy) * z.w + 1.;
  z.x = x*x - y*y + offset.x;
  z.y = 2.*x*y + offset.y;
  return z;
}
vec4 invert_73(vec4 z, float factor) {
  float r2 = dot(z.xyz, z.xyz);
  float f = factor / r2;
  return z * f;
}
vec4 rotateXY_73(vec4 z, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  mat2 m = mat2(c, s, -s, c);
  return vec4(m * z.xy, z.zw);
}
vec4 rotateXZ_73(vec4 z, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  mat2 m = mat2(c, s, -s, c);
  vec2 r = m * z.xz;
  return vec4(r.x, z.y, r.y, z.w);
}
vec4 shiftXY_73(vec4 z, float angle, float radius) {
  float c = cos(angle);
  float s = sin(angle);
  return vec4(vec2(c, s) * radius + z.xy, z.zw);
}
float fractal_de73(vec3 p) {
    //vec3 pmod = mod(p + 2.0, 4.0) - 2.0;
    vec4 z = vec4(p, 1.0);
    float t = 344. * .2; // change this number for different shapes
    vec3 vo1 = vec3(sin(t * .1), cos(t * .0961), sin(t * .017)) * 1.1;
    vec3 vo2 = vec3(cos(t * .07), sin(t * .0533), sin(t * .138)) * 1.1;
    vec3 vo3 = vec3(sin(t * .031), sin(t * .0449), cos(t * .201)) * 1.1;

    z = invertRadius_73(z, 10.0, 1.5);
    z = invertRadius_73(z, 10.0*10.0, 2.0);
    z = rotateXY_73(z, t);
    z = fold1_73(z);
    z = rotateXZ_73(z, t * 1.112);
    z.xyz += vo3;
    z = fold2_73(z);
    z.xyz += vo1;
    z = affine_73(z, -1.5, p);
    z = invertRadius_73(z, 4.0*4.0, 2.0);
    z = affine_73(z, -1.5, p);
    z = rotateXY_73(z, t * .881);
    z = fold1_73(z);
    z = rotateXZ_73(z, t * .783);
    z = fold1_73(z);
    z = affine_73(z, -1.5, p);
    z = invertRadius_73(z, 10.0*10.0, 3.0);
    z = fold1_73(z);
    z = fold1_73(z);
    z = affine_73(z, -1.5, p);
    z = invertRadius_73(z, 10.0*10.0, 2.0);

    vec3 po = vec3(0.0, 0.0, 0.0);
    vec3 box = abs(z.xyz);
    float d1 = (max(box.x - 2.0, max(box.y - 2.0, box.z - 10.0))) / z.w;
    float d2 = (max(box.x - 20.0, max(box.y - .5, box.z - .5))) / z.w;
    float d3 = min(d1, d2);
    if (d2 == d3) {
      escape = 1.0;
    }
    else {
      escape = 0.0;
    }
    return d3;
}



// by lewiz
void sphereFold74(inout vec3 z, inout float dz)
{
	float r2 = dot(z,z);
	if (r2 < 0.5)
    { 
		float temp = 2.0;
		z *= temp;
		dz*= temp;
	}
    else if (r2 < 1.0)
    { 
		float temp = 1.0 / r2;
		z *= temp;
		dz*= temp;
	}
}
void boxFold74(inout vec3 z, inout float dz)
{
	z = clamp(z, -1.0, 1.0) * 2.0 - z;
}
float fractal_de74(vec3 z)
{
    float scale = 2.0;
	vec3 offset = z;
	float dr = 1.0;
	for (int n = 0; n < 10; n++)
    {
		boxFold74(z,dr);
		sphereFold74(z,dr);
        z = scale * z + offset;
        dr = dr * abs(scale) + 1.0;
	}
	float r = length(z);
	return r / abs(dr);
}



// by gaz
float fractal_de75(vec3 p){
    vec3 q;
    p.z-=1.5;
    q=p;
    float e=0.;
    float s=3.;
    for(int j=0;j++<8;s*=e)
        p=sign(p)*(1.-abs(abs(p-2.)-1.)),
        p=p*(e=6./clamp(dot(p,p),.3,3.))+q-vec3(8,.2,8);
    return length(p)/s;
}


// by gaz
float fractal_de76(vec3 p){
#define R(a)a=vec2(a.x+a.y,a.x-a.y)*.7
#define G(a,n)R(a);a=abs(a)-n;R(a)
    p=fract(p)-.5;
    G(p.xz,.3);
    G(p.zy,.1);
    G(p.yz,.15);
    return .6*length(p.xy)-.01;
#undef R
#undef G
}




// by raziel
float op_u_77(float d1, float d2){
	return (d1 < d2) ? d1 : d2;
}
void sphere_fold_77(inout vec3 p, inout float dr, float m_rad_sq, float f_rad_sq, float m_rad_sq_inv){
    float r_sq = dot(p, p);
    if (r_sq < m_rad_sq){
        float t = f_rad_sq * m_rad_sq_inv;
        p *= t;
        dr *= t;
    }
    else if (r_sq < f_rad_sq){
        float t = f_rad_sq / r_sq;
        p *= t;
        dr *= t;
    }
}
void box_fold_77(inout vec3 p, float fold_limit){
    p = clamp(p, -fold_limit, fold_limit) * 2.0 - p;
}
// estimators return (dist, mat_id, custom_value)
float estimator_mandelbox_77(vec3 p, float scale, float m_rad_sq, float f_rad_sq, float fold_limit){
    vec3 off = p;
    float dr = 1.0;
    float mrs_inv = 1.0 / m_rad_sq;
    for (int i = 0; i < 10; ++i){
        box_fold_77(p, fold_limit);
        sphere_fold_77(p, dr, m_rad_sq, f_rad_sq, mrs_inv);

        p = scale * p + off;
        dr = dr * abs(scale) + 1.0;
        vec3 ot = p - vec3(0.5);
    }
    return length(p) / abs(dr);
}
vec3 mod_pos_77(vec3 p, float a, float b){
    p.zx = mod(p.zx, a) - b;  
    return p;
}
float fractal_de77(vec3 p){
    vec3 p_mb = mod_pos_77(p, 4.4, 2.2);
    float res_mb = estimator_mandelbox_77(p, -2.5, 0.1, 2.5, 1.0);
    // second
    vec3 p_pl = p;
    p_pl.y += 4.0;
    p_pl = mod_pos_77(p_pl, 2.0, 1.0);
    float res_pl = estimator_mandelbox_77(p_pl, -1.5, 0.3, 2.9, 1.0);

    return op_u_77(res_mb, res_pl);
}

// by gaz (fractal 40)
float fractal_de78(vec3 p){
    float s=2.;
    float l=dot(p,p);
    float e=0.;
    escape=0.;
    p=abs(abs(p)-.7)-.5;
    p.x<p.y?p=p.yxz:p;
    p.y<p.z?p=p.xzy:p;
    for(int i=0;i++<8;){
        s*=e=2./clamp(dot(p,p),.004+tan(12.)*.002,1.35);
        p=abs(p)*e-vec2(.5*l,12.).xxy;
        // escape+=exp(-0.002*dot(vec3(e),p));
    }
    return length(p-clamp(p,-1.,1.))/s;
}

// by gaz
float fractal_de79(vec3 p){
    p.z-=1.5;
    vec3 q=p;
    float s=1.5;
    float e=0.;
    for(int j=0;j++<8;s*=e)
        p=sign(p)*(1.2-abs(p-1.2)),
        p=p*(e=8./clamp(dot(p,p),.6,5.5))+q-vec3(.3,8,.3);
    return length(p)/s;
}

// by gaz
float fractal_de80(vec3 p){
    float e=1.,s,B=2.95,H=.9;
    s=2.;
    p.z=mod(p.z-2.,4.)-2.;
    for(int j=0;j++<8;)
    {
        p=abs(p);
        p.x<p.z?p=p.zyx:p;
        p.x=H-abs(p.x-H);
        p.y<p.z?p=p.xzy:p;
        p.xz+=.1;
        p.y<p.x?p=p.yxz:p;
        p.y-=.1;
    }
    p*=B;
    p-=2.5;
    s*=B;
    return length(p.xy)/s-.007;
}

// by gaz
float fractal_de81(vec3 p){
#define hash(n) fract(sin(n*234.567+123.34))
    float seed=dot(floor((p+3.5)/7.)+3.,vec3(123.12,234.56,678.22));   
    p-=clamp(p,-3.5,3.5)*2.;
	float scale=-5.;
	float mr2=.38;
	float off=1.2;
	float s=3.;
	p=abs(p);
	vec3  p0 = p;
	for (float i=0.; i<4.+hash(seed)*6.; i++){
    	p=1.-abs(p-1.);
    	float g=clamp(mr2*max(1.2/dot(p,p),1.),0.,1.);
    	p=p*scale*g+p0*off;
        s=s*abs(scale)*g+off;
	}
	return length(cross(p,normalize(vec3(1))))/s-.005;
#undef hash
}


// by gaz
float fractal_de82(vec3 p){
#define hash(n) fract(sin(n*234.567+123.34))
    float zoom=2.1;
    p*=zoom;
    float seed=dot(floor((p+3.5)/7.)+3.,vec3(123.12,234.56,678.22));   
    p-=clamp(p,-8.,8.)*2.;
	float s=3.*zoom;
	p=abs(p);
	vec3  p0 = p*1.6;
	for (float i=0.; i<10.; i++){
        p=1.-abs(abs(p-2.)-1.); 
    	float g=-8.*clamp(.43*max(1.2/dot(p,p),.8),0.,1.3);
        s*=abs(g);
		p*=g;
        p+=p0;
    }
	return length(cross(p,normalize(vec3(1))))/s-.005;
#undef hash
}


// by gaz
#define opRepEven(p,s) mod(p,s)-0.5*s 
#define opRepOdd(p,s) p-s*round(p/s)
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
float lpNorm_83(vec3 p, float n){
	p = pow(abs(p), vec3(n));
	return pow(p.x+p.y+p.z, 1.0/n);
}
vec2 pSFold_83(vec2 p,float n){
    float h=floor(log2(n)),a =6.2831*exp2(h)/n;
    for(float i=0.0; i<h+2.0; i++){
	 	vec2 v = vec2(-cos(a),sin(a));
		float g= dot(p,v);
 		p-= (g - sqrt(g * g + 5e-3))*v;
 		a*=0.5;
    }
    return p;
}
vec2 sFold45_83(vec2 p, float k){
    vec2 v = vec2(-1,1)*0.7071;
    float g= dot(p,v);
 	return p-(g-sqrt(g*g+k))*v;
}
float frameBox_83(vec3 p, vec3 s, float r){   
    p = abs(p)-s;
    p.yz=sFold45_83(p.yz, 1e-3);
    p.xy=sFold45_83(p.xy, 1e-3);
    p.x = max(0.0,p.x);
	return lpNorm_83(p,5.0)-r;
}
float sdRoundBox_83( vec3 p, vec3 b, float r ){   
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}
float deObj_83(vec3 p){   
    return min(sdRoundBox_83(p,vec3(0.3),0.05),frameBox_83(p,vec3(0.8),0.1));
}
float fractal_de83(vec3 p){
    float de=1.0;
    // p.z-=iTime*1.1;
    vec3 q= p;
    p.xy=pSFold_83(-p.xy,3.0);
    p.y-=8.5;
    p.xz=opRepEven(p.xz,8.5);
    float de1=length(p.yz)-1.;
    de=min(de,de1);
    p.xz=pSFold_83(p.xz,8.0);
    p.z-=2.0;
    float rate=0.5;
    float s=1.0;
    for(int i=0;i<3;i++){
        p.xy=abs(p.xy)-.8;
        p.xz=abs(p.xz)-0.5;
        p.xy*=rot(0.2);
        p.xz*=rot(-0.9);
        s*=rate;
        p*=rate;
        de=min(de,deObj_83(p/s));
    }
    q.z=opRepOdd(q.z,8.5);
    float de0=length(q)-1.5;
    de=min(de,de0);
    return de;    
}
#undef opRepEven
#undef opRepOdd
#undef rot


// by gaz
float fractal_de84(vec3 p){
    p.z-=16.;
    float s=3.;
    float e=0.;
    p.y=abs(p.y)-1.8;
    p=clamp(p,-3.,3.)*2.-p;
    s*=e=6./clamp(dot(p,p),1.5,50.);
    p=abs(p)*e-vec3(0,1.8,0);
    p.xz =.8-abs(p.xz-2.);
    p.y =1.7-abs(p.y-2.);
    s*=e=12./clamp(dot(p,p),1.0,50.);
    p=abs(p)*e-vec2(.2,1).xyx;
    p.y =1.5-abs(p.y-2.);
    s*=e=16./clamp(dot(p,p),.1,9.);
    // escape = e;
    p=abs(p)*e-vec2(.3,-.7).xyx;
    return min(
            length(p.xz)-.5,
            length(vec2(length(p.xz)-12.,p.y))-3.
            )/s;
}




// by gaz
vec3 rot_85(vec3 p,vec3 a,float t){
	a=normalize(a);
	vec3 v = cross(a,p),u = cross(v,a);
	return u * cos(t) + v * sin(t) + a * dot(p, a);   
}
float lpNorm_85(vec2 p, float n){
	p = pow(abs(p), vec2(n));
	return pow(p.x+p.y, 1.0/n);
}
float sdTorus_85( vec3 p, vec2 t ){
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float smin_85( float a, float b, float k ) {
    float h = clamp(.5+.5*(b-a)/k, 0., 1.);
    return mix(b, a, h) - k*h*(1.-h);
}
float deTetra_85(vec3 p){
	vec2 g=vec2(-1,1)*0.577;
	return pow(
		pow(max(0.0,dot(p,g.xxx)),8.0)
		+pow(max(0.0,dot(p,g.xyy)),8.0)
		+pow(max(0.0,dot(p,g.yxy)),8.0)
		+pow(max(0.0,dot(p,g.yyx)),8.0),
		0.125);
}
float deStella_85(vec3 p){
    p=rot_85(p,vec3(1,2,3),time*3.0);
	return smin_85(deTetra_85(p)-1.0,deTetra_85(-p)-1.0,0.05);
}
#define Circle 2.2
vec2 hash2_85( vec2 p ){
    p = mod(p, Circle*2.0); 
	return fract(sin(vec2(
        dot(p,vec2(127.1,311.7)),
        dot(p,vec2(269.5,183.3))
    ))*43758.5453);
}
vec3 voronoi_85(vec2 x){
    x*=Circle;
    vec2 n = floor(x);
    vec2 f = fract(x);
	vec2 mg, mr;
    float md = 8.0;
    for( int j=-1; j<=1; j++ )
    for( int i=-1; i<=1; i++ ){
        vec2 g = vec2(float(i),float(j));
		vec2 o = hash2_85( n + g );
		o = 0.5 + 0.5*sin( time*0.3 + 6.2831*o );
        vec2 r = g + o - f;
        float d = dot(r,r);
        if( d<md ){
            md = d;
            mr = r;
            mg = g;
        }
    }
    md = 8.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ ){
        vec2 g = mg + vec2(float(i),float(j));
		vec2 o = hash2_85( n + g );
		o = 0.5 + 0.5*sin( time*0.3 + 6.2831*o );
        vec2 r = g + o - f;
        if( dot(mr-r,mr-r)>0.00001 )
        md = min( md, dot( 0.5*(mr+r), normalize(r-mr) ) );
    }
    return vec3( md, mr );
}
float voronoiTorus_85(vec3 p){
    vec2 size = vec2(12,5);
    vec2 q = vec2(length(p.xz) - size.x, p.y);
	vec2 uv=vec2(atan(p.z, p.x),atan(q.y, q.x))/3.1415;
	vec3 vr=voronoi_85(uv*vec2(20,8));
    vec2 p2=vec2(lpNorm_85(vr.yz,12.0)-0.5, sdTorus_85(p,size));
	return lpNorm_85(p2,5.0)-0.1; 
}
float fractal_de85(vec3 p)
{   
    vec3 offset = vec3(6,0,0);
    float de = min(voronoiTorus_85(p-offset),voronoiTorus_85(p.xzy+offset));
    vec3 co = vec3(cos(time),0,sin(time))*10.0;
    float s1= abs(sin(time))*3.0+2.0;
    float deSG = min(deStella_85((p-co-offset)/s1),deStella_85((p-(co-offset).xzy)/s1))*s1;
    float deS = min(deStella_85(p-co-offset),deStella_85(p-(co-offset).xzy));
    de=min(de,deS);    
    return de;
}
#undef Circle

// by nameless
float fractal_de86(vec3 p0){
    vec4 p = vec4(p0/10., 1.);
    escape = 0.;
    p=abs(p);
    for(int i = 0; i < 8; i++){
        if(p.x > p.z)p.xz = p.zx;
        if(p.z > p.y)p.zy = p.yz;
        if(p.y > p.x)p.yx = p.xy;
        p*=(1.3/clamp(dot(p.xyz,p.xyz),0.1,1.));
        p.xyz-=vec3(.5,0.2,0.2);
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    float m = 1.5;
    p.xyz-=clamp(p.xyz,-m,m);
return (length(p.xyz)/p.w)*10.;
}

// by nameless
float fractal_de87(vec3 p0){
    vec4 p = vec4(p0/10., 1.);
    escape = 0.;
    p=abs(p);
    if(p.x < p.z)p.xz = p.zx;
    if(p.z < p.y)p.zy = p.yz;
    if(p.y < p.x)p.yx = p.xy;
    for(int i = 0; i < 6; i++){
        if(p.x < p.z)p.xz = p.zx;
        if(p.z < p.y)p.zy = p.yz;
        if(p.y < p.x)p.yx = p.xy;
        p = abs(p);
        p*=(1.9/clamp(dot(p.xyz,p.xyz),0.1,1.));
        p.xyz-=vec3(0.2,1.9,0.6);
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    float m = 1.2;
    p.xyz-=clamp(p.xyz,-m,m);
    return (length(p.xyz)/p.w)*10.;
}


// by nameless
float fractal_de88(vec3 p0){
    vec4 p = vec4(p0/10., 1.);
    escape = 0.;
    p=abs(p);
    if(p.x < p.z)p.xz = p.zx;
    if(p.z < p.y)p.zy = p.yz;
    if(p.y < p.x)p.yx = p.xy;
    for(int i = 0; i < 6; i++){
        if(p.x < p.z)p.xz = p.zx;
        if(p.z < p.y)p.zy = p.yz;
        if(p.y < p.x)p.yx = p.xy;
        p = abs(p);

        p*=(2./clamp(dot(p.xyz,p.xyz),0.1,1.));
        p.xyz-=vec3(0.9,1.9,0.9);
        // escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    float m = 1.5;
    p.xyz-=clamp(p.xyz,-m,m);
    return (length(p.xyz)/p.w)*10.;
}



// by Ivan Dianov
float fractal_de89(vec3 p){
#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
  p.z-=.25;
  float j=0.,c=0.,s=1.;
  p.y = fract(p.y)-.5;
  for(;j<10.;j++){
    p=abs(p);
    p-=vec2(.05,.5).xyx;
    p.xz*=rot(1.6);
    p.yx*=rot(.24);
    p*=2.;
    s*=2.;
  }
  return (length(p)-1.)/s*.5;
#undef rot
}


// by yonatan - some aliasing
float fractal_de90(vec3 p){
    float j = 0.5;
    for(p.xz=mod(p.xz,6.)-3.;++j<9.;p=3.*p-.9)
        p.xz=abs(p.xz),
        p.z>p.x?p=p.zyx:p,
        p.y>p.z?p=p.xzy:p,
        p.z--,
        p.x-=++p.y*.5;
    return min(.2,p.x/4e3-.2);
}


// by gaz
float fractal_de91(vec3 p){
    float s=4.;
    float l=0;
    p.z-=.9;
    vec3 q=p;
    s=2.;
    for(int j=0;j++<9;)
        p-=clamp(p,-1.,1.)*2.,
        p=p*(l=8.8*clamp(.72/min(dot(p,p),2.),0.,1.))+q,
        s*=l;
    return length(p)/s;
}


// by gaz
float fractal_de92(vec3 p){
    float s=3., l=0.;
    //p=g*d+vec3(0,0,t);
    vec3 q=p;
    p=mod(p,4.)-2.;
    p=abs(p);
    for(int j=0;j++<8;)
        p=1.-abs(p-1.),
        p=p*(l=-1.*max(1./dot(p,p),1.))+.5,
        s*=l;
    return max(.2-length(q.xy),length(p)/s);
}


// by eiffie  https://www.shadertoy.com/view/4sy3zh
float fractal_de93(vec3 p){
    const int iters=5;
    const int iter2=3;
    const float scale=3.48;
    const vec3 offset=vec3(1.9,0.0,2.56);
    const float psni=pow(scale,-float(iters));
    const float psni2=pow(scale,-float(iter2));

    p = abs(mod(p+3., 12.)-6.)-3.;
	vec3 p2;
	for (int n = 0; n < iters; n++) {
		if(n==iter2)p2=p;
		p = abs(p);
		if (p.x<p.y)p.xy = p.yx;
		p.xz = p.zx;
		p = p*scale - offset*(scale-1.0);
		if(p.z<-0.5*offset.z*(scale-1.0))
            p.z+=offset.z*(scale-1.0);
	}
    float d1=(length(p.xy)-1.0)*psni;
    float d2=length(max(abs(p2)-vec3(0.2,5.1,1.3),0.0))*psni2;
    escape=(d1<d2)?0.:1.;
	return min(d1,d2);
}


// by yonatan
float fractal_de94(vec3 p){
    p=fract(p)-.5;
    vec3 O=vec3(2.,0,3.);
    for(int j=0;j++<7;){
        p=abs(p);
        p=(p.x<p.y?p.zxy:p.zyx)*3.-O;
        if(p.z<-.5*O.z)
            p.z+=O.z;
    } 
    return length(p.xy)/3e3;
}


// by yonatan
float fractal_de95(vec3 p){
    p=fract(p)-.5;
    vec3 O=vec3(2.,0,5.);
    for(int j=0;j++<7;){
        p=abs(p);
        p=(p.x<p.y?p.zxy:p.zyx)*3.-O;
        if(p.z<-.5*O.z)
            p.z+=O.z;
    } 
    return length(p.xy)/3e3;
}


// by gaz
float fractal_de96(vec3 p){
    vec3 a=vec3(.5);
    p.z-=55.;
    p = abs(p);
    float s=2., l=0.;
    for(int j=0;j++<8;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        s*=l=-2.12/max(.2,dot(p,p)),
        p=p*l-.55;
    return dot(p,a)/s;
}


// variant of code by gaz
float fractal_de97(vec3 p){
    vec3 a=vec3(.5);
    p.z-=55.;
    float s=2., l=0.;
    for(int j=0;j++<8;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        s*=l=-2.12/max(.2,dot(p,p)),
        p=p*l-.55;
    return dot(p,a)/s;
}


// variant of code by gaz
float fractal_de98(vec3 p){
    vec3 a=vec3(.5, 0.1, 0.2);
    p.z-=55.;
    float s=2., l=0.;
    for(int j=0;j++<8;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        s*=l=-2.12/max(.2,dot(p,p)),
        p=p*l-.55;
    return dot(p,a)/s;
}


// by gaz
float fractal_de99(vec3 p){
    float i,g,e=1.,s,l;
    vec3 a=vec3(.5);
    p.z-=55.;
    p=abs(p);
    s=2.;
    for(int j=0;j++<8;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        s*=l=-1.55/max(.4,dot(p,p)),
        p=p*l-.535;
    return dot(p,a)/s;
}


// by gaz
float fractal_de100(vec3 p){
    float i,g,e,R,S;vec3 q;
    q=p*2.;
    R=7.;
    for(int j=0;j++<9;){
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.);
   
        S=-9.*clamp(.7/min(dot(p,p),3.),0.,1.);
        p=p*S+q;
        R=R*abs(S);
    }
    return length(p)/R; 
}


// by gaz
void rot101(inout vec3 p,vec3 a,float t){
	a=normalize(a);
	vec3 u=cross(a,p),v=cross(a,u);
	p=u*sin(t)+v*cos(t)+a*dot(a,p);   
}
#define G dot(p,vec2(1,-1)*.707)
#define V v=vec2(1,-1)*.707
void sfold101(inout vec2 p)
{
    vec2 v=vec2(1,-1)*.707;
    float g=dot(p,v);
    p-=(G-sqrt(G*G+.01))*v;
}
#undef G
#undef V
float fractal_de101(vec3 p){
    float k=.01;
    for(int i=0;i<8;i++)
    {
        p=abs(p)-1.;
        sfold101(p.xz);
        sfold101(p.yz);
        sfold101(p.xy);
        rot101(p,vec3(1,2,2),.6);
        p*=2.;
    }
    return length(p.xy)/exp2(8.)-.01;
}


// by gaz
float fractal_de102(vec3 p){
    #define V vec2(.7,-.7)
    #define G(p)dot(p,V)
    float i=0.,g=0.,e=1.;
    float t = 0.34; // this was the time varying parameter - change it to see different behavior
    for(int j=0;j++<8;){
        p=abs(rotate3D(0.34,vec3(1,-3,5))*p*2.)-1.,
        p.xz-=(G(p.xz)-sqrt(G(p.xz)*G(p.xz)+.05))*V;
    }
    return length(p.xz)/3e2;
    #undef V
    #undef G
}



// by Kali
float fractal_de103(vec3 p) {
    const float width=.22;
    const float scale=4.;
	// float t=iTime;
	float t=0.2;
	float dotp=dot(p,p);
	p.x+=sin(t*40.)*.007;
	p=p/dotp*scale;
	p=sin(p+vec3(sin(1.+t)*2.,-t,-t*2.));
	float d=length(p.yz)-width;
	d=min(d,length(p.xz)-width);
	d=min(d,length(p.xy)-width);
	d=min(d,length(p*p*p)-width*.3);
	return d*dotp/scale;
}



// by yonatan
float fractal_de104(vec3 p){
    
    #define F(X)d=min(d,length(p.X))-3e-4;
    float d=1.,i,h=0.,D;
    float t = 1.2; // previously time varying
    // p=vec3((FC.xy*2.-r)/r.y,1)*h,
    p*=h;
    p.z--;
    p=fract(p*rotate3D(t,vec3(1))/(D=dot(p,p)))-.5;
    F(xy)
    F(yz)
    F(zx)
    h+=d*D*.5;
  // o.xyz+=5e-4/abs(D);
    return D; // it's either d, D, or h, but it's crashing on my desktop so I'll need to do this on the laptop
}


// by butadiene121 - maybe not a valid distance bound, I'd like to figure this one out
// https://twitter.com/butadiene121/status/1392392236730486786
// bit.ly/3uIxVQO
float fractal_de105(vec3 p)
{   
    vec3 m=vec3(0,1.3,time*0.15),q;
    float e=0.,c=0.,d=1.;
    q=fract(m)-.5,
    q.y=m.y;
    for(int j=0;j<12;j++)
        q=abs(q),
        q.y-=2.,
        c=2./clamp(dot(q,q),.4,1.),
        q*=c,
        d*=c,
        q.xz-=.5,
        q.y--;
    m+=e*p;
    if(e>.001)escape+=.02*exp(-3.*e);
    return length(q)/d-.001;
}


// by yonatan - almost like asteroids
float fractal_de106(vec3 p){
    float i,a,n,h,d=1.,t=0.3; // t is the time varying term, change it for different behavior
    vec3 q;
    n=.4;
    for(a=1.;a<2e2;n+=q.x*q.y*q.z/a)
        p.xy*=rotate2D(a+=a),
        q=cos(p*a+t);
    return n*.3;
}


// by yonatan - kind of a landscape sort of thing
float fractal_de107(vec3 p){
    vec3 z,q;
    p.z -= 9.;
    z=p;
    float a=1.,n=.9;
    for(int j=0;j++<15;){
        p.xy*=rotate2D(float(j*j));
        a*=.66;
        q=sin(p*=1.5);
        n+=q.x*q.y*q.z*a;
    }
    return (n*.2-z.z*.2);
}



// by yonatan
float fractal_de108(vec3 p){
      vec3 q;
      float s=1., a=1., n=.5;
      for(int j=0;j++<9;){
          p.xy*=rotate2D(float(j*j));
          a*=.5;
          q=sin(p+=p);
          n+=q.x*q.y*q.z*a;
      }
      return n*.2;
}



// by yonatan
float fractal_de109(vec3 p){
    float h,d=1.,i,u,s, t = 0.8; // t was the time varying term
    p+=vec3(1,1,sin(t/4.)*3.);
    s=2.;
    for(int j=0;j<9;j++){
        p.xy*=rotate2D(t/4.);
        u=4./3./dot(p,p);
        s*=u;
        p=mod(1.-p*u,2.)-1.;
    }
    return (length(p)/s);
}



// by gaz, with some adaptation
float fractal_de110(vec3 p){
    float i,g,d=1.,s,h;
    vec3 e,q;
    s=2.;h=.3;
    for(int j=0;j++<8;){
        p=abs(p)-1.;
        q=p;
        for(int k=0;++k<3;)
            p-=clamp(dot(q,e=vec3(9>>k&1,k>>1&1,k&1)-.5),-h,h)*e*2.;
        p*=1.4;s*=1.4;
    }
    return length(p)/(4.*s); // play with this scale factor to match to a given epsilon
}



// by gaz
float  fractal_de111(vec3 p){
    float i,g,e=1.,s,l;
    // vec3 p=rotate3D(t,vec3(1))*vec3(g*(FC.xy*2.-r)/r.y,g-9.);
    p.z-=9.;
    s=2.;
    p=abs(p);
    for(int j=0;j++<6;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        p*=l=-2./max(.3,sqrt(min(min(p.x,p.y),p.z))),
        p-=2.,
        s*=l;
    return length(p)/s;
}



// by gaz
float fractal_de112(vec3 p){
    float i,g,e,s;
    vec3 q=p;
    s=5.;
    for(int j=0;j++<6;s*=e)
        p=sign(p)*(1.7-abs(p-1.7)),
        p=p*(e=8./clamp(dot(p,p),.3,5.))+q-vec3(.8,12,.8);
    return length(p.yz)/s;
}



// adapted from code by catzpaw
float fractal_de113(vec3 p){
    float k = M_PI*2.;
    vec3 v = vec3(0.,3.,fract(k));
    return (length(cross(p=cos(p+v),p.zxy))-0.1)*0.4;
}


// happy accident while converting 113
float fractal_de114(vec3 p){
    float k = M_PI*2.;
    vec3 v = vec3(0.,3.,fract(k));
    return (length(cross(cos(p+v),p.zxy))-0.4)*0.2;
}


// by catzpaw (distance bound doesn't hold)
float fractal_de115(vec3 p){
    vec3 v=vec3(0,1.5,6.3);
    return min(6.-length((p-v).xy+sin(p.yx)), dot(cos(p),sin(p.yzx)))+sin(sin(p.z*3.5)+v.z)*.1+1.;
}


// by gaz
float fractal_de116(vec3 p){
    // p=rotate3D(t,vec3(1))*vec3(g*(FC.xy*2.-r)/r.y,g-80.);
    p.z-=80.;
    float s=3., l=0.;
    p=abs(p);
    for(int j=0;j++<8;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        p*=l=-.8/min(2.,length(p)),
        p-=.5,
        s*=l;
    return (length(p)/s)-0.1;
}


// by gaz
float fractal_de117(vec3 p){
    float s = 1.;
    for(int j=0;j<7;j++)
        p=mod(p-1.,2.)-1.,
        p*=1.2,
        s*=1.2,
        p=abs(abs(p)-1.)-1.;
    return (length(cross(p,normalize(vec3(2,2.03,1))))/s)-0.02;
}


// by gaz
float fractal_de118(vec3 p){
    float s=2., l=0.;
    p=abs(p);
    for(int j=0;j++<8;)
        p=1.-abs(abs(p-2.)-1.),
        p*=l=1.2/dot(p,p),
        s*=l;
    return dot(p,normalize(vec3(3,-2,-1)))/s;
}


// by gaz
float fractal_de119(vec3 p){
    float s=2., l=0.;
    p=abs(p);
    for(int j=0;j++<8;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        p*=l=-1.3/dot(p,p),
        p-=.15,
        s*=l;
    return length(p)/s;
}


// by gaz
float fractal_de120(vec3 p){
    p.z-=20;
    float s=3., l=0.;
    p=abs(p);
    for(int j=0;j++<10;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        p*=l=-1./max(.19,dot(p,p)),
        p-=.24,
        s*=l;
    return (length(p)/s);
}


// by gaz
float fractal_de121(vec3 p){
      // vec3 p=g*normalize(vec3((FC.xy-.5*r)/r.y,1))+vec3(0,1,t);
      float s=2., l=0.;
      p=abs(mod(p-1.,2.)-1.);
      for(int j=0;j++<8;)
          p=1.-abs(abs(abs(p-5.)-2.)-2.),
          p*=l=-1.3/dot(p,p),
          p-=vec3(.3,.3,.4),
          s*=l;
      return length(p.yz)/s;
}


// by gaz
float fractal_de122(vec3 p){
    float i,g,e=1.,R,S;
    vec3 q;
    q=p*8.;
    R=8.;
    for(int j=0;j++<6;)
        p=-sign(p)*(abs(abs(abs(p)-2.)-1.)-1.),
        S=-5.*clamp(1.5/dot(p,p),.8,5.),
        p=p*S+q,
        R*=S;
    return length(p)/R;
}


// by gaz - hard crash on desktop
float fractal_de123(vec3 p){
    float i,g,e,s,l;
    vec3 q;
    s=2.;
    p=abs(mod(p-1.,2.)-1.)-1.;
    for(int j=0;j<8;j++)
        p=1.-abs(abs(abs(p-5.)-2.)-2.),
        p=p*(l=-1.4/dot(p,p))-vec3(.2),
        s*=abs(l);
    return length(p.xy)/s;
}



// by gaz
float fractal_de124(vec3 p){
    float i,g,e,s,k;
    vec3 q;
    p=vec3(length(p.xy)-PI,atan(p.y,p.x)*PI,p.z);
    p.yz=mod(p.yz,4.)-2.;
    s=2.;
    p=abs(p);
    q=p;
    for(int j=0;++j<5;)
        p=1.-abs(p-1.),
        p=-p*(k=max(3./dot(p,p),3.))+q,
        s*=k;
    return length(p.xz)/s;
}



// by gaz
float fractal_de125(vec3 p){
    float i,g,e,R,S;
    R=2.;
    for(int j=0;j++<9;)
        p=1.-abs(p-1.),
        p*=S=(j%3>1)?1.3:1.2/dot(p,p),
        R*=S;
    return length(cross(p,vec3(.5)))/R-5e-3;
}



// by gaz
float fractal_de126(vec3 p){
	float s=2.,r2;
	p=abs(p);
    for(int i=0; i<12;i++){
		p=1.-abs(p-1.);
		r2=(i%3==1)?1.1:1.2/dot(p,p);
    	p*=r2;
    	s*=r2;
	}
	return length(cross(p,normalize(vec3(1))))/s-0.005;
}



// by gaz
float fractal_de127(vec3 p){
    float i,g,e,s,l;
    vec3 q;
    q=p;s=3.;
    for(int j=0;j++<9;)
        p=mod(p-1.,2.)-1.,
        l=1.2/pow(pow(dot(pow(abs(p),vec3(5)),vec3(1)),.2),1.6),
        p*=l,
        s*=l;
    return abs(p.y)/s;
}


// by gaz - https://twitter.com/gaziya5/status/1291673093694357505
float fractal_de128(vec3 p){
    float i,g,e,s,l;
    vec3 q;
    q=p;
    s=4.;
    for(int j=0;j++<9;)
        p=mod(p-1.,2.)-1.,
        l=1.2/dot(p,p),
        p*=l,
        s*=l;
    return abs(p.y)/s;
}


// by gaz - hard crash on desktop
float fractal_de129(vec3 p){
    float i,g,e,s,l;
    vec3 q;
    q=p;
    s=1.;
    for(int j=0;j++<4;)
        p=mod(p-1.,2.)-1.,
        l=2./dot(p,p),
        p*=l,
        s*=l;
    return length(p.xy)/s;
}


// by gaz
float fractal_de130(vec3 p){
#define F1(s)p.s=abs(p.s)-1.
    p+=vec3(0,3.8,5.);
    vec3 q=p;
    p=mod(p,vec3(8,8,2))-vec3(4,4,1);
    F1(yx);
    F1(yx);
    F1(xz);
    return min(length(cross(p,vec3(.5)))-.03,length(p.xy)-.05);
#undef F1
}


// by gaz
float fractal_de131(vec3 p){
    float l,s=3.;
    float t = 4.5;
    for(int j=0;j++<5;p.xy=fract(p.xy+p.x)-.5)
        p=vec3(log(l=length(p.xy)),atan(p.y,p.x)/PI*2.,p.z/l+1.),
        s*=.5*l;
    return abs(p.z)*s;
}




// by iq - 'Fractal Cave'
float maxcomp132(in vec3 p ) { return max(p.x,max(p.y,p.z));}
float sdBox132( vec3 p, vec3 b ){
  vec3  di = abs(p) - b;
  float mc = maxcomp132(abs(p)-b);
  return min(mc,length(max(di,0.0)));
}
float fractal_de132(vec3 p){
    vec3 w = p; vec3 q = p;
    q.xz = mod( q.xz+1.0, 2.0 ) -1.0;
    float d = sdBox132(q,vec3(1.0));
    float s = 1.0;
    for( int m=0; m<7; m++ ){
        float h = float(m)/6.0;
        p =  q.yzx - 0.5*sin( 1.5*p.x + 6.0 + p.y*3.0 + float(m)*5.0 + vec3(1.0,0.0,0.0));
        vec3 a = mod( p*s, 2.0 )-1.0;
        s *= 3.0;
        vec3 r = abs(1.0 - 3.0*abs(a));
        float da = max(r.x,r.y);
        float db = max(r.y,r.z);
        float dc = max(r.z,r.x);
        float c = (min(da,min(db,dc))-1.0)/s;
        d = max( c, d );
   }
   return d*0.5;
}


// by gaz
float fractal_de133(vec3 p){
    float i,g,e,R,S;
    vec3 q;
    q=p;
    R=2.;
    for(int j=0;j++<9;)
        p-=clamp(p,-1.,1.)*2.,
        S=9.*clamp(.7/min(dot(p,p),3.),0.,1.),
        p=p*S+q,
        R=R*abs(S)+1.,
        p=p.yzx;
    return length(p)/R;
}


// by gaz
float fractal_de134(vec3 p){
    float i,g,e,R,S;
    vec3 q;
    q=p;
    R=1.;
    for(int j=0;j++<9;)
        p-=clamp(p,-1.,1.)*2.,S=6.*clamp(.2/min(dot(p,p),7.),0.,1.),
        p=p*S+q*.7,
        R=R*abs(S)+.7;
    return length(p)/R;
}


// by gaz
float fractal_de135(vec3 p){
    float i,g,e,R,S;
    vec3 q;
    p.z-=3.;
    q=p;
    R=1.;
    for(int j=0;j++<9;)
        p-=clamp(p,-.9,.9)*2.,
        S=9.*clamp(.1/min(dot(p,p),1.),0.,1.),
        p=p*S+q,
        R=R*S+1.;
    return .7*length(p)/R;
}


// by gaz
float fractal_de136(vec3 p){
    float i,g,e,R,S;
    vec3 q;
    p.z-=4.;
    q=p;
    R=1.;
    for(int j=0;j++<9;)
        p-=clamp(p,-1.,1.)*2.,
        S=9.*clamp(.3/min(dot(p,p),1.),0.,1.),
        p=p*S+q*.5,
        R=R*abs(S)+.5;
    return .6*length(p)/R-1e-3;
}


// by takusakuw
float fractal_de137(vec3 p){
    return length(sin(p)+cos(p*.5))-.4;
}


// by yosshin
float fractal_de138(vec3 p){
    return min(.65-length(fract(p+.5)-.5),p.y+.2);
}


// by takusakuw
float fractal_de139(vec3 p){
    return (length(sin(p.zxy)-cos(p.zzx))-.5);
}


// by yuruyurau
float fractal_de140(vec3 p){
#define b(p)length(max(abs(mod(p,.8)-.4)-.05,0.))
    vec3 l;
    p=cos(p)-vec3(.3), p.yx*=mat2(cos(.8+vec4(0,3,5,0)));
    return min(min(b(p.xy),b(p.xz)),b(p.yz));
#undef b
}


// by gaz
float fractal_de141(vec3 p){
    #define F1(a,n)a=abs(a)-n,a=vec2(a.x*.5+a.y,a.x-a.y*.5)
    p=fract(p)-.5;
    for(int j=0;j++<8;)
      F1(p.zy,.0),
      F1(p.xz,.55);
    return .4*length(p.yz)-2e-3;
    #undef F1
}


// by gaz
float fractal_de142(vec3 p){
#define M(a)mat2(cos(a+vec4(0,2,5,0)))
#define F1(a)for(int j=0;j<5;j++)p.a=abs(p.a*M(3.));(p.a).y-=3.
    float t = 0.96;
    p.z-=9.;
    p.xz*=M(t);
    F1(xy);
    F1(zy);
    return dot(abs(p),vec3(.3))-.5;
#undef M
#undef F1
}


// adapted from code by alia
float fractal_de143(vec3 p){
    vec3 q=fract(p)-.5;
    float f=-length(p.xy)+2., g=length(q)-.6;
    return max(f,-g);
}


// adapted from code by wrighter - aliasing issues
float fractal_de144(vec3 p){
   vec3 a = sin(p/dot(p,p)*4);
   return 0.95*min(length(a.yx),length(a.yz))-0.52+0.2;
}


// by phi16
float fractal_de145(vec3 p){ 
    return length(.05*cos(9.*p.y*p.x)+cos(p)-.1*cos(9.*(p.z+.3*p.x-p.y)))-1.; 
}


// by gaz
float fractal_de146(vec3 p){
    vec3 q=p;
    float s=5., e=0.;
    for(int j=0;j++<8;s*=e)
        p=sign(p)*(1.-abs(abs(p-2.)-1.)),
        p=p*(e=6./clamp(dot(p,p),.1,3.))-q*vec3(2,8,5);
    return length(p)/s;
}


// by gaz
float fractal_de147(vec3 p){
    float e=2., s=0., z=0.;
    // p.y+=sin(t*.1)*e;
    for(int j=0;++j<6;p=abs(p)-1.5,e/=s=min(dot(p,p),.75),p/=s);
    z+=length(p.xz)/e;
    return z;
}


// by yonatan
float fractal_de148(vec3 p){
    float i,j,e,g,h,s;
    p.y-=p.z*.5;
    for(j=s=h=.01;j++<9.;s+=s)
        p.xz*=rotate2D(2.),
        h+=abs(sin(p.x*s)*sin(p.z*s))/s;
    return max(0.,p.y+h);
}


// by yonatan
float fractal_de149(vec3 p){
    float i,g,e,s,q;
    q=length(p)-1.;
    p.y++;
    s=3.;
    for(int i=0;i++<7;p=vec3(0,5,0)-abs(abs(p)*e-3.))
        s*=e=max(1.,14./dot(p,p));
    return max(q,min(1.,length(p.xz)-.3))/s;
}



// by gaz
float fractal_de150(vec3 p){
    float s=2., e=0.;
    for(int i=0;i++<8;p=abs(p)*e)
        p=vec3(.8,2,1)-abs(p-vec3(1,2,1)),
        s*=e=1.3/clamp(dot(p,p),.1,1.2);
    return min(length(p.xz),p.y)/s+.001;
}



// by gaz
float fractal_de151(vec3 p){
    float i,g=.3,e,s=2.,q;
    for(int i=0;i++<7;p=vec3(2,5,1)-abs(abs(abs(p)*e-3.)-vec3(2,5,1)))
        s*=e=12./min(dot(p,p),12.);
    return min(1.,length(p.xz)-.2)/s;
}



// by kamoshika
float fractal_de152(vec3 p){
    vec3 Q;
    float i,d=1.,a,b=sqrt(3.);
    Q=mod(p,b*2.)-b;
    a=1.;
    d=9.;
    for(int j=0;j++<7;){
        Q=abs(Q);
        d=min(d,(dot(Q,vec3(1)/b)-1.)/a);
        Q=Q*3.-6./b;a*=3.;
    }
    return d;
}


// by kamoskika
float fractal_de153(vec3 p){
    float i,d=1.,b=1.73;
    vec3 Q=mod(p,b*2.)-b;
    for(int j=0;j++<6;){
        Q=abs(Q);
        if(Q.y>Q.x)Q.xy=Q.yx;
        if(Q.z>Q.x)Q.zx=Q.xz;
        Q*=2.;
        Q.x-=b;
    }
    return (dot(abs(Q),vec3(1)/b)-1.)/64.;
}


// by yonatan
float fractal_de154(vec3 p){
    return (length(vec2((length(vec2(length(p.xy)-1.3,length(p.zy)-1.3))-.5),dot(cos(p*12.),sin(p.zxy*12.))*.1))-.02)*.3;
}


// by gaz - hangs on desktop
float fractal_de155(vec3 p){
    float s=5., e=0.;
    for(int i=0;i++<5;)
        p=1.-abs(p),
        s*=e=1.3/min(dot(p,p),1.7),
        p*=e-.15;
    p=abs(p);
    p.x<p.z?p=p.zyx:p;
    p.y<p.z?p=p.xzy:p;
    return dot(p,vec3(1,1,-1))/s-.007;
}



// by gaz
float fractal_de156(vec3 p){
      p.yz*=rotate2D(-.3);
      float ss=3., s=1.;
      for(int j=0; j++<7;){
          p=abs(p);
          p.y-=.5;

          // change sphere hold
          s = 1./clamp(dot(p,p),.0,1.);
          p*=s;
          ss*=s;
          p-=vec2(1,.1).xxy;
          p.xyz=p.zxy;
      }
  
      // change SDF
      return length(p.xy)/ss-.01;
}



// by gaz
float fractal_de157(vec3 p){
    p.yz*=rotate2D(-.3);
    float ss=3., s=1.;
    for(int j=0; j++<7;){
        p=abs(p);
        p.y-=.5;
        s = 1./clamp(dot(p,p),.0,1.);
        p*=s;
        ss*=s;
        p-=vec2(1,.1).xxy;
        p.xyz=p.zxy;
    }
    return length(max(abs(p)-.6,0.))/ss-.01;
}



// by gaz
float fractal_de158(vec3 p){
    float s=2., e;
    for(int j=0;j++<8;){
        p=.1-abs(p-.2);
        p.x<p.z?p=p.zyx:p;
        s*=e=1.6;
        p=abs(p)*e-vec3(.1,3,1);
        p.yz*=rotate2D(.8);
    }
    return length(p.yx)/s-.04;
}



// by gaz
float fractal_de159(vec3 p){
    float s=2., e;
    for(int i=0;i++<8;){
        p=.5-abs(p);
        p.x<p.z?p=p.zyx:p;
        p.z<p.y?p=p.xzy:p;
        s*=e=1.6;
        p=abs(p)*e-vec3(.5,30,5);
        p.yz*=rotate2D(.3);
    }
    return length(p.xy)/s-.005;
}


// by gaz
float fractal_de160(vec3 p){
    float s=3.,e;
    for(int i=0;i++<3;p=vec3(2,4,2)-abs(abs(p)*e-vec3(3,6,1)))
        s*=e=1./min(dot(p,p),.6);
    return min(length(p.xz),abs(p.y))/s+.001;
}




// by gaz
float fractal_de161(vec3 p){
    float s=3., e;
    s*=e=3./min(dot(p,p),50.);
    p=abs(p)*e;
    for(int i=0;i++<5;)
        p=vec3(2,4,2)-abs(p-vec3(4,4,2)),
            s*=e=8./min(dot(p,p),9.),
            p=abs(p)*e;
    return min(length(p.xz)-.1,p.y)/s;
}




// by yonatan
float fractal_de162(vec3 p){
    float s=3., offset=8., e;
    for(int i=0;i++<9;p=vec3(2,4,2)-abs(abs(p)*e-vec3(4,4,2)))
        s*=e=max(1.,(8.+offset)/dot(p,p));
    return min(length(p.xz),p.y)/s;
}



// by gaz
float fractal_de163(vec3 p){
    p=sin(2.8*p+5.*sin(p*.3));
    float s=2., e;
    for(int i=0;i++<6;)
        p=abs(p-1.7)-1.5,
        s*=e=2.3/clamp(dot(p,p),.3,1.2),
        p=abs(p)*e;
    return length(p.zy)/s;
}



// by nameless
float fractal_de164(vec3 p0){
    p0=p0/10.;
    p0 = mod(p0, 2.)-1.;
    vec4 p = vec4(p0, 1.);
    escape = 0.;
    p=abs(p);
    if(p.x < p.z)p.xz = p.zx;
    if(p.z < p.y)p.zy = p.yz;
    if(p.y < p.x)p.yx = p.xy;
    for(int i = 0; i < 8; i++){
        if(p.x < p.z)p.xz = p.zx;
        if(p.z < p.y)p.zy = p.yz;
        if(p.y < p.x)p.yx = p.xy;
        
        p.xyz = abs(p.xyz);

        p*=(1.6/clamp(dot(p.xyz,p.xyz),0.6,1.));
        p.xyz-=vec3(0.7,1.8,0.5);
        p*=1.2;

        // escape += exp(-0.2*dot(p.xyz,p.xyz));
    }
    float m = 1.5;
    p.xyz-=clamp(p.xyz,-m,m);
    return (length(p.xyz)/p.w)*10.;
}



// by gaz
float fractal_de165(vec3 p){
    float s=5., e;
    p=p/dot(p,p)+1.;
    for(int i=0;i++<8;p*=e)
        p=1.-abs(p-1.),
        s*=e=1.6/min(dot(p,p),1.5);
    return length(cross(p,normalize(vec3(1))))/s-5e-4;
}


// by gaz
float fractal_de166(vec3 p){
    float s=3., e, offset = 1.; //offset can be adjusted 
    for(int i=0;i++<8;p*=e)
        p=abs(p-vec3(1,3,1.5+offset*.3))-vec3(1,3.+offset*.3,2),
        p*=-1.,
        s*=e=7./clamp(dot(p,p),.7,7.);
    return (p.z)/s+1e-3;
}


// by gaz
float fractal_de167(vec3 p){
    p=sin(p+3.*sin(p*.5));
    float s=2., e;
    for(int i=0;i++<5;)
        p=abs(p-1.7)-1.3,
        s*=e=2./min(dot(p,p),1.5),
        p=abs(p)*e-1.;
    return length(p)/s;
}


// by gaz
float fractal_de168(vec3 p){
#define M(p)p=vec2(sin(atan(p.x,p.y)*4.)/3.,1)*length(p),p.y-=2.
    float i,g,e,s;
    for(s=3.;s<4e4;s*=3.)
        M(p.xy),
        M(p.zy),
        p*=3.;
    return length(p.xy)/s-.001;
#undef M
}



// by gaz
float fractal_de169(vec3 p){
    p=1.-abs(abs(p+sin(p))-1.);
    p=p.x<p.y?p.zxy:p.zyx;
    float s=5., l;
    for(int j=0;j++<4;)
        s*=l=2./min(dot(p,p),1.5),
        p=abs(p)*l-vec3(2,1,3);
    return length(p.yz)/s;
}


// by gaz
float fractal_de170(vec3 p){
    float s=3., e;
    for(int j=0;++j<5;)
        s*=e=1./min(dot(p,p),1.),
        p=abs(p)*e-1.5;
    return length(p.yz)/s;
}



// by gaz - pillar
float fractal_de171(vec3 p){
    float s=2., e;
    for(int j=0;j++<8;)
        s*=e=2./clamp(dot(p,p),.2,1.),
        p=abs(p)*e-vec3(.5,8,.5);
    return length(cross(p,vec3(1,1,-1)))/s;
}



// by yonatan
float fractal_de172(vec3 p){
    p.xz=mod(p.xz,2.)-1.;
    vec3 q=p;
    float s=2., e;
    for(int j=0;j++<8;)
        s*=e=2./clamp(dot(p,p),.5,1.),
        p=abs(p)*e-vec3(.5,8,.5);
    return max(q.y,length(p.xz)/s);
}



// by gaz
float fractal_de173(vec3 p){
    p.xz=abs(p.xz)-1.;
    p.x>p.z?p=p.zyx:p;
    float s=2., e;
    for(int j=0;j++<7;)
        s*=e=2.2/clamp(dot(p,p),.3,1.2),
        p=abs(p)*e-vec3(1,8,.03);
    return length(p.yz)/s;
}



// by gaz
float fractal_de174(vec3 p){
    // Enclose with this fold
    //p.xz=abs(p.xz)-1.;p.x>p.z?p=p.zyx:p;
    float s=2., e;
    for(int j=0;j++<7;)
        s*=e=2.2/clamp(dot(p,p),.3,1.2),
        // Eliminate the thickness with this fold offset
        //p=abs(p)*e-vec3(1,8,.03);
        p=abs(p)*e-vec3(1,8,1);
    // Changed sdf to make it easier to understand
    return length(cross(p,vec3(1,1,-1)))/s;
}



// by gaz
float fractal_de175(vec3 p){
    p.xz=mod(p.xz,2.)-1.;
    float s=2., e;
    for(int j=0;j++<8;)
        s*=e=2./clamp(dot(p,p),.5,1.),
        p=abs(p)*e-vec3(.5,8,.5);
    return length(p.xz)/s;
}


// by gaz
float fractal_de176(vec3 p){
    float s=2.,e;
    for(int i=0;i<9;i++){
        p=.5-abs(p-.5);
        p.x<p.z?p=p.zyx:p;
        p.z<p.y?p=p.xzy:p;
        s*=e=2.4;
        p=abs(p)*e-vec3(.1,13,5);
    }
    return length(p)/s-0.01;
}


// by gaz
float fractal_de177(vec3 p){
    float s=2., e;
    for(int i=0;i++<7;){
        p.xz=.8-abs(p.xz);
        p.x<p.z?p=p.zyx:p;
        s*=e=2.1/min(dot(p,p),1.);
        p=abs(p)*e-vec3(1,18,9);
    }
    return length(p)/s-0.01;
}



// by kamoshika
float fractal_de178(vec3 p){
    float e, offset = 1.;
    e=length(max(abs(p)-.5,0.))-.1;
    return max(abs(e)-.05,sin((acos(p.y/length(p))*5.+sign(p.z)*acos(p.x/length(p.zx))+offset*3.)*3.)*.01);
}


// by yonatan
float fractal_de179(vec3 p){
    float n=1.+snoise3D(p), s=4., e;
    for(int i=0;i++<7;p.y-=20.*n)
        p.xz=.8-abs(p.xz),
        p.x<p.z?p=p.zyx:p,
        s*=e=2.1/min(dot(p,p),1.),
        p=abs(p)*e-n;
    return length(p)/s+1e-4;
}


// by gaz
float fractal_de180(vec3 p){
    float s=3., e;
    for(int i=0;i++<8;)
        p=mod(p-1.,2.)-1.,
        s*=e=1.4/dot(p,p),
        p*=e;
    return length(p.yz)/s;
}


// by gaz
float fractal_de181(vec3 p){
    float s=4., e;
    for(int i=0;i++<7;p.y-=10.)
        p.xz=.8-abs(p.xz),
        p.x<p.z?p=p.zyx:p,
        s*=e=2.5/clamp(dot(p,p),.1,1.2),
        p=abs(p)*e-1.;
    return length(p)/s+.001;
}


// by gaz
float fractal_de182(vec3 p){
    float s=2., e;
    for(int i=0;i++<10;){
        p=.3-abs(p-.8);
        p.x<p.z?p=p.zyx:p;
        p.z<p.y?p=p.xzy:p;
        s*=e=1.7;
        p=abs(p)*e-vec3(1,50,5);
    }
    return length(p.xy)/s+.001;
}


// by gaz
float fractal_de183(vec3 p){
    float s=1., e, offset=0.26; // vary between 0 and 1
    for(int i=0;i++<5;){
        s*=e=2./min(dot(p,p),1.);
        p=abs(p)*e-vec3(1,10.*offset,1);
    }
    return length(max(abs(p)-1.,0.))/s;
}


// by gaz - takes a second to compile but does work
float fractal_de184(vec3 p){
    float s=2.5, e;
    p=abs(mod(p-1.,2.)-1.)-1.;
    for(int j=0;j++<10;)
        p=1.-abs(p-1.),
        s*=e=-1.8/dot(p,p),
        p=p*e-.7;
    return abs(p.z)/s+.001;
}



// by gaz
float fractal_de185(vec3 p){
    float s=2., e;
    for(int j=0;++j<8;s*=e=2./clamp(dot(p,p),.4,1.),p=abs(p)*e-vec3(2,1,.7));
    return length(p)/s;
}


// by gaz
float fractal_de186(vec3 p){
    float s=2., e;
    for(int j=0;++j<8;s*=e=2./clamp(dot(p,p),.4,1.),p=abs(p)*e-vec3(2,1,.7));
    return length(p-clamp(p,-2.,2.))/s;
}


// by gaz
float fractal_de187(vec3 p){
    float s=2., e;
    for(int j=0;++j<18;s*=e=2./clamp(dot(p,p),.4,1.),p=abs(p)*e-vec3(2,1,.7));
    return length(p)/s;
}


// by amini
float fractal_de188(vec3 p){
    float s=3., e;
    s*=e=3./min(dot(p,p),50.);
    p=abs(p)*e;
    for(int i=0;i++<5;)
        p=vec3(8,4,2)-abs(p-vec3(8,4,2)),
        s*=e=8./min(dot(p,p),9.),
        p=abs(p)*e;
    return min(length(p.xz)-.1,p.y)/s;
}


// adapted from code by sdfgeoff
float sdRoundBox189( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float df189(vec3 co) {
    float rad = clamp(co.z * 0.05 + 0.45, 0.1, 0.3);
    co = mod(co, vec3(1.0)) - 0.5;
    return sdRoundBox189(co, vec3(rad, rad, 0.3), 0.1);
}

float fractal_de189(vec3 p){

    float body = 999.0;
    float scale = 0.2;
    vec3 co = p;
    mat4 m = mat4(
		vec4(0.6373087, -0.0796581,  0.7664804, 0.0),
  		vec4(0.2670984,  0.9558195, -0.1227499, 0.0),
  		vec4(-0.7228389,  0.2829553,  0.6304286, 0.0),
        vec4(0.1, 0.6, 0.2, 0.0));
    
    for (int i=0; i<3; i++) {
        co = (m * vec4(co, float(i))).xyz;
        scale *= (3.0);
        
        float field = df189(co * scale) / scale;
        body = smin_op(body, field, 0.05);
    }

    return -body;
}


// by kali (xor3d)
float fractal_de190(vec3 p){
    float f = 1.;
    p=abs(p);
    float d=100.;
    ivec3 ip=ivec3(p*d);
    float c = float(ip.x^ip.y^ip.z)/d;
    return step(f,c);
}


// by kali
float fractal_de191(vec3 p){
    p.x = abs(p.x) - 3.3;
    p.z = mod(p.z + 2.0, 4.0) -  2.0;
    vec4 q = vec4(p, 1);
    q.xyz -= 1.0;

    q.xyz = q.zxy;
    for(int i = 0; i < 6; i++) {
        q.xyz = abs(q.xyz + 1.0) - 1.0;
        q /= clamp(dot(q.xyz, q.xyz), 0.25, 1.0);
        q *= 1.1;

        float s = sin(-0.35);
        float c = cos(-0.35);
        
        q.xy = mat2(c,s,-s,c)*q.xy;
    }
    return (length(q.xyz) - 1.5)/q.w;
}



// by kali
float fractal_de192(vec3 p){
    p.xz=abs(.5-mod(p.xz,1.))+.01;
    float DEfactor=1.;
    for (int i=0; i<14; i++) {
        p = abs(p)-vec3(0.,2.,0.);  
        float r2 = dot(p, p);
        float sc=2./clamp(r2,0.4,1.);
        p*=sc; 
        DEfactor*=sc;
        p = p - vec3(0.5,1.,0.5);
    }
    return length(p)/DEfactor-.0005;
}


// by kali
float fractal_de193(vec3 pos){
    vec3 tpos=pos;
    tpos.xz=abs(.5-mod(tpos.xz,1.));
    vec4 p=vec4(tpos,1.);
    float y=max(0.,.35-abs(pos.y-3.35))/.35;
    for (int i=0; i<7; i++) {//LOWERED THE ITERS
        p.xyz = abs(p.xyz)-vec3(-0.02,1.98,-0.02);
        p=p*(2.0+0.*y)/clamp(dot(p.xyz,p.xyz),.4,1.)-vec4(0.5,1.,0.4,0.);
        p.xz*=mat2(-0.416,-0.91,0.91,-0.416);
    }
    return (length(max(abs(p.xyz)-vec3(0.1,5.0,0.1),vec3(0.0)))-0.05)/p.w;
}



// by Shane
float fractal_de194(vec3 p){
    // I'm never sure whether I should take constant stuff like the following outside the function, 
    // or not. My 1990s CPU brain tells me outside, but it doesn't seem to make a difference to frame 
    // rate in this environment one way or the other, so I'll keep it where it looks tidy. If a GPU
    // architecture\compiler expert is out there, feel free to let me know.
    
    const vec3 offs = vec3(1, .75, .5); // Offset point.
    const vec2 a = sin(vec2(0, 1.57079632) + 1.57/2.);
    const mat2 m = mat2(a.y, -a.x, a);
    const vec2 a2 = sin(vec2(0, 1.57079632) + 1.57/4.);
    const mat2 m2 = mat2(a2.y, -a2.x, a2);
    
    const float s = 5.; // Scale factor.
    float d = 1e5; // Distance.
    
    p  = abs(fract(p*.5)*2. - 1.); // Standard spacial repetition.
     
    float amp = 1./s; // Analogous to layer amplitude.
    
    // With only two iterations, you could unroll this for more speed,
    // but I'm leaving it this way for anyone who wants to try more
    // iterations.
    for(int i=0; i<2; i++){
        // Rotating.
        p.xy = m*p.xy;
        p.yz = m2*p.yz;
        
        p = abs(p);
        //p = sqrt(p*p + .03);
        //p = smin(p, -p, -.5); // Etc.
        
        // Folding about tetrahedral planes of symmetry... I think, or is it octahedral? 
        // I should know this stuff, but topology was many years ago for me. In fact, 
        // everything was years ago. :)
        // Branchless equivalent to: if (p.x<p.y) p.xy = p.yx;
        p.xy += step(p.x, p.y)*(p.yx - p.xy);
        p.xz += step(p.x, p.z)*(p.zx - p.xz);
        p.yz += step(p.y, p.z)*(p.zy - p.yz);
 
        // Stretching about an offset.
        p = p*s + offs*(1. - s);
        
        // Branchless equivalent to:
        // if( p.z < offs.z*(1. - s)*.5)  p.z -= offs.z*(1. - s);
        p.z -= step(p.z, offs.z*(1. - s)*.5)*offs.z*(1. - s);
        
        // Loosely speaking, construct an object, and combine it with
        // the object from the previous iteration. The object and
        // comparison are a cube and minimum, but all kinds of 
        // combinations are possible.
        p=abs(p);
        d = min(d, max(max(p.x, p.y), p.z)*amp);
        
        amp /= s; // Decrease the amplitude by the scaling factor.
    }
    return d - .035; // .35 is analous to the object size.
}


// by avi
float fractal_de195(vec3 p) {
    const vec3 va = vec3(  0.0,  0.57735,  0.0 );
    const vec3 vb = vec3(  0.0, -1.0,  1.15470 );
    const vec3 vc = vec3(  1.0, -1.0, -0.57735 );
    const vec3 vd = vec3( -1.0, -1.0, -0.57735 );

    float a = 0.0;
    float s = 1.0;
    float r = 1.0;
    float dm;
    vec3 v;
    for(int i=0; i<16; i++) {
        float d, t;
        d = dot(p-va,p-va);              v=va; dm=d; t=0.0;
        d = dot(p-vb,p-vb); if( d<dm ) { v=vb; dm=d; t=1.0; }
        d = dot(p-vc,p-vc); if( d<dm ) { v=vc; dm=d; t=2.0; }
        d = dot(p-vd,p-vd); if( d<dm ) { v=vd; dm=d; t=3.0; }
        p = v + 2.0*(p - v); r*= 2.0;
        a = t + 4.0*a; s*= 4.0;
    }
    
    return (sqrt(dm)-1.0)/r;
}




// by guil
vec3 foldY196(vec3 P, float c)
{
	float r = length(P.xz);
	float a = atan(P.z, P.x);

	a = mod(a, 2.0 * c) - c; 

	P.x = r * cos(a);
	P.z = r * sin(a);

	return P;
}
float fractal_de196(vec3 p)
{ 
    float l= length(p)-1.;
    float dr = 1.0, g = 1.25;
    vec4 ot=vec4(.3,.5,0.21,1.);
    ot = vec4(1.);
    mat3 tr = rotate3D(-0.55, normalize(vec3(-1., -1., -0.5)));
				
	for(int i=0;i<15;i++) {

		if(i-(i/3)*5==0)
			p = foldY196(p, .95);
		p.yz = abs(p.yz);				
        p = tr * p * g -1.;		
		dr *= g;
		ot=min(ot,vec4(abs(p),dot(p,p)));
        l = min (l ,(length(p)-1.) / dr);
	}
			
    return l;    
}


// by marvelousbilly
mat3 rotmat197(float angle, vec3 axis){
	axis = normalize(axis);
	float s = sin(angle);
	float c = cos(angle);
	float oc = 1.0 - c;
	return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s, 
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c         );
}
float fractal_de197(vec3 p){    
    mat3 r = rotmat197(3.14159, vec3(0.,1.,0.)); //rotation matrix
    float scale= 2.;
    int Iterations = 10;
    int i;
    vec3 C = vec3(1.,5.4,10.+10.*sin(0.5));
    for(i = 0; i < Iterations; i++){ 
        p = r * (p);
        float x = p.x; float y = p.y; float z = p.z; float x1 = x; float y1 = y;

        x=abs(x);y = abs(y);
        if(x-y<0.){x1=y;y=x;x=x1;}
        if(x-z<0.){x1=z;z=x;x=x1;}
        if(y-z<0.){y1=z;z=y;y=y1;}

        z-=0.5*C.z*(scale-1.)/scale;
        z=-abs(-z);
        z+=0.5*C.z*(scale-1.)/scale;
        
        p = vec3(x,y,z);
        r = rotmat197(31.4159/4.+5.60,vec3(1.,0.5,0.6));
        p = r * (p);
        x = p.x; y = p.y; z = p.z;
        
        x=scale*x-C.y*(scale-1.);
        y=scale*y-C.y*(scale-1.);
        z=scale*z;

        p = vec3(x,y,z);
    }
    return (length(p) - 2.) * pow(scale,float(-i)); 
}




// adapted from above
float fractal_de198(vec3 p){    
    mat3 r = rotate3D(3.14159, vec3(0.,1.,0.)); //rotation matrix
    float scale= 2.;
    int Iterations = 10;
    int i;
    vec3 C = vec3(1.,5.4,10.+10.*sin(0.5));
    for(i = 0; i < Iterations; i++){ 
        p = r * (p);
        float x = p.x; float y = p.y; float z = p.z; float x1 = x; float y1 = y;

        x=abs(x);y = abs(y);
        if(x-y<0.){x1=y;y=x;x=x1;}
        if(x-z<0.){x1=z;z=x;x=x1;}
        if(y-z<0.){y1=z;z=y;y=y1;}

        z-=0.5*C.z*(scale-1.)/scale;
        z=-abs(-z);
        z+=0.5*C.z*(scale-1.)/scale;
        
        p = vec3(x,y,z);
        r = rotate3D(31.4159/4.+5.60,vec3(1.,0.5,0.6));
        p = r * (p);
        x = p.x; y = p.y; z = p.z;
        
        
        x=scale*x-C.y*(scale-1.);
        y=scale*y-C.y*(scale-1.);
        z=scale*z;

        p = vec3(x,y,z);
    }
    return (length(p) - 2.) * pow(scale,float(-i)); 
}



// by nameless
float fractal_de199(vec3 p0){
    vec4 p = vec4(p0, 1.);
        p.xyz=abs(p.xyz);
        if(p.x > p.z)p.xz = p.zx;
        if(p.z < p.y)p.zy = p.yz;
        if(p.y > p.x)p.yx = p.xy;
    for(int i = 0; i < 8; i++){
        if(p.x > p.z)p.xz = p.zx;
        if(p.z < p.y)p.zy = p.yz;
        if(p.y > p.x)p.yx = p.xy;
        
        p.xyz = abs(p.xyz);

        p*=(2.15/clamp(dot(p.xyz,p.xyz),.4,1.));
        p.xyz-=vec3(0.3,0.2,1.6);

    }
    float m = 1.5;
    p.xyz-=clamp(p.xyz,-m,m);
    return length(p.xyz)/p.w;
}



void ry200(inout vec3 p, float a){  
    float c,s;vec3 q=p;  
    c = cos(a); s = sin(a);  
    p.x = c * q.x + s * q.z;  
    p.z = -s * q.x + c * q.z; 
}  
float plane200(vec3 p, float y) {
    return length(vec3(p.x, y, p.z) - p);
}
float menger_spone200(in vec3 z0){
    z0=z0.yzx;
    vec4 z=vec4(z0,1.0);
    vec3 offset =0.83*normalize(vec3(3.4,2., .2));
    float scale = 2.;
    for (int n = 0; n < 8; n++) {
        z = abs(z);
        ry200(z.xyz, 1.5);
        if (z.x<z.y)z.xy = z.yx;
        if (z.x<z.z)z.xz = z.zx;
        if (z.y<z.z)z.yz = z.zy;
        ry200(z.xyz, -1.21);
        z = z*scale;
        z.xyz -= offset*(scale-1.0);
    }
    return (length(max(abs(z.xyz)-vec3(1.0),0.0))-0.01)/z.w;
}
float fractal_de200(vec3 p){ 
    float d1 = plane200(p, -0.5);
    float d2 = menger_spone200(p+vec3(0.,-0.1,0.));
    float d = d1;
    vec3 res = vec3(d1, 0., 0.);
    if(d > d2){
        d = d2;
        res = vec3(d2, 1., 0.0);
    }
    return res.x;
} 



// by plento
float sdBox201( vec3 p, vec3 b ){
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

vec2 rotate201(vec2 k,float t){
  return vec2(cos(t) * k.x - sin(t) * k.y, sin(t) * k.x + cos(t) * k.y);
}

float fractal_de201(vec3 pos)
{
    vec3 b = vec3(0.9 , 4.5, 0.70);
    float p = sin(pos.z * 0.1) * 2.0;
  
    pos = vec3(rotate201(pos.xy, p), pos.z);
    
    pos.y += 1.2;
    pos = mod(pos, b) -0.5 * b;
    
    pos.x *= sin(length(pos * 1.8) * 2.0) * 1.4;
    
    float boxScale = 0.4;
    
    return sdBox201(pos - vec3(0.0, 0.0, 0.0), vec3(boxScale));
}




// dr2
float sdGyroidTorus (vec3 q, float rt, float rg, float ws) {
    // workable args
    // float rt = 15.;
    // float rg = 4.;
    // float ws = 0.3;
  q.xz = vec2 (rt * atan (q.z, - q.x), length (q.xz) - rt);
  q.yz = vec2 (rg * atan (q.z, - q.y), length (q.yz) - rg);
  return .6* max(abs(dot(sin(q), cos(q).yzx)) - ws, abs(q.z) - .5*PI);
}

// FabriceNeyret2
float sdGyroid(vec3 p, float scale, float thickness, float bias) {
  p *= scale;
  return (abs(dot(sin(p*.5), cos(p.zxy * 1.23)) - bias) / scale - thickness)*0.55;
}

// by iq
float sdEllipsoid( in vec3 p ) {
    vec3 r = vec3(0.2, 0.25, 0.05); // the radii on each axis
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

// by iq
float sdBoundingBox( vec3 p){
    float e = 0.05;
    vec3 b = vec3(.3,.5,.4);
       p = abs(p  )-b;
  vec3 q = abs(p+e)-e;

  return min(min(
      length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

// iq
float sdCappedCone(vec3 p){
    float h = 1.;
    float r1 = 0.5;
    float r2 = 0.2;
    
    vec2 q = vec2( length(p.xz), p.y );
    
    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot(k2,k2), 0.0, 1.0 );
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot(ca,ca),dot(cb,cb)) );
}


// // iq
// float sdCappedCone(vec3 p){
//     vec3 a = vec3(0,0,0); // point a
//     vec3 b = vec3(0,1,0); // point b
//     float ra = .5; // radius at a
//     float rb = .2; // radius at b
    
//     float rba  = rb-ra;
//     float baba = dot(b-a,b-a);
//     float papa = dot(p-a,p-a);
//     float paba = dot(p-a,b-a)/baba;

//     float x = sqrt( papa - paba*paba*baba );

//     float cax = max(0.0,x-((paba<0.5)?ra:rb));
//     float cay = abs(paba-0.5)-0.5;

//     float k = rba*rba + baba;
//     float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );

//     float cbx = x-ra - f*rba;
//     float cby = paba - f;
    
//     float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    
//     return s*sqrt( min(cax*cax + cay*cay*baba,
//                        cbx*cbx + cby*cby*baba) );
// }

// iq
float sdSolidAngle(vec3 p){
    float angle = 1+0.2*sin(time); // desired cone angle
    
    vec2 c = vec2(cos(angle), sin(angle));
    float ra = 1.; // radius of the sphere from which it is cut
    
    vec2 p0 = vec2( length(p.xz), p.y );
    float l = length(p0) - ra;
    float m = length(p0 - c*clamp(dot(p0,c),0.0,ra) );
    return max(l,m*sign(c.y*p0.x-c.x*p0.y));
}

//iq
float sdCappedTorus(vec3 p){
    float angle = 2.0; // angle spanned 
    float ra = 0.25; // major radius
    float rb = 0.05; // minor radius
    
    vec2 sc = vec2(sin(angle), cos(angle));
    p.x = abs(p.x);
    float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

// iq
float sdPyramid(vec3 p){
    float h = 1.;
    float m2 = h*h + 0.25;
    
    // symmetry
    p.xz = abs(p.xz);
    p.xz = (p.z>p.x) ? p.zx : p.xz;
    p.xz -= 0.5;

    // project into face plane (2D)
    vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);
   
    float s = max(-q.x,0.0);
    float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );
    
    float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
    float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
    
    float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
    
    // recover 3D and scale, and add sign
    return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));;
}

// iq
float sdTriPrism( vec3 p ){
    vec2 h = vec2(0.5, 0.2); // height, thickness
    const float k = sqrt(3.0);
    h.x *= 0.5*k;
    p.xy /= h.x;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) p.xy=vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0, 0.0 );
    float d1 = length(p.xy)*sign(-p.y)*h.x;
    float d2 = abs(p.z)-h.y;
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}


// la,lb=semi axis, h=height, ra=corner
float sdRhombus(vec3 p){
    float la = 0.15; // first axis
    float lb = 0.25; // second axis
    float h  = 0.04; // thickness
    float ra = 0.08; // corner radius

    p = abs(p);
    vec2 b = vec2(la,lb);
    vec2 bb = b-2.0*p.xz;
    
    float f = clamp((b.x*bb.x-b.y*bb.y)/dot(b,b), -1.0, 1.0 );
	vec2 q = vec2(length(p.xz-0.5*b*vec2(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
    return min(max(q.x,q.y),0.0) + length(max(q,0.0));
}


// float sdRoundCone( in vec3 p, in float r1, float r2, float h )
// {
//     vec2 q = vec2( length(p.xz), p.y );
    
//     float b = (r1-r2)/h;
//     float a = sqrt(1.0-b*b);
//     float k = dot(q,vec2(-b,a));
    
//     if( k < 0.0 ) return length(q) - r1;
//     if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
//     return dot(q, vec2(a,b) ) - r1;
// }

float sdRoundCone(vec3 p){
    vec3 a = vec3(0,0,0);
    vec3 b = vec3(0,3,0);
    float r1 = 1.0;
    float r2 = 0.1;
    
    vec3  ba = b - a;
    float l2 = dot(ba,ba);
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;
    float il2 = 1.0/l2;
    
    vec3 pa = p - a;
    float y = dot(pa,ba);
    float z = y - l2;
    vec3 d2 =  pa*l2 - ba*y;
    float x2 = dot(d2, d2);
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)        *il2 - r1;
                            return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}


float sdOctagonPrism(vec3 p){
  float r = 2.;
  float h = 0.2;

  const vec3 k = vec3(-0.9238795325,   // sqrt(2+sqrt(2))/2 
                       0.3826834323,   // sqrt(2-sqrt(2))/2
                       0.4142135623 ); // sqrt(2)-1 
  // reflections
  p = abs(p);
  p.xy -= 2.0*min(dot(vec2( k.x,k.y),p.xy),0.0)*vec2( k.x,k.y);
  p.xy -= 2.0*min(dot(vec2(-k.x,k.y),p.xy),0.0)*vec2(-k.x,k.y);
  // polygon side
  p.xy -= vec2(clamp(p.x, -k.z*r, k.z*r), r);
  vec2 d = vec2( length(p.xy)*sign(p.y), p.z-h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdLink(vec3 p){
    float le = 0.13;  // length
    float r1 = 0.20;  // major radius
    float r2 = 0.09;  // minor radius
    
    vec3 q = vec3( p.x, max(abs(p.y)-le,0.0), p.z );
    return length(vec2(length(q.xy)-r1,q.z)) - r2;
}

float udRoundBox(vec3 p){
    vec3 b = vec3(1,2,3); // box dimensions
    float r = 0.1;      // rounding radius
    return length(max(abs(p)-b, 0.0))-r;
}

float sdCross(vec3 p){
  float s = 0.2;
  float da = max (abs(p.x), abs(p.y));
  float db = max (abs(p.y), abs(p.z));
  float dc = max (abs(p.z), abs(p.x));
  return min(da,min(db,dc)) - s;
}

float sdWaveSphere(vec3 p){
    float radius = .3; // radius of sphere
    int waves = 7; // number of waves
    float waveSize = 0.4; // displacement of waves
    
    //bounding Sphere
    float d = length(p) - radius*2.2;
    if(d > 0.0) return 0.2;

    // deformation of radius
    d = waveSize * (radius*radius-(p.y*p.y));
    radius += d * cos(atan(p.x,p.z) * float(waves));
    return 0.5*(length(p) - radius);
}

// Dodecahedron: radius = circumsphere radius
float sdDodecahedron(vec3 p, float radius)
{
  const float phi = 1.61803398875;  // Golden Ratio = (sqrt(5)+1)/2;
  const vec3 n = normalize(vec3(phi,1,0));

  p = abs(p / radius);
  float a = dot(p, n.xyz);
  float b = dot(p, n.zxy);
  float c = dot(p, n.yzx);
  return (max(max(a,b),c)-n.x) * radius;
}

// Icosahedron: radius = circumsphere radius
float sdIcosahedron(vec3 p, float radius)
{
  const float q = 2.61803398875;  // Golden Ratio + 1 = (sqrt(5)+3)/2;
  const vec3 n1 = normalize(vec3(q,1,0));
  const vec3 n2 = vec3(0.57735026919);  // = sqrt(3)/3);

  p = abs(p / radius);
  float a = dot(p, n1.xyz);
  float b = dot(p, n1.zxy);
  float c = dot(p, n1.yzx);
  float d = dot(p, n2) - n1.x;
  return max(max(max(a,b),c)-n1.x,d) * radius;
}


float sdIcosDodecaStar(vec3 p, float radius)
{
  return min(sdDodecahedron(p,radius),  sdIcosahedron(p.zyx,radius));
}

float sdRoundedCylinder( vec3 p){
  float ra = 0.5;  // radius of cylinder
  float rb = 0.1;  // radius of rounding
  float h  = 0.4;  // height of cylinder

  vec2 d = vec2( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0)) - rb;
}

float length2( vec2 p )  // sqrt(x^2+y^2)
{
  return sqrt( p.x*p.x + p.y*p.y );
}

float length6( vec2 p )  // (x^6+y^6)^(1/6)
{
  p = p*p*p;
  p = p*p;
  return pow( p.x + p.y, 1.0/6.0 );
}

float length8( vec2 p )  // (x^8+y^8)^(1/8)
{
  p = p*p;
  p = p*p;
  p = p*p;
  return pow( p.x + p.y, 1.0/8.0 );
}
// Cylinder: h=dimension, h.y=height
float sdCylinder (in vec3 p, in vec3 h)
{
  return length(p.xz - h.xy) - h.z;
}
#define opRepeat(p,c) (mod(p,c)-0.5*c)
#define opDifference(a,b) max(a,-b)
float sdTorus82( vec3 p)
{
  vec2 t = vec2(1,0.2);
  vec2 q = vec2(length8(p.xz)-t.x, p.y);
  return length8(q) - t.y;
}


// float sdRackWheel(vec3 pos)
// {
//   return opDifference(sdTorus82(pos, vec2(0.20, 0.1)),
//     sdCylinder (opRepeat (vec3 (atan(pos.x, pos.z)/6.2831
//                                 ,pos.y
//                                 ,0.02+0.5*length(pos))
//                           ,vec3(0.05, 1.0, 0.05))
//                 ,vec2(0.02, 0.6)));
// }



float dot2( in vec3 v ) { return dot(v,v); }

float udTriangle( in vec3 v1, in vec3 v2, in vec3 v3, in vec3 p )
{
    vec3 v21 = v2 - v1; vec3 p1 = p - v1;
    vec3 v32 = v3 - v2; vec3 p2 = p - v2;
    vec3 v13 = v1 - v3; vec3 p3 = p - v3;
    vec3 nor = cross( v21, v13 );

    return sqrt( (sign(dot(cross(v21,nor),p1)) + 
                  sign(dot(cross(v32,nor),p2)) + 
                  sign(dot(cross(v13,nor),p3))<2.0) 
                  ?
                  min( min( 
                  dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0,1.0)-p1), 
                  dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0,1.0)-p2) ), 
                  dot2(v13*clamp(dot(v13,p3)/dot2(v13),0.0,1.0)-p3) )
                  :
                  dot(nor,p1)*dot(nor,p1)/dot2(nor) );
}

float udQuad( in vec3 v1, in vec3 v2, in vec3 v3, in vec3 v4, in vec3 p )
{
    #if 1
    // handle ill formed quads
    if( dot( cross( v2-v1, v4-v1 ), cross( v4-v3, v2-v3 )) < 0.0 )
    {
        vec3 tmp = v3;
        v3 = v4;
        v4 = tmp;
    }
    #endif

    
    vec3 v21 = v2 - v1; vec3 p1 = p - v1;
    vec3 v32 = v3 - v2; vec3 p2 = p - v2;
    vec3 v43 = v4 - v3; vec3 p3 = p - v3;
    vec3 v14 = v1 - v4; vec3 p4 = p - v4;
    vec3 nor = cross( v21, v14 );

    return sqrt( (sign(dot(cross(v21,nor),p1)) + 
                  sign(dot(cross(v32,nor),p2)) + 
                  sign(dot(cross(v43,nor),p3)) + 
                  sign(dot(cross(v14,nor),p4))<3.0) 
                  ?
                  min( min( dot2(v21*clamp(dot(v21,p1)/dot2(v21),0.0,1.0)-p1), 
                            dot2(v32*clamp(dot(v32,p2)/dot2(v32),0.0,1.0)-p2) ), 
                       min( dot2(v43*clamp(dot(v43,p3)/dot2(v43),0.0,1.0)-p3),
                            dot2(v14*clamp(dot(v14,p4)/dot2(v14),0.0,1.0)-p4) ))
                  :
                  dot(nor,p1)*dot(nor,p1)/dot2(nor) );
}


float sdCylinder6 (vec3 p){
    float diameter = 0.2;
    float height = 0.1;
  return max( length6(p.xz) - diameter, abs(p.y) - height );
}

float sdCylinder(vec3 p){
  float radius = 1.;
  return length(p.xz)-radius;
}







float mandelbulb202(vec3 p)
{
    p /= 1.192;
    p.xyz = p.xzy;
    vec3 z = p;
    vec3 dz = vec3(0.0);
    float dr = 1.0;
    float power = 8.0;
    float r, theta, phi;
    for (int i = 0; i < 7; i++)
    {
        r = length(z);
        if (r > 2.0)
            break;
        float theta = atan(z.y / z.x);
        float phi = asin(z.z / r);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        r = pow(r, power);
        theta = theta * power;
        phi = phi * power;
        z = r * vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), sin(phi)) + p;
    }
    return 0.5 * log(r) * r / dr;
}

float sdSponge202(vec3 z)
{
    for(int i = 0; i < 9; i++)
    {
        z = abs(z);
        z.xy = (z.x < z.y) ? z.yx : z.xy;
        z.xz = (z.x < z.z) ? z.zx : z.xz;
        z.zy = (z.y < z.z) ? z.yz : z.zy;	 
        z = z * 3.0 - 2.0;
        z.z += (z.z < -1.0) ? 2.0 : 0.0;
    }
    z = abs(z) - vec3(1.0);
    float dis = min(max(z.x, max(z.y, z.z)), 0.0) + length(max(z, 0.0)); 
    return dis * 0.6 * pow(3.0, -float(9)); 
}

float fractal_de202(vec3 p)
{
    float d1 = mandelbulb202(p);
    float d2 = sdSponge202(p);
    return max(d1, d2);
    
}




float fractal_de203( vec3 p ){
    vec3  di = abs(p) - vec3(1.);
    float mc = max(di.x, max(di.y, di.z));
    float d =  min(mc,length(max(di,0.0)));
    vec4 res = vec4( d, 1.0, 0.0, 0.0 );

    const mat3 ma = mat3( 0.60, 0.00,  0.80,
                          0.00, 1.00,  0.00,
                          -0.20, 0.00,  0.30 );
    float off = 0.0005;
    float s = 1.0;
    for( int m=0; m<4; m++ ){
        p = ma*(p+off);
        vec3 a = mod( p*s, 2.0 )-1.0;
        s *= 3.0;
        vec3 r = abs(1.0 - 3.0*abs(a));
        float da = max(r.x,r.y);
        float db = max(r.y,r.z);
        float dc = max(r.z,r.x);
        float c = (min(da,min(db,dc))-1.0)/s;
        if( c > d )
            d = c;
    }
    return d;
}


float fractal_de204(in vec3 z0){
    const float mr=0.25, mxr=1.0;
    const vec4 scale=vec4(-3.12,-3.12,-3.12,3.12),p0=vec4(0.0,1.59,-1.0,0.0);
    vec4 z = vec4(z0,1.0);
    for (int n = 0; n < 3; n++) {
        z.xyz=clamp(z.xyz, -0.94, 0.94)*2.0-z.xyz;
        z*=scale/clamp(dot(z.xyz,z.xyz),mr,mxr);
        z+=p0;
    }
    z.y-=3.0*sin(3.0+floor(z0.x+0.5)+floor(z0.z+0.5));
    float dS=(length(max(abs(z.xyz)-vec3(1.2,49.0,1.4),0.0))-0.06)/z.w;
    return dS;
}






float sdHexPrism205( vec3 p, vec2 h ){
  const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
  p = abs(p);
  p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
  vec2 d = vec2(
       length(p.xy-vec2(clamp(p.x,-k.z*h.x,k.z*h.x), h.x))*sign(p.y-h.x),
       p.z-h.y );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdCrossHex205( in vec3 p ){
    
    float sdh1= sdHexPrism205(  p-vec3(0.0), vec2(1.0,1.0) );
    float sdh2= sdHexPrism205(  p-vec3(0.0), vec2(0.5,1.5) );
    float sdh3= sdHexPrism205(  p.xzy-vec3(0.0), vec2(0.5,1.1) );
    float sdh4= sdHexPrism205(  p.yzx-vec3(0.0), vec2(0.5,1.5) );
    
     return max( max( max(sdh1, -sdh2), -sdh3),-sdh4);
}

float sdCrossRep205(vec3 p) {
	vec3 q = mod(p + 1.0, 2.0) - 1.0;
	return sdCrossHex205(q);
}

float sdCrossRepScale205(vec3 p, float s) {
	return sdCrossRep205(p * s) / s;	
}

//--------------------------

float fractal_de205(vec3 p) {
    float scale = 3.025;
    float dist= sdHexPrism205(p, vec2(1.0,2.0) );
	for (int i = 0; i < 5; i++) {
		dist = max(dist, -sdCrossRepScale205(p, scale));
		scale *= 3.0;
	}
    
	return dist;
}




mat2 rot206(float r){
    vec2 s = vec2(cos(r),sin(r));
    return mat2(s.x,s.y,-s.y,s.x);
}
float cube206(vec3 p,vec3 s){
    vec3 q = abs(p);
    vec3 m = max(s-q,0.);
    return length(max(q-s,0.))-min(min(m.x,m.y),m.z);
}
float tetcol206(vec3 p,vec3 offset,float scale,vec3 col){
    vec4 z = vec4(p,1.);
    for(int i = 0;i<12;i++){
        if(z.x+z.y<0.0)z.xy = -z.yx,col.z+=1.;
        if(z.x+z.z<0.0)z.xz = -z.zx,col.y+=1.;
        if(z.z+z.y<0.0)z.zy = -z.yz,col.x+=1.;
        z *= scale;
        z.xyz += offset*(1.0-scale);
    }
    return (cube206(z.xyz,vec3(1.5)))/z.w;
}
float fractal_de206(vec3 p){
    float s = 1.;
    p = abs(p)-4.*s;
    p = abs(p)-2.*s;
    p = abs(p)-1.*s;

    return tetcol206(p,vec3(1),1.8,vec3(0.));
}





float sdTriPrism( vec3 p, vec2 h ){
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}
float sdCrossHex( in vec3 p ){
    float sdfin=1000.0;
    float sdt1= sdTriPrism( p- vec3(0.0), vec2(1.0) );
    float sdt2= sdTriPrism( -p.xyz- vec3(0.0), vec2(0.5,1.2) );
    float sdt3= sdTriPrism( p.xzy- vec3(0.0), vec2(0.5,1.2) );
    sdfin =max(sdt1, -sdt2);
    sdfin =max(sdfin, -sdt3);
    return sdfin;
}
float sdCrossRep(vec3 p) {
	vec3 q = mod(p + 1.0, 2.0) - 1.0;
	return sdCrossHex(q);
}
float sdCrossRepScale(vec3 p, float s) {
	return sdCrossRep(p * s) / s;	
}
float fractal_de207(vec3 p) {
    float scale = 4.0;
    float dist=sdTriPrism( p-vec3(0.0), vec2(1.0,1.0) );
	for (int i = 0; i < 5; i++) {
		dist = max(dist, -sdCrossRepScale(p, scale));
		scale *= 3.0;
	}
	return dist;
}





float cylUnion(vec3 p){
    float xy = dot(p.xy,p.xy);
    float xz = dot(p.xz,p.xz);
    float yz = dot(p.yz,p.yz);
    return sqrt(min(xy,min(xz,yz))) - 1.;
}

float cylIntersection(vec3 p){
    float xy = dot(p.xy,p.xy);
    float xz = dot(p.xz,p.xz);
    float yz = dot(p.yz,p.yz);
    return sqrt(max(xy,max(xz,yz))) - 1.;
}

float fractal_de208(vec3 p){
    float d = cylIntersection(p);
    float s = 1.;
    for(int i = 0;i<5;i++){
        p *= 3.;
    	s*=3.;
    	float d2 = cylUnion(p) / s;
        float m = 1.; // -1 or 1
    	d = max(d,m*d2);
   	 	p = mod(p+1. , 2.) - 1.; 	
    }
    return d;
}




float maxcomp(in vec3 p ) { return max(p.x,max(p.y,p.z));}
float sdBox( vec3 p, vec3 b ){
  vec3  di = abs(p) - b;
  float mc = maxcomp(di);
  return min(mc,length(max(di,0.0)));
}

float dsCapsule(vec3 point_a, vec3 point_b, float r, vec3 point_p)
{
 	vec3 ap = point_p - point_a;
    vec3 ab = point_b - point_a;
    float ratio = dot(ap, ab) / dot(ab , ab);
    ratio = clamp(ratio, 0.0, 1.0);
    vec3 point_c = point_a + ratio * ab;
    return length(point_c - point_p) - r;
}

float DE(vec3 p){
    float distToCapsule = dsCapsule(vec3(-0.0,0.0,0.0), vec3(2.0,1.0,0.1), 1.0, p);    
    float d=distToCapsule;
    float s = 1.;
    for(int i = 0;i<5;i++){
        p *= 3.; s*=3.;
    	float d2 = cylUnion(p) / s;
        float d3=sdBox(p, vec3(2.0,1.0,2.5));
    	d = max(d,-d2);
   	 	p = mod(p+1. , 2.) - 1.; 	
    }
    return d;
}




float de(vec3 p){
    // return fractal_de6(p);
    // return fractal_de20(p);
    // return fractal_de78(p);
    // return fractal_de165(p);
    // return fractal_de192(p);
    // return fractal_de193(p);
    // return fractal_de201(p);
    return DE(p);

    // return opSmoothSubtraction(fractal_de192(p), fractal_de193(p), 0.04); 
    // return smin_op(fractal_de192(p-vec3(1,1,0)), fractal_de193(p), 0.04); 
    // return max(fSphere(p, 4.+0.1*sin(time)),sdGyroid(p, 7., 0.05, 0.1));

    
    // return smin_op(fractal_de6(p), fractal_de192(rotate3D(1.4,vec3(1.,1.,1.))*p), 0.123);
    // return smin_op(smin_op(fractal_de127(p), fractal_de130(rotate3D(2.3,vec3(1.,1.,1.))*p*4.)/4., 0.04), fractal_de195(rotate3D(0.2*time, vec3(1,2,1))*(p-vec3(0,1,0))), 0.1);
    // return smin_op(fractal_de6(p), fractal_de193(p), 0.385);
    // return smin_op(screw_de(p), fractal_de26(p), 0.333);
    // return old_de(p);
}


//  ╦═╗┌─┐┌┐┌┌┬┐┌─┐┬─┐┬┌┐┌┌─┐  ╔═╗┌─┐┌┬┐┌─┐
//  ╠╦╝├┤ │││ ││├┤ ├┬┘│││││ ┬  ║  │ │ ││├┤ 
//  ╩╚═└─┘┘└┘─┴┘└─┘┴└─┴┘└┘└─┘  ╚═╝└─┘─┴┘└─┘
// global state tracking
uint num_steps = 0; // how many steps taken by the raymarch function
float dmin = 1e10; // minimum distance initially large

float raymarch(vec3 ro, vec3 rd) {
    float d0 = 0.0, d1 = 0.0;
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * d0;      // point for distance query from parametric form
        d1 = de(p); d0 += d1;       // increment distance by de evaluated at p
        dmin = min( dmin, d1);      // tracking minimum distance
        num_steps++;                // increment step count
        if(d0 > MAX_DIST || d1 < EPSILON || i == (MAX_STEPS-1)) return d0; // return the final ray distance
    }
}

vec3 norm(vec3 p) { // to get the normal vector for a point in space, this function evaluates the gradient of the distance function
#define METHOD 2
#if METHOD == 0 
    // tetrahedron version, unknown source - 4 evaluations
    vec2 e = vec2(1,-1) * EPSILON;
    return normalize(e.xyy*de(p+e.xyy)+e.yyx*de(p+e.yyx)+e.yxy*de(p+e.yxy)+e.xxx*de(p+e.xxx));

#elif METHOD == 1
    // by iq = more efficient, 4 evaluations
    vec2 e = vec2( EPSILON, 0.); // computes the gradient of the estimator function
    return normalize( vec3(de(p)) - vec3( de(p-e.xyy), de(p-e.yxy), de(p-e.yyx) ));

#elif METHOD == 2
    // by iq - less efficient, 6 evaluations
    vec3 eps = vec3(EPSILON,0.0,0.0);
    return normalize( vec3(
                          de(p+eps.xyy) - de(p-eps.xyy),
                          de(p+eps.yxy) - de(p-eps.yxy),
                          de(p+eps.yyx) - de(p-eps.yyx)));
#endif
}

float sharp_shadow( in vec3 ro, in vec3 rd, float mint, float maxt ){
    for( float t=mint; t<maxt; )    {
        float h = de(ro + rd*t);
        if( h<0.001 )
            return 0.0;
        t += h;
    }
    return 1.0;
}

float soft_shadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k /*higher is sharper*/ ){
    float res = 1.0;
    float ph = 1e20;
    for( float t=mint; t<maxt; )
    {
        float h = de(ro + rd*t);
        if( h<EPSILON )
            return 0.0;
        float y = h*h/(2.0*ph);
        float d = sqrt(h*h-y*y);
        res = min( res, k*d/max(0.0,t-y) );
        ph = h;
        t += h;
    }
    // return res;
    res = clamp( res, 0.0, 1.0 );
    return res*res*(3.0-2.0*res);
}

vec3 visibility_only_lighting(int lightnum, vec3 hitloc){
    vec3 shadow_rd, lightpos, lightcol;
    float mint, maxt, sharpness;

    switch(lightnum){
        case 1: lightpos = lightPos1; lightcol = lightCol1d; sharpness = shadow1; break;
        case 2: lightpos = lightPos2; lightcol = lightCol2d; sharpness = shadow2; break;
        case 3: lightpos = lightPos3; lightcol = lightCol3d; sharpness = shadow3; break;
        default: break;
    }

    shadow_rd = normalize(lightpos-hitloc);

    mint = EPSILON;
    maxt = distance(hitloc, lightpos);

    if(sharpness > 99)
        return lightcol * sharp_shadow(hitloc, shadow_rd, mint, maxt);
    else
        return lightcol * soft_shadow(hitloc, shadow_rd, mint, maxt, sharpness);
}

vec3 phong_lighting(int lightnum, vec3 hitloc, vec3 norm, vec3 eye_pos){


    vec3 shadow_rd, lightpos, lightcoldiff, lightcolspec;
    float mint, maxt, lightspecpow, sharpness;

    switch(lightnum){ // eventually handle these as uniform vector inputs, to handle more than three
        case 1:
            lightpos     = eye_pos + lightPos1 * (basis_x + basis_y + basis_z);
            lightcoldiff = lightCol1d;
            lightcolspec = lightCol1s;
            lightspecpow = specpower1;
            sharpness    = shadow1;
            break;
        case 2:
            lightpos     = eye_pos + lightPos2 * (basis_x + basis_y + basis_z);
            lightcoldiff = lightCol2d;
            lightcolspec = lightCol2s;
            lightspecpow = specpower2;
            sharpness    = shadow2;
            break;
        case 3:
            lightpos     = eye_pos + lightPos3 * (basis_x + basis_y + basis_z);
            lightcoldiff = lightCol3d;
            lightcolspec = lightCol3s;
            lightspecpow = specpower3;
            sharpness    = shadow3;
            break;
        default:
            break;
    }

    mint = EPSILON;
    maxt = distance(hitloc, lightpos);
    
    /*vec3 l = -normalize(hitloc - lightpos);
    vec3 v = normalize(hitloc - eye_pos);
    vec3 n = normalize(norm);
    vec3 r = normalize(reflect(l, n));
        
    diffuse_component = occlusion_term * dattenuation_term * max(dot(n, l),0.) * lightcoldiff;
    specular_component = (dot(n,l)>0) ? occlusion_term * dattenuation_term * pow(max(dot(r,v),0.),lightspecpow) * lightcolspec : vec3(0);
    */
    
    vec3 l = normalize(lightpos - hitloc);
    vec3 v = normalize(eye_pos - hitloc);
    vec3 h = normalize(l+v);
    vec3 n = normalize(norm);
    
    // then continue with the phong calculation
    vec3 diffuse_component, specular_component;
    
    // check occlusion with the soft/sharp shadow
    float occlusion_term;
    
    if(sharpness > 99)
        occlusion_term = sharp_shadow(hitloc, l, mint, maxt);
    else
        occlusion_term = soft_shadow(hitloc, l, mint, maxt, sharpness);

    float dattenuation_term = 1./pow(distance(hitloc, lightpos), 1.1);
    
    diffuse_component = occlusion_term * dattenuation_term * max(dot(n, l), 0.) * lightcoldiff;
    specular_component = (dot(n,l) > 0) ? occlusion_term * dattenuation_term * ((lightspecpow+2)/(2*M_PI)) * pow(max(dot(n,h),0.),lightspecpow) * lightcolspec : vec3(0);

    return diffuse_component + specular_component;
}


float calcAO( in vec3 pos, in vec3 nor )
{
    float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.001 + 0.15*float(i)/4.0;
        float d = de( pos + h*nor );
        occ += (h-d)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 1.5*occ, 0.0, 1.0 );
}

// By TekF... getting a crash trying to use this (default value used was 0.5 degree)
// void BarrelDistortion( inout vec3 ray, float degree )
// {
// 	ray.z /= degree;
// 	ray.z = ( ray.z*ray.z - dot(ray.xy,ray.xy) );
// 	ray.z = degree*sqrt(ray.z);
// }

void main()
{

    // imageStore(current, ivec2(gl_GlobalInvocationID.xy), uvec4( 120, 45, 12, 255 ));

    vec4 col = vec4(0, 0, 0, 1);
    float dresult_avg = 0.;

    for(int x = 0; x < AA; x++)
    for(int y = 0; y < AA; y++)
    {
        vec2 offset = vec2(float(x), float(y)) / float(AA) - 0.5;

        vec2 pixcoord = (vec2(gl_GlobalInvocationID.xy + offset)-vec2(imageSize(current)/2.)) / vec2(imageSize(current)/2.);
        vec3 ro = ray_origin;

        float aspect_ratio;
        // aspect_ratio = 1.618;
        aspect_ratio = float(imageSize(current).x) / float(imageSize(current).y);
        vec3 rd = normalize(aspect_ratio*pixcoord.x*basis_x + pixcoord.y*basis_y + (1./fov)*basis_z);

        escape = 0.;
        float dresult = raymarch(ro, rd);
        float escape_result = escape;

        // vec3 lightpos = vec3(8.); pR(lightpos.xz, time);
        vec3 lightpos = vec3(2*sin(time), 2., 2*cos(time));

        vec3 hitpos = ro+dresult*rd;
        vec3 normal = norm(hitpos);

        vec3 shadow_ro = hitpos+normal*EPSILON*2.;

        vec3 sresult1 = vec3(0.);
        vec3 sresult2 = vec3(0.);
        vec3 sresult3 = vec3(0.);
        
        sresult1 = phong_lighting(1, hitpos, normal, ro) * flickerfactor1;
        sresult2 = phong_lighting(2, hitpos, normal, ro) * flickerfactor2;
        sresult3 = phong_lighting(3, hitpos, normal, ro) * flickerfactor3;
        
        // vec3 temp = ((norm(hitpos)/2.)+vec3(0.5)); // visualizing normal vector
        
        vec3 palatte_read = 0.4 * basic_diffuse * pal( escape_result, vec3(0.5,0.5,0.5),vec3(0.5,0.5,0.5),vec3(1.0,1.0,1.0),vec3(0.0,0.10,0.20) );
        
        // apply lighting
        // vec3 temp = basic_diffuse + sresult1 + sresult2  + sresult3;
        // vec3 temp = palatte_read + sresult1 + sresult2  + sresult3;
        vec3 temp = palatte_read * (sresult1 + sresult2  + sresult3);


        temp *= ((1./AO_scale) * calcAO(shadow_ro, normal)); // ambient occlusion calculation

        // do the depth scaling here
        // compute the depth scale term
        float depth_term = depth_scale * dresult;
        switch(depth_falloff)
        {
            case 0: depth_term = 0.;
            case 1: depth_term = 2.-2.*(1./(1.-depth_term)); break;
            case 2: depth_term = 1.-(1./(1+0.1*depth_term*depth_term)); break;
            case 3: depth_term = (1-pow(depth_term/30., 1.618)); break;

            case 4: depth_term = clamp(exp(0.25*depth_term-3.), 0., 10.); break;
            case 5: depth_term = exp(0.25*depth_term-3.); break;
            case 6: depth_term = exp( -0.002 * depth_term * depth_term * depth_term ); break;
            case 7: depth_term = exp(-0.6*max(depth_term-3., 0.0)); break;
    
            case 8: depth_term = (sqrt(depth_term)/8.) * depth_term; break;
            case 9: depth_term = sqrt(depth_term/9.); break;
            case 10: depth_term = pow(depth_term/10., 2.); break;
            default: break;
        }
        // do a mix here, between col and the fog color, with the selected depth falloff term
        temp.rgb = mix(temp.rgb, fog_color.rgb, depth_term);
        
        col.rgb += temp;
    }

    col.rgb /= float(AA*AA);
    dresult_avg /= float(AA*AA);

    dresult_avg *= depth_scale;

    // compute the depth scale term
    float depth_term; 

    switch(depth_falloff)
    {
        case 0: depth_term = 2.-2.*(1./(1.-dresult_avg)); break;
        case 1: depth_term = 1.-(1./(1+0.1*dresult_avg*dresult_avg)); break;
        case 2: depth_term = (1-pow(dresult_avg/30., 1.618)); break;

        case 3: depth_term = clamp(exp(0.25*dresult_avg-3.), 0., 10.); break;
        case 4: depth_term = exp(0.25*dresult_avg-3.); break;
        case 5: depth_term = exp( -0.002 * dresult_avg * dresult_avg * dresult_avg ); break;
        case 6: depth_term = exp(-0.6*max(dresult_avg-3., 0.0)); break;
    
        case 7: depth_term = (sqrt(dresult_avg)/8.) * dresult_avg; break;
        case 8: depth_term = sqrt(dresult_avg/9.); break;
        case 9: depth_term = pow(dresult_avg/10., 2.); break;
        case 10: depth_term = dresult_avg/MAX_DIST;
        default: break;
    }
    // do a mix here, between col and the fog color, with the selected depth falloff term
    col.rgb = mix(col.rgb, fog_color.rgb, depth_term);

    // color stuff happens here, because the imageStore will be quantizing to 8 bit
    // tonemapping 
    switch(tonemap_mode)
    {
        case 0: // None (Linear)
            break;
        case 1: // ACES (Narkowicz 2015)
            col.xyz = cheapo_aces_approx(col.xyz);
            break;
        case 2: // Unreal Engine 3
            col.xyz = pow(tonemap_unreal3(col.xyz), vec3(2.8));
            break;
        case 3: // Unreal Engine 4
            col.xyz = aces_fitted(col.xyz);
            break;
        case 4: // Uncharted 2
            col.xyz = uncharted2(col.xyz);
            break;
        case 5: // Gran Turismo
            col.xyz = tonemap_uchimura(col.xyz);
            break;
        case 6: // Modified Gran Turismo
            col.xyz = tonemap_uchimura2(col.xyz);
            break;
        case 7: // Rienhard
            col.xyz = rienhard(col.xyz);
            break;
        case 8: // Modified Rienhard
            col.xyz = rienhard2(col.xyz);
            break;
        case 9: // jt_tonemap
            col.xyz = jt_toneMap(col.xyz);
            break;
        case 10: // robobo1221s
            col.xyz = robobo1221sTonemap(col.xyz);
            break;
        case 11: // robo
            col.xyz = roboTonemap(col.xyz);
            break;
        case 12: // jodieRobo
            col.xyz = jodieRoboTonemap(col.xyz);
            break;
        case 13: // jodieRobo2
            col.xyz = jodieRobo2ElectricBoogaloo(col.xyz);
            break;
        case 14: // jodieReinhard
            col.xyz = jodieReinhardTonemap(col.xyz);
            break;
        case 15: // jodieReinhard2
            col.xyz = jodieReinhard2ElectricBoogaloo(col.xyz);
            break;
    }   
    // gamma correction
    col.rgb = pow(col.rgb, vec3(1/gamma));

    imageStore(current, ivec2(gl_GlobalInvocationID.xy), uvec4( col.r*255, col.g*255, col.b*255, col.a*255 ));
}
