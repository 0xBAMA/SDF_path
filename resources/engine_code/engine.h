#ifndef ENGINE
#define ENGINE

#include "includes.h"

// These defines are used to simplify the ImGui::Combo things in engine_utils.cc
 
// colorspace
#define RGB       0
#define SRGB      1
#define XYZ       2
#define XYY       3
#define HSV       4
#define HSL       5
#define HCY       6
#define YPBPR     7
#define YPBPR601  8
#define YCBCR1    9
#define YCBCR2    10
#define YCCBCCRC  11
#define YCOCG     12
#define BCH       13
#define CHROMAMAX 14
#define OKLAB     15

// dither pattern
#define BAYER             1
#define STATIC_MONO_BLUE  2
#define STATIC_RGB_BLUE   3
#define CYCLED_MONO_BLUE  4
#define CYCLED_RGB_BLUE   5
#define UNIFORM           6
#define INTERLEAVED_GRAD  7
#define VLACHOS           8
#define TRIANGLE_VLACHOS  9
#define TRIANGLE_MONO     10
#define TRIANGLE_RGB      11

// dither methodology
#define BITCRUSH      0
#define EXPONENTIAL   1

class engine
{
public:

	engine();
	~engine();

private:

	SDL_Window * window;
	SDL_GLContext GLcontext;

	ImVec4 clear_color;
	int total_screen_width, total_screen_height;

// OpenGL Handles
    GLuint display_texture;
    GLuint display_shader;
	GLuint display_vao;
	GLuint display_vbo;

	// dither patterns
	GLuint dither_bayer;
	GLuint dither_blue;

	// compute shaders
	GLuint dither_shader;
	GLuint blue_cycle_shader;
	GLuint raymarch_shader;

// raymarcher state
	// rotation
	float rotation_about_x = 0.;
	float rotation_about_y = 0.;
	float rotation_about_z = 0.;

	// base color
	glm::vec3 basic_diffuse = glm::vec3(45./255., 45./255., 45./255.);
		
	// light animation factors
	void animate_lights(float t);
	float flickerfactor1 = 1.0;
	float orbitradius1 = 2.0;
	float orbitrate1 = 1.2;
	float phaseoffset1 = 0.5;
		
	float flickerfactor2 = 1.0;
	float orbitradius2 = 2.2;
	float orbitrate2 = 1.3;
	float phaseoffset2 = 5.6;

	float flickerfactor3 = 1.0;
	float orbitradius3 = 2.9;
	float orbitrate3 = 1.8;
	float phaseoffset3 = 4.4;
		
	// diffuse light colors
	glm::vec3 lightCol1d = glm::vec3( 0.6f, 0.6f, 0.6f);
	glm::vec3 lightCol2d = glm::vec3( 0.75f, 0.3f, 0.0f);
	glm::vec3 lightCol3d = glm::vec3( 0.1f, 0.35f, 0.65f);
	// specular light colors	
	glm::vec3 lightCol1s = glm::vec3( 0.5f, 0.5f, 0.5f);
	glm::vec3 lightCol2s = glm::vec3( 0.5f, 0.5f, 0.5f);
	glm::vec3 lightCol3s = glm::vec3( 0.5f, 0.5f, 0.5f);
	// specular power terms
	float specpower1 = 5;
	float specpower2 = 50;
	float specpower3 = 150;
	// shadow sharpness terms
	float shadow1 = 100;
	float shadow2 = 100;
	float shadow3 = 100;
	// light orbit radius
	float orbit1 = 100;
	float orbit2 = 100;
	float orbit3 = 100;

	// light positions
	glm::vec3 lightPos1 = glm::vec3( 0.1, 0., 0.);
    glm::vec3 lightPos2 = glm::vec3( 0., 0.1, 0.);
    glm::vec3 lightPos3 = glm::vec3( 0., 0., 0.1);

	float AO_scale = 0.7;
	float depth_scale = 1.;
	int depth_selector = 8;	

	float fov = 1.;

	// position
	glm::vec3 position = glm::vec3(1., 1., 1.);

	glm::vec3 basis_x, basis_y, basis_z;

	float gamma_correction = 1.244;
	int current_tmode = 3;
    int current_colorspace = CHROMAMAX;
    int current_noise_func = BAYER;
    int current_dither_mode = EXPONENTIAL;

	int num_bits = 4; 

	void screenshot(std::string name);

// main loop functions
	void create_window();
	void gl_setup();
	void draw_everything();
	void start_imgui();
	void end_imgui();
	void control_window();
	void editor_window();


	// performance reporting
	int dither_microseconds;
	int raymarch_microseconds;
	int display_microseconds;
		
	int total_loop_microseconds=0;
		
	// render flags (toggle shader execution)
	bool raymarch_stage = true;
	bool dither_stage = true;

	// to confirm quit
	bool quitconfirm = false;
	void quit_conf(bool *open);

	// main loop control
	void quit();
	bool pquit = false;

public:
// OBJ data (per mesh)
	void load_OBJ(std::string filename);

	// this may vary in length
	std::vector<glm::vec4> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec3> texcoords;

	// these should all be the same length, the number of triangles
	std::vector<glm::ivec3> triangle_indices;
	std::vector<glm::ivec3> normal_indices;
	std::vector<glm::ivec3> texcoord_indices;

};

#endif
