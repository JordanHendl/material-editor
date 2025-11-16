#version 450
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(set = 0, binding = 0) uniform Settings {
    mat4 jittered_proj;
    vec2 film_grain_seed;
    float vignette_strength;
} settings;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec2 grain_seed;

void main() {
    frag_uv = in_uv;
    grain_seed = settings.film_grain_seed + in_uv * 13.37;
    gl_Position = settings.jittered_proj * vec4(in_position, 1.0);
}
