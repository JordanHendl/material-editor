#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 3) in vec2 in_uv;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;

layout(binding = 0) uniform PreviewUniforms {
    mat4 view_proj;
    mat3 normal_matrix;
    vec4 light_dir;
    vec4 fallback_color;
    vec4 flags;
} uniforms;

void main() {
    gl_Position = uniforms.view_proj * vec4(in_position, 1.0);
    v_normal = uniforms.normal_matrix * in_normal;
    v_uv = in_uv;
}
