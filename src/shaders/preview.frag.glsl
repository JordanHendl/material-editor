#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform PreviewUniforms {
    mat4 view_proj;
    mat3 normal_matrix;
    vec4 light_dir;
    vec4 fallback_color;
    vec4 flags;
} uniforms;

layout(binding = 1) uniform sampler2D preview_texture;

void main() {
    vec4 sampled = uniforms.flags.x > 0.5
        ? texture(preview_texture, v_uv)
        : uniforms.fallback_color;
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(uniforms.light_dir.xyz);
    float ndotl = max(dot(normal, light), 0.1);
    vec3 lit = sampled.rgb * ndotl;
    out_color = vec4(lit, sampled.a);
}
