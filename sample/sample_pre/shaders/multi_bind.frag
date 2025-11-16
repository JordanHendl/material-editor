#version 450
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec3 frag_light;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view_proj;
    vec3 eye;
    float exposure;
} camera;

layout(set = 1, binding = 1) readonly buffer LightData {
    vec4 positions[16];
    vec4 colors[16];
} lights;

layout(set = 2, binding = 0) uniform sampler2D material_layers[4];
layout(set = 3, binding = 0) uniform sampler2D bindless_layers[64];

vec4 sample_material(uint index, vec2 uv) {
    if (index < 4u) {
        return texture(material_layers[index], uv);
    }
    uint bindless_index = index - 4u;
    return texture(bindless_layers[bindless_index % 64u], uv * vec2(1.0, -1.0) + vec2(0.0, 1.0));
}

void main() {
    vec4 albedo = sample_material(0u, frag_uv);
    vec4 overlay = sample_material(5u, frag_uv * 0.75 + vec2(0.1));
    vec3 lit_color = albedo.rgb * (0.25 + frag_light);

    float exposure = max(camera.exposure, 0.0001);
    vec3 tone_mapped = vec3(1.0) - exp(-lit_color * exposure);
    out_color = vec4(mix(tone_mapped, overlay.rgb, overlay.a * 0.6), albedo.a);
}
