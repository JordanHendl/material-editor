#version 450
layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec2 grain_seed;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 1) uniform Exposure {
    float exposure;
    float gamma;
} exposure_data;

layout(set = 3, binding = 0) uniform sampler2D bindless_textures[32];
layout(set = 3, binding = 1) readonly buffer BindlessIndices {
    uint albedo_index;
    uint bloom_index;
} bindless_indices;

float vignette(vec2 uv, float strength) {
    vec2 dist = uv - vec2(0.5);
    float falloff = 1.0 - dot(dist, dist) * strength;
    return clamp(falloff, 0.0, 1.0);
}

vec3 tonemap(vec3 color, float exposure) {
    return vec3(1.0) - exp(-color * exposure);
}

void main() {
    vec4 albedo = texture(bindless_textures[bindless_indices.albedo_index % 32u], frag_uv);
    vec4 bloom = texture(bindless_textures[bindless_indices.bloom_index % 32u], frag_uv * 0.5);

    vec3 combined = albedo.rgb + bloom.rgb * 0.4;
    combined = tonemap(combined, max(exposure_data.exposure, 0.0001));
    combined = pow(combined, vec3(1.0 / max(exposure_data.gamma, 0.0001)));

    float grain = fract(sin(dot(grain_seed, vec2(12.9898, 78.233))) * 43758.5453);
    float vignette_mask = vignette(frag_uv, 1.5);

    out_color = vec4(combined * vignette_mask + grain * 0.02, albedo.a);
}
