#version 450
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in uint in_instance;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view_proj;
    vec3 eye;
    float exposure;
} camera;

layout(set = 1, binding = 0) readonly buffer InstanceTransforms {
    mat4 models[128];
} instance_data;

layout(set = 1, binding = 1) readonly buffer LightData {
    vec4 positions[16];
    vec4 colors[16];
} lights;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec3 frag_light;

void main() {
    mat4 model = instance_data.models[in_instance % 128u];
    vec4 world = model * vec4(in_position, 1.0);
    vec3 light_color = lights.colors[in_instance % 16u].rgb;

    frag_uv = in_uv;
    frag_light = light_color * max(dot(normalize(lights.positions[0].xyz - world.xyz), vec3(0.0, 0.0, 1.0)), 0.1);
    gl_Position = camera.view_proj * world;
}
