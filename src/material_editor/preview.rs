use std::{
    collections::HashMap,
    f32::consts::PI,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

use bytemuck::{Pod, Zeroable};
use dashi::gpu::execution::CommandRing;
use dashi::{
    AspectMask, BindGroupLayout, BindGroupLayoutInfo, BindGroupVariable, BindGroupVariableType,
    BufferInfo, BufferUsage, ClearValue, CommandQueueInfo2, Context, ContextInfo, DepthInfo,
    Format, GraphicsPipelineDetails, GraphicsPipelineInfo, GraphicsPipelineLayoutInfo, Handle,
    ImageInfo, ImageView, MemoryVisibility, PipelineShaderInfo, Rect2D, ShaderPrimitiveType,
    ShaderType, VertexDescriptionInfo, VertexEntryInfo, VertexRate, Viewport,
    builders::RenderPassBuilder,
    driver::command::{BeginRenderPass, CopyImageBuffer, DrawIndexed},
    gpu::CommandStream,
};
use eframe::egui::{self, Color32, ColorImage};
use glam::{Mat3, Mat4, Vec3};
use noren::datatypes::{
    DatabaseEntry, ImageDB, ShaderDB, ShaderModule, leak_database_entry, primitives::Vertex,
};
use shaderc::{Compiler, ShaderKind};

use noren::parsing::GraphicsShaderLayout;

use crate::{
    material_bindings::{TextureBindingKind, TextureBindingSlot, texture_binding_slots},
    material_editor::project::{GraphTexture, MaterialEditorProjectState},
    material_editor_types::{
        MaterialEditorDatabaseLayout, MaterialEditorGraphicsShader, MaterialEditorMaterial,
    },
};

const PREVIEW_WIDTH: usize = 320;
const PREVIEW_HEIGHT: usize = 240;

pub struct MaterialPreviewPanel {
    renderer: PreviewRenderer,
    texture: Option<egui::TextureHandle>,
    texture_label: String,
}

impl MaterialPreviewPanel {
    pub fn new(state: &MaterialEditorProjectState) -> Self {
        Self {
            renderer: PreviewRenderer::new(state),
            texture: None,
            texture_label: "material_preview".to_string(),
        }
    }

    pub fn sync_with_state(&mut self, state: &MaterialEditorProjectState) {
        self.renderer.sync_with_state(state);
        self.texture = None;
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        state: &MaterialEditorProjectState,
        material_id: &str,
        material: &MaterialEditorMaterial,
    ) {
        egui::CollapsingHeader::new("Preview")
            .default_open(true)
            .show(ui, |ui| {
                self.draw_controls(ui);

                let result = self.renderer.render(material_id, material, state);
                self.update_texture(ui, result.image_changed);

                if let Some(texture) = &self.texture {
                    let aspect = PREVIEW_HEIGHT as f32 / PREVIEW_WIDTH as f32;
                    let width = ui.available_width().min(PREVIEW_WIDTH as f32);
                    let height = width * aspect;
                    let image =
                        egui::Image::new(texture).fit_to_exact_size(egui::vec2(width, height));
                    ui.add(image);
                } else {
                    ui.label("Preview unavailable");
                }

                if !result.warnings.is_empty() {
                    ui.separator();
                    for warning in result.warnings {
                        ui.label(
                            egui::RichText::new(warning).color(Color32::from_rgb(235, 168, 75)),
                        );
                    }
                }
            });
    }

    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Mesh");
            egui::ComboBox::from_id_source("preview_mesh_selector")
                .selected_text(self.renderer.config.mesh_kind.label())
                .show_ui(ui, |ui| {
                    for kind in PreviewMeshKind::ALL {
                        let selected = self.renderer.config.mesh_kind == kind;
                        if ui.selectable_label(selected, kind.label()).clicked() {
                            self.renderer.config.mesh_kind = kind;
                        }
                    }
                });
        });

        ui.horizontal(|ui| {
            ui.label("Camera");
            ui.add(
                egui::Slider::new(&mut self.renderer.config.camera.azimuth, -180.0..=180.0)
                    .text("Azimuth"),
            );
            ui.add(
                egui::Slider::new(&mut self.renderer.config.camera.elevation, -80.0..=80.0)
                    .text("Elevation"),
            );
            ui.add(
                egui::Slider::new(&mut self.renderer.config.camera.distance, 1.0..=6.0)
                    .text("Distance"),
            );
            if ui.button("Reset").clicked() {
                self.renderer.config.camera = OrbitCamera::default();
            }
        });

        ui.horizontal(|ui| {
            ui.label("Background");
            ui.color_edit_button_rgb(&mut self.renderer.config.background_rgb);
            ui.checkbox(&mut self.renderer.config.wireframe, "Wireframe");
        });
    }

    fn update_texture(&mut self, ui: &mut egui::Ui, image_changed: bool) {
        let image = self.renderer.image();
        if self.texture.is_none() {
            let handle = ui.ctx().load_texture(
                self.texture_label.clone(),
                image.clone(),
                egui::TextureOptions::LINEAR,
            );
            self.texture = Some(handle);
            return;
        }

        if image_changed {
            if let Some(texture) = &mut self.texture {
                texture.set(image.clone(), egui::TextureOptions::LINEAR);
            }
        }
    }
}

struct PreviewRenderer {
    config: PreviewConfig,
    mesh_cache: PreviewMeshCache,
    assets: PreviewAssetCache,
    image: ColorImage,
    gpu: Option<PreviewGpu>,
    gpu_error: Option<String>,
}

impl PreviewRenderer {
    fn new(state: &MaterialEditorProjectState) -> Self {
        let (gpu, gpu_error) = match PreviewGpu::new() {
            Ok(gpu) => (Some(gpu), None),
            Err(err) => (None, Some(err)),
        };
        Self {
            config: PreviewConfig::default(),
            mesh_cache: PreviewMeshCache::default(),
            assets: PreviewAssetCache::new(state.root(), &state.layout),
            image: ColorImage::new([PREVIEW_WIDTH, PREVIEW_HEIGHT], Color32::BLACK),
            gpu,
            gpu_error,
        }
    }

    fn sync_with_state(&mut self, state: &MaterialEditorProjectState) {
        self.assets.reset(state.root(), &state.layout);
    }

    fn render(
        &mut self,
        material_id: &str,
        material: &MaterialEditorMaterial,
        state: &MaterialEditorProjectState,
    ) -> PreviewResult {
        let mut warnings = Vec::new();

        if self.config.wireframe {
            warnings.push("Wireframe preview is not available in the GPU renderer yet".into());
        }

        if self.gpu.is_none() {
            if let Some(err) = &self.gpu_error {
                warnings.push(format!("Preview renderer unavailable: {err}"));
            } else {
                warnings.push("Preview renderer unavailable".into());
            }
            return PreviewResult {
                warnings,
                image_changed: false,
            };
        }

        {
            let gpu_for_assets = self
                .gpu
                .as_mut()
                .expect("gpu should be present after check");
            self.assets
                .ensure_imagery(state.root(), &state.layout, gpu_for_assets);
        }
        self.assets.ensure_shaders(state.root(), &state.layout);

        let resolved = match self.resolve_textures(material, state, &mut warnings) {
            Some(result) => result,
            None => {
                warnings.push(
                    "Shader information unavailable; preview will use fallback colors".into(),
                );
                ResolvedTextureBindings::default()
            }
        };

        let texture = resolved
            .descriptor_bindings
            .iter()
            .flatten()
            .next()
            .cloned();
        if texture.is_none() {
            warnings.push("No previewable texture bindings; using fallback colors".to_string());
        }

        let shader_label = material
            .shader
            .as_deref()
            .unwrap_or("builtin_preview")
            .to_string();
        let shader_modules = match material.shader.as_deref() {
            Some(shader_id) => match state.graph.shaders.get(shader_id) {
                Some(shader) => {
                    match self.assets.shader_modules(shader_id, &shader.resource.data) {
                        Ok(mods) => Some(mods),
                        Err(err) => {
                            warnings.push(err);
                            None
                        }
                    }
                }
                None => {
                    warnings.push(format!("Shader '{shader_id}' is missing"));
                    None
                }
            },
            None => None,
        };

        let gpu = self
            .gpu
            .as_mut()
            .expect("gpu should be present after check");
        let mesh = self.mesh_cache.mesh(self.config.mesh_kind, gpu);
        let background = Color32::from_rgb(
            (self.config.background_rgb[0].clamp(0.0, 1.0) * 255.0) as u8,
            (self.config.background_rgb[1].clamp(0.0, 1.0) * 255.0) as u8,
            (self.config.background_rgb[2].clamp(0.0, 1.0) * 255.0) as u8,
        );

        let light_dir = Vec3::new(0.3, 0.8, 0.6).normalize();

        let render_result = gpu.render(
            mesh,
            &self.config,
            texture,
            background,
            light_dir,
            &mut self.image,
            &shader_label,
            shader_modules
                .as_ref()
                .map(|modules| (&modules.vertex, &modules.fragment)),
        );

        if let Err(err) = render_result {
            warnings.push(format!(
                "Failed to render preview for '{material_id}': {err}"
            ));
            return PreviewResult {
                warnings,
                image_changed: false,
            };
        }

        PreviewResult {
            warnings,
            image_changed: true,
        }
    }

    fn resolve_textures(
        &mut self,
        material: &MaterialEditorMaterial,
        state: &MaterialEditorProjectState,
        warnings: &mut Vec<String>,
    ) -> Option<ResolvedTextureBindings> {
        let shader_id = material.shader.as_ref()?;
        let shader = state.graph.shaders.get(shader_id)?;
        let layout: GraphicsShaderLayout = shader.resource.data.clone().into();
        let slots = texture_binding_slots(&layout);
        let mut handles = Vec::with_capacity(slots.len());
        let mut bindless_ids = Vec::new();
        for (index, slot) in slots.iter().enumerate() {
            let binding = material.textures.get(index).cloned().unwrap_or_default();
            match slot.kind {
                TextureBindingKind::BindGroup { .. } => {
                    let Some(value) = binding.as_texture() else {
                        warnings.push(format!(
                            "{} is unassigned; preview will use fallback colors",
                            Self::describe_slot(slot)
                        ));
                        handles.push(None);
                        continue;
                    };
                    let Some(GraphTexture { resource }) = state.graph.textures.get(value) else {
                        warnings.push(format!("Texture '{value}' is missing"));
                        handles.push(None);
                        continue;
                    };
                    let image_entry = resource.data.image.clone();
                    if image_entry.is_empty() {
                        warnings.push(format!("Texture '{value}' does not reference imagery"));
                        handles.push(None);
                        continue;
                    }
                    if let Some(texture) = self.assets.texture(&image_entry) {
                        handles.push(Some(texture));
                    } else {
                        warnings.push(format!(
                            "Failed to load imagery '{}' for texture '{}'",
                            image_entry, value
                        ));
                        handles.push(None);
                    }
                }
                TextureBindingKind::BindTable { .. } => {
                    let numeric_id = binding
                        .as_bindless()
                        .or_else(|| binding.value())
                        .map(|reference| {
                            if !state.graph.textures.contains_key(reference) {
                                warnings
                                    .push(format!("Bindless reference '{}' is missing", reference));
                            }

                            reference.parse::<u32>().unwrap_or_else(|_| {
                                warnings.push(format!(
                                    "Bindless reference '{}' is not a numeric ID; using 0",
                                    reference
                                ));
                                0
                            })
                        })
                        .unwrap_or_default();
                    bindless_ids.push(numeric_id);
                    handles.push(None);
                }
            }
        }
        Some(ResolvedTextureBindings {
            slots,
            descriptor_bindings: handles,
            bindless_ids,
        })
    }

    fn describe_slot(slot: &TextureBindingSlot) -> String {
        match slot.kind {
            TextureBindingKind::BindGroup { group, binding } => {
                format!("Set {} binding {} [{}]", group, binding, slot.element)
            }
            TextureBindingKind::BindTable { table, binding } => {
                format!("Table {} binding {} [{}]", table, binding, slot.element)
            }
        }
    }

    fn image(&self) -> &ColorImage {
        &self.image
    }
}

struct PreviewResult {
    warnings: Vec<String>,
    image_changed: bool,
}

#[derive(Default)]
struct ResolvedTextureBindings {
    slots: Vec<TextureBindingSlot>,
    descriptor_bindings: Vec<Option<PreviewTextureHandle>>,
    bindless_ids: Vec<u32>,
}

struct PreviewGpu {
    ctx: Context,
    ring: CommandRing,
    render_pass: Handle<dashi::RenderPass>,
    target: PreviewTarget,
    pipeline: Option<PreviewPipeline>,
    pipeline_signature: Option<PreviewShaderSignature>,
    sampler: Handle<dashi::Sampler>,
    bind_group_layout: Handle<BindGroupLayout>,
    uniform_buffer: Handle<dashi::Buffer>,
    fallback_texture: PreviewTextureHandle,
    bind_groups: HashMap<String, Handle<dashi::BindGroup>>,
    builtin_shader: BuiltinShader,
}

impl PreviewGpu {
    fn new() -> Result<Self, String> {
        let mut ctx = Context::headless(&ContextInfo::default())
            .map_err(|err| format!("unable to create GPU context: {err}"))?;
        let ring = ctx
            .make_command_ring(&CommandQueueInfo2 {
                debug_name: "preview",
                ..Default::default()
            })
            .map_err(|err| format!("unable to create command ring: {err}"))?;

        let viewport = Viewport {
            area: dashi::FRect2D {
                x: 0.0,
                y: 0.0,
                w: PREVIEW_WIDTH as f32,
                h: PREVIEW_HEIGHT as f32,
            },
            scissor: Rect2D {
                x: 0,
                y: 0,
                w: PREVIEW_WIDTH as u32,
                h: PREVIEW_HEIGHT as u32,
            },
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let color_attachment = dashi::AttachmentDescription {
            format: Format::RGBA8,
            ..Default::default()
        };
        let depth_attachment = dashi::AttachmentDescription {
            format: Format::D24S8,
            ..Default::default()
        };

        let render_pass = RenderPassBuilder::new("material_preview", viewport)
            .add_subpass(&[color_attachment], Some(&depth_attachment), &[])
            .build(&mut ctx)
            .map_err(|_| "failed to build preview render pass".to_string())?;

        let target = PreviewTarget::new(&mut ctx)?;
        let uniform_buffer = ctx
            .make_buffer(&BufferInfo {
                debug_name: "preview_uniforms",
                byte_size: std::mem::size_of::<PreviewUniforms>() as u32,
                visibility: MemoryVisibility::CpuAndGpu,
                usage: BufferUsage::UNIFORM,
                initial_data: None,
            })
            .map_err(|_| "failed to allocate uniform buffer".to_string())?;

        let sampler = ctx
            .make_sampler(&Default::default())
            .map_err(|_| "failed to create sampler".to_string())?;

        let bind_group_layout = create_bind_group_layout(&mut ctx)?;
        let fallback_texture = PreviewTextureHandle::solid_color(&mut ctx, [200, 120, 240, 255])?;
        let bind_group = create_bind_group(
            &mut ctx,
            bind_group_layout,
            uniform_buffer,
            sampler,
            fallback_texture.view,
        )?;
        let mut bind_groups = HashMap::new();
        bind_groups.insert("fallback".to_string(), bind_group);

        let builtin_shader = builtin_shader_modules()?;

        Ok(Self {
            ctx,
            ring,
            render_pass,
            target,
            pipeline: None,
            pipeline_signature: None,
            sampler,
            bind_group_layout,
            uniform_buffer,
            fallback_texture,
            bind_groups,
            builtin_shader,
        })
    }

    fn render(
        &mut self,
        mesh: &GpuPreviewMesh,
        config: &PreviewConfig,
        texture: Option<PreviewTextureHandle>,
        background: Color32,
        light_dir: Vec3,
        image: &mut ColorImage,
        shader_label: &str,
        shader_modules: Option<(&ShaderModule, &ShaderModule)>,
    ) -> Result<(), String> {
        self.ensure_pipeline(shader_label, shader_modules)?;
        let pipeline_handle = self
            .pipeline
            .as_ref()
            .map(|pipeline| pipeline.pipeline)
            .ok_or_else(|| "preview pipeline unavailable".to_string())?;
        let has_texture = texture.is_some();
        self.update_uniforms(config, light_dir, has_texture)?;
        let resolved_texture = match texture {
            Some(handle) => handle,
            None => self.fallback_texture.clone(),
        };
        let bind_group = self.bind_group_for(&resolved_texture)?;

        let clear = ClearValue::Color([
            background.r() as f32 / 255.0,
            background.g() as f32 / 255.0,
            background.b() as f32 / 255.0,
            1.0,
        ]);

        let depthclear = ClearValue::DepthStencil {
            depth: 1.0,
            stencil: 0,
        };

        self.ring
            .record(|cmd| {
                let stream = CommandStream::new().begin();
                let begin_pass = BeginRenderPass {
                    viewport: self.target.viewport,
                    render_pass: self.render_pass,
                    color_attachments: [Some(self.target.color_view), None, None, None],
                    depth_attachment: Some(self.target.depth_view),
                    clear_values: [Some(clear), Some(depthclear), None, None],
                };
                let pending = stream.begin_render_pass(&begin_pass);
                let mut drawing = pending.bind_graphics_pipeline(pipeline_handle);
                let mut draw_cmd = DrawIndexed::default();
                draw_cmd.vertices = mesh.vertex_buffer;
                draw_cmd.indices = mesh.index_buffer;
                draw_cmd.index_count = mesh.index_count;
                draw_cmd.bind_groups[0] = Some(bind_group);
                drawing.draw_indexed(&draw_cmd);
                let pending = drawing.unbind_graphics_pipeline();
                let mut recording = pending.stop_drawing();

                let copy = CopyImageBuffer {
                    src: self.target.color,
                    dst: self.target.readback,
                    range: Default::default(),
                    dst_offset: 0,
                };
                recording.copy_image_to_buffer(&copy);
                let exec = recording.end();
                exec.append(cmd);
            })
            .map_err(|err| format!("failed to record preview commands: {err}"))?;

        self.ring
            .submit(&Default::default())
            .map_err(|err| format!("failed to submit preview commands: {err}"))?;
        self.ring
            .wait_all()
            .map_err(|err| format!("failed to wait for preview commands: {err}"))?;

        self.readback(image)?;
        Ok(())
    }

    fn ctx_ptr(&mut self) -> *mut Context {
        &mut self.ctx as *mut _
    }

    fn ensure_pipeline(
        &mut self,
        shader_label: &str,
        shader_modules: Option<(&ShaderModule, &ShaderModule)>,
    ) -> Result<(), String> {
        let (vertex, fragment, label) = match shader_modules {
            Some((vertex, fragment)) => (vertex, fragment, shader_label),
            None => (
                &self.builtin_shader.vertex,
                &self.builtin_shader.fragment,
                "builtin_preview",
            ),
        };

        let signature = PreviewShaderSignature::from_modules(label, vertex, fragment);
        if self.pipeline_signature.as_ref() != Some(&signature) {
            let pipeline = PreviewPipeline::new(
                &mut self.ctx,
                self.render_pass,
                self.bind_group_layout,
                vertex,
                fragment,
                label,
            )?;
            self.pipeline = Some(pipeline);
            self.pipeline_signature = Some(signature);
        }
        Ok(())
    }

    fn bind_group_for(
        &mut self,
        texture: &PreviewTextureHandle,
    ) -> Result<Handle<dashi::BindGroup>, String> {
        if let Some(handle) = self.bind_groups.get(&texture.name) {
            return Ok(*handle);
        }
        let handle = create_bind_group(
            &mut self.ctx,
            self.bind_group_layout,
            self.uniform_buffer,
            self.sampler,
            texture.view,
        )?;
        self.bind_groups.insert(texture.name.clone(), handle);
        Ok(handle)
    }

    fn update_uniforms(
        &mut self,
        config: &PreviewConfig,
        light_dir: Vec3,
        has_texture: bool,
    ) -> Result<(), String> {
        let view = config.camera.view_matrix();
        let aspect = PREVIEW_WIDTH as f32 / PREVIEW_HEIGHT as f32;
        let proj = Mat4::perspective_rh_gl(45.0_f32.to_radians(), aspect, 0.1, 50.0);
        let view_proj = proj * view;

        let model = Mat4::IDENTITY;
        let normal_matrix = Mat3::from_mat4(model).inverse().transpose();

        let uniforms = PreviewUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            normal_matrix: mat3_to_std140(normal_matrix),
            light_dir: [light_dir.x, light_dir.y, light_dir.z, 0.0],
            fallback_color: [0.7, 0.3, 0.8, 1.0],
            flags: [if has_texture { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };

        let bytes = bytemuck::bytes_of(&uniforms);
        let mapped: &mut [u8] = self
            .ctx
            .map_buffer_mut(self.uniform_buffer)
            .map_err(|_| "failed to map uniform buffer".to_string())?;
        mapped[..bytes.len()].copy_from_slice(bytes);
        self.ctx
            .unmap_buffer(self.uniform_buffer)
            .map_err(|_| "failed to unmap uniform buffer".to_string())?;
        Ok(())
    }

    fn readback(&mut self, image: &mut ColorImage) -> Result<(), String> {
        let mapped: &[u8] = self
            .ctx
            .map_buffer(self.target.readback)
            .map_err(|_| "failed to map readback buffer".to_string())?;
        let mut pixels = Vec::with_capacity(PREVIEW_WIDTH * PREVIEW_HEIGHT);
        for chunk in mapped.chunks_exact(4) {
            let color = Color32::from_rgba_unmultiplied(chunk[0], chunk[1], chunk[2], chunk[3]);
            pixels.push(color);
        }
        self.ctx
            .unmap_buffer(self.target.readback)
            .map_err(|_| "failed to unmap readback buffer".to_string())?;
        image.pixels = pixels;
        image.size = [PREVIEW_WIDTH, PREVIEW_HEIGHT];
        Ok(())
    }
}

struct PreviewTarget {
    color: Handle<dashi::Image>,
    color_view: ImageView,
    depth_view: ImageView,
    readback: Handle<dashi::Buffer>,
    viewport: Viewport,
}

impl PreviewTarget {
    fn new(ctx: &mut Context) -> Result<Self, String> {
        let color = ctx
            .make_image(&ImageInfo {
                debug_name: "preview_color",
                dim: [PREVIEW_WIDTH as u32, PREVIEW_HEIGHT as u32, 1],
                layers: 1,
                format: Format::RGBA8,
                mip_levels: 1,
                initial_data: None,
                ..Default::default()
            })
            .map_err(|_| "failed to create color target".to_string())?;
        let depth = ctx
            .make_image(&ImageInfo {
                debug_name: "preview_depth",
                dim: [PREVIEW_WIDTH as u32, PREVIEW_HEIGHT as u32, 1],
                layers: 1,
                format: Format::D24S8,
                mip_levels: 1,
                initial_data: None,
                ..Default::default()
            })
            .map_err(|_| "failed to create depth target".to_string())?;

        let color_view = ImageView {
            img: color,
            range: Default::default(),
            aspect: AspectMask::Color,
        };
        let depth_view = ImageView {
            img: depth,
            range: Default::default(),
            aspect: AspectMask::DepthStencil,
        };
        let readback = ctx
            .make_buffer(&BufferInfo {
                debug_name: "preview_readback",
                byte_size: (PREVIEW_WIDTH * PREVIEW_HEIGHT * 4) as u32,
                visibility: MemoryVisibility::CpuAndGpu,
                usage: BufferUsage::ALL,
                initial_data: None,
            })
            .map_err(|_| "failed to allocate readback buffer".to_string())?;

        let viewport = Viewport {
            area: dashi::FRect2D {
                x: 0.0,
                y: 0.0,
                w: PREVIEW_WIDTH as f32,
                h: PREVIEW_HEIGHT as f32,
            },
            scissor: Rect2D {
                x: 0,
                y: 0,
                w: PREVIEW_WIDTH as u32,
                h: PREVIEW_HEIGHT as u32,
            },
            min_depth: 0.0,
            max_depth: 1.0,
        };

        Ok(Self {
            color,
            color_view,
            depth_view,
            readback,
            viewport,
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
struct PreviewShaderSignature {
    key: String,
    vertex_hash: u64,
    fragment_hash: u64,
}

impl PreviewShaderSignature {
    fn from_modules(label: &str, vertex: &ShaderModule, fragment: &ShaderModule) -> Self {
        Self {
            key: label.to_string(),
            vertex_hash: hash_words(vertex.words()),
            fragment_hash: hash_words(fragment.words()),
        }
    }
}

struct PreviewPipeline {
    pipeline: Handle<dashi::GraphicsPipeline>,
}

impl PreviewPipeline {
    fn new(
        ctx: &mut Context,
        render_pass: Handle<dashi::RenderPass>,
        bind_group_layout: Handle<BindGroupLayout>,
        vertex: &ShaderModule,
        fragment: &ShaderModule,
        label: &str,
    ) -> Result<Self, String> {
        let vertex_info = VertexDescriptionInfo {
            entries: &VERTEX_ENTRIES,
            stride: std::mem::size_of::<Vertex>(),
            rate: VertexRate::Vertex,
        };

        let shaders = [
            PipelineShaderInfo {
                stage: ShaderType::Vertex,
                spirv: vertex.words(),
                specialization: &[],
            },
            PipelineShaderInfo {
                stage: ShaderType::Fragment,
                spirv: fragment.words(),
                specialization: &[],
            },
        ];

        let mut bg_layouts: [Option<Handle<BindGroupLayout>>; 4] = Default::default();
        bg_layouts[0] = Some(bind_group_layout);

        let layout = GraphicsPipelineLayoutInfo {
            debug_name: label,
            vertex_info,
            bg_layouts,
            bt_layouts: Default::default(),
            shaders: &shaders,
            details: GraphicsPipelineDetails {
                depth_test: Some(DepthInfo {
                    should_test: true,
                    should_write: true,
                }),
                ..Default::default()
            },
        };

        let pipeline_layout = ctx
            .make_graphics_pipeline_layout(&layout)
            .map_err(|_| "failed to build pipeline layout".to_string())?;
        let pipeline = ctx
            .make_graphics_pipeline(&GraphicsPipelineInfo {
                debug_name: label,
                layout: pipeline_layout,
                render_pass,
                subpass_id: 0,
            })
            .map_err(|_| "failed to build graphics pipeline".to_string())?;
        Ok(Self { pipeline })
    }
}

fn hash_words(words: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    words.hash(&mut hasher);
    hasher.finish()
}

const VERTEX_ENTRIES: [VertexEntryInfo; 5] = [
    VertexEntryInfo {
        format: ShaderPrimitiveType::Vec3,
        location: 0,
        offset: 0,
    },
    VertexEntryInfo {
        format: ShaderPrimitiveType::Vec3,
        location: 1,
        offset: 12,
    },
    VertexEntryInfo {
        format: ShaderPrimitiveType::Vec4,
        location: 2,
        offset: 24,
    },
    VertexEntryInfo {
        format: ShaderPrimitiveType::Vec2,
        location: 3,
        offset: 40,
    },
    VertexEntryInfo {
        format: ShaderPrimitiveType::Vec4,
        location: 4,
        offset: 48,
    },
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PreviewUniforms {
    view_proj: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 3],
    light_dir: [f32; 4],
    fallback_color: [f32; 4],
    flags: [f32; 4],
}

fn mat3_to_std140(matrix: Mat3) -> [[f32; 4]; 3] {
    let cols = matrix.to_cols_array_2d();
    [
        [cols[0][0], cols[0][1], cols[0][2], 0.0],
        [cols[1][0], cols[1][1], cols[1][2], 0.0],
        [cols[2][0], cols[2][1], cols[2][2], 0.0],
    ]
}

struct PreviewMeshCache {
    sphere: Option<GpuPreviewMesh>,
    quad: Option<GpuPreviewMesh>,
}

impl Default for PreviewMeshCache {
    fn default() -> Self {
        Self {
            sphere: None,
            quad: None,
        }
    }
}

impl PreviewMeshCache {
    fn mesh<'a>(&'a mut self, kind: PreviewMeshKind, gpu: &mut PreviewGpu) -> &'a GpuPreviewMesh {
        match kind {
            PreviewMeshKind::Sphere => {
                if self.sphere.is_none() {
                    self.sphere = Some(GpuPreviewMesh::new_sphere(&mut gpu.ctx).unwrap());
                }
                self.sphere.as_ref().unwrap()
            }
            PreviewMeshKind::Quad => {
                if self.quad.is_none() {
                    self.quad = Some(GpuPreviewMesh::new_quad(&mut gpu.ctx).unwrap());
                }
                self.quad.as_ref().unwrap()
            }
        }
    }
}

struct GpuPreviewMesh {
    vertex_buffer: Handle<dashi::Buffer>,
    index_buffer: Handle<dashi::Buffer>,
    index_count: u32,
}

impl GpuPreviewMesh {
    fn new_sphere(ctx: &mut Context) -> Result<Self, String> {
        let data = PreviewMesh::sphere();
        Self::upload(ctx, data)
    }

    fn new_quad(ctx: &mut Context) -> Result<Self, String> {
        let data = PreviewMesh::quad();
        Self::upload(ctx, data)
    }

    fn upload(ctx: &mut Context, mesh: PreviewMesh) -> Result<Self, String> {
        let vertices: Vec<Vertex> = mesh
            .vertices
            .into_iter()
            .map(|v| Vertex {
                position: [v.position.x, v.position.y, v.position.z],
                normal: [v.normal.x, v.normal.y, v.normal.z],
                tangent: [0.0, 0.0, 0.0, 1.0],
                uv: [v.uv.x, v.uv.y],
                color: [1.0, 1.0, 1.0, 1.0],
            })
            .collect();
        let vertex_bytes = bytemuck::cast_slice(&vertices);
        let vertex_buffer = ctx
            .make_buffer(&BufferInfo {
                debug_name: "preview_vertices",
                byte_size: vertex_bytes.len() as u32,
                visibility: MemoryVisibility::Gpu,
                usage: BufferUsage::VERTEX,
                initial_data: Some(vertex_bytes),
            })
            .map_err(|_| "failed to upload vertices".to_string())?;

        let index_bytes = bytemuck::cast_slice(&mesh.indices);
        let index_buffer = ctx
            .make_buffer(&BufferInfo {
                debug_name: "preview_indices",
                byte_size: index_bytes.len() as u32,
                visibility: MemoryVisibility::Gpu,
                usage: BufferUsage::INDEX,
                initial_data: Some(index_bytes),
            })
            .map_err(|_| "failed to upload indices".to_string())?;

        Ok(Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        })
    }
}

struct PreviewMesh {
    vertices: Vec<PreviewVertex>,
    indices: Vec<u32>,
}

impl PreviewMesh {
    fn sphere() -> Self {
        let stacks = 24;
        let slices = 32;
        let mut vertices = Vec::new();
        for stack in 0..=stacks {
            let v = stack as f32 / stacks as f32;
            let phi = v * PI;
            let y = phi.cos();
            let radius = phi.sin();
            for slice in 0..=slices {
                let u = slice as f32 / slices as f32;
                let theta = u * PI * 2.0;
                let x = radius * theta.cos();
                let z = radius * theta.sin();
                let normal = Vec3::new(x, y, z).normalize();
                vertices.push(PreviewVertex {
                    position: normal,
                    normal,
                    uv: glam::Vec2::new(u, 1.0 - v),
                });
            }
        }

        let mut indices = Vec::new();
        for stack in 0..stacks {
            for slice in 0..slices {
                let first = (stack * (slices + 1) + slice) as u32;
                let second = first + slices as u32 + 1;
                indices.push(first);
                indices.push(second);
                indices.push(first + 1);
                indices.push(second);
                indices.push(second + 1);
                indices.push(first + 1);
            }
        }

        Self { vertices, indices }
    }

    fn quad() -> Self {
        let vertices = vec![
            PreviewVertex {
                position: Vec3::new(-1.0, -1.0, 0.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                uv: glam::Vec2::new(0.0, 1.0),
            },
            PreviewVertex {
                position: Vec3::new(1.0, -1.0, 0.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                uv: glam::Vec2::new(1.0, 1.0),
            },
            PreviewVertex {
                position: Vec3::new(1.0, 1.0, 0.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                uv: glam::Vec2::new(1.0, 0.0),
            },
            PreviewVertex {
                position: Vec3::new(-1.0, 1.0, 0.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                uv: glam::Vec2::new(0.0, 0.0),
            },
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        Self { vertices, indices }
    }
}

struct PreviewVertex {
    position: Vec3,
    normal: Vec3,
    uv: glam::Vec2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PreviewMeshKind {
    Sphere,
    Quad,
}

impl PreviewMeshKind {
    const ALL: [Self; 2] = [Self::Sphere, Self::Quad];

    fn label(&self) -> &'static str {
        match self {
            Self::Sphere => "Sphere",
            Self::Quad => "Quad",
        }
    }
}

#[derive(Clone)]
struct PreviewTextureHandle {
    name: String,
    view: ImageView,
}

impl PreviewTextureHandle {
    fn solid_color(ctx: &mut Context, rgba: [u8; 4]) -> Result<Self, String> {
        let image = ctx
            .make_image(&ImageInfo {
                debug_name: "preview_fallback",
                dim: [1, 1, 1],
                layers: 1,
                format: Format::RGBA8,
                mip_levels: 1,
                initial_data: Some(&rgba),
                ..Default::default()
            })
            .map_err(|_| "failed to create fallback texture".to_string())?;
        Ok(Self {
            name: "fallback".into(),
            view: ImageView {
                img: image,
                range: Default::default(),
                aspect: AspectMask::Color,
            },
        })
    }
}

struct PreviewShaderModules {
    vertex: ShaderModule,
    fragment: ShaderModule,
}

struct PreviewAssetCache {
    project_root: PathBuf,
    layout: MaterialEditorDatabaseLayout,
    imagery: Option<ImageDB>,
    shaders: Option<ShaderDB>,
    leaks: HashMap<String, DatabaseEntry>,
    textures: HashMap<String, PreviewTextureHandle>,
}

impl PreviewAssetCache {
    fn new(root: &Path, layout: &MaterialEditorDatabaseLayout) -> Self {
        Self {
            project_root: root.to_path_buf(),
            layout: layout.clone(),
            imagery: None,
            shaders: None,
            leaks: HashMap::new(),
            textures: HashMap::new(),
        }
    }

    fn reset(&mut self, root: &Path, layout: &MaterialEditorDatabaseLayout) {
        self.project_root = root.to_path_buf();
        self.layout = layout.clone();
        self.imagery = None;
        self.shaders = None;
        self.leaks.clear();
        self.textures.clear();
    }

    fn ensure_imagery(
        &mut self,
        root: &Path,
        layout: &MaterialEditorDatabaseLayout,
        gpu: &mut PreviewGpu,
    ) {
        if self.project_root != root || self.layout_changed(layout) {
            self.reset(root, layout);
        }
        if self.imagery.is_none() {
            let path = self.project_root.join(&self.layout.imagery);
            if let Some(str_path) = path.to_str() {
                let ptr = gpu.ctx_ptr();
                self.imagery = Some(ImageDB::new(ptr, str_path));
            }
        }
    }

    fn ensure_shaders(&mut self, root: &Path, layout: &MaterialEditorDatabaseLayout) {
        if self.project_root != root || self.layout_changed(layout) {
            self.reset(root, layout);
        }
        if self.shaders.is_none() {
            let path = self.project_root.join(&self.layout.shaders);
            if let Some(str_path) = path.to_str() {
                self.shaders = Some(ShaderDB::new(str_path));
            }
        }
    }

    fn shader_modules(
        &mut self,
        shader_id: &str,
        shader: &MaterialEditorGraphicsShader,
    ) -> Result<PreviewShaderModules, String> {
        let vertex_entry = shader
            .vertex
            .as_deref()
            .ok_or_else(|| format!("Shader '{shader_id}' is missing a vertex stage entry"))?;
        let fragment_entry = shader
            .fragment
            .as_deref()
            .ok_or_else(|| format!("Shader '{shader_id}' is missing a fragment stage entry"))?;

        let vertex_handle = self.leak_entry(vertex_entry);
        let fragment_handle = self.leak_entry(fragment_entry);

        let shader_db = self
            .shaders
            .as_mut()
            .ok_or_else(|| "shader database unavailable".to_string())?;

        let vertex = shader_db
            .fetch_module(vertex_handle)
            .map_err(|err| format!("Failed to load vertex module for '{shader_id}': {err}"))?;
        let fragment = shader_db
            .fetch_module(fragment_handle)
            .map_err(|err| format!("Failed to load fragment module for '{shader_id}': {err}"))?;

        Ok(PreviewShaderModules { vertex, fragment })
    }

    fn texture(&mut self, entry: &str) -> Option<PreviewTextureHandle> {
        if let Some(handle) = self.textures.get(entry) {
            return Some(handle.clone());
        }
        let leaked = self.leak_entry(entry);
        let imagery = self.imagery.as_mut()?;
        let device = imagery.fetch_gpu_image(leaked).ok()?;
        let handle = PreviewTextureHandle {
            name: entry.to_string(),
            view: ImageView {
                img: device.img,
                range: Default::default(),
                aspect: AspectMask::Color,
            },
        };
        self.textures.insert(entry.to_string(), handle.clone());
        Some(handle)
    }

    fn leak_entry(&mut self, entry: &str) -> DatabaseEntry {
        if let Some(existing) = self.leaks.get(entry) {
            return *existing;
        }
        let leaked: DatabaseEntry = leak_database_entry(entry);
        self.leaks.insert(entry.to_string(), leaked);
        leaked
    }

    fn layout_changed(&self, layout: &MaterialEditorDatabaseLayout) -> bool {
        self.layout.geometry != layout.geometry
            || self.layout.imagery != layout.imagery
            || self.layout.models != layout.models
            || self.layout.materials != layout.materials
            || self.layout.render_passes != layout.render_passes
            || self.layout.shaders != layout.shaders
    }
}

#[derive(Clone, Copy)]
struct PreviewConfig {
    mesh_kind: PreviewMeshKind,
    camera: OrbitCamera,
    background_rgb: [f32; 3],
    wireframe: bool,
}

impl Default for PreviewConfig {
    fn default() -> Self {
        Self {
            mesh_kind: PreviewMeshKind::Sphere,
            camera: OrbitCamera::default(),
            background_rgb: [0.12, 0.12, 0.12],
            wireframe: false,
        }
    }
}

#[derive(Clone, Copy)]
struct OrbitCamera {
    azimuth: f32,
    elevation: f32,
    distance: f32,
}

impl OrbitCamera {
    fn view_matrix(&self) -> Mat4 {
        let yaw = self.azimuth.to_radians();
        let pitch = self.elevation.to_radians();
        let cos_pitch = pitch.cos();
        let position =
            Vec3::new(yaw.sin() * cos_pitch, pitch.sin(), yaw.cos() * cos_pitch) * self.distance;
        Mat4::look_at_rh(position, Vec3::ZERO, Vec3::Y)
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            azimuth: 45.0,
            elevation: 25.0,
            distance: 2.5,
        }
    }
}

struct BuiltinShader {
    vertex: ShaderModule,
    fragment: ShaderModule,
}

fn builtin_shader_modules() -> Result<BuiltinShader, String> {
    let vertex_src = include_str!("../shaders/preview.vert.glsl");
    let fragment_src = include_str!("../shaders/preview.frag.glsl");
    let compiler = Compiler::new().ok_or_else(|| "shader compiler unavailable".to_string())?;
    let vertex = compiler
        .compile_into_spirv(vertex_src, ShaderKind::Vertex, "preview.vert", "main", None)
        .map_err(|err| format!("failed to compile preview vertex shader: {err}"))?;
    let fragment = compiler
        .compile_into_spirv(
            fragment_src,
            ShaderKind::Fragment,
            "preview.frag",
            "main",
            None,
        )
        .map_err(|err| format!("failed to compile preview fragment shader: {err}"))?;
    Ok(BuiltinShader {
        vertex: ShaderModule::from_words(vertex.as_binary().to_vec()),
        fragment: ShaderModule::from_words(fragment.as_binary().to_vec()),
    })
}

fn create_bind_group_layout(ctx: &mut Context) -> Result<Handle<BindGroupLayout>, String> {
    let uniform_vars = [BindGroupVariable {
        var_type: BindGroupVariableType::Uniform,
        binding: 0,
        count: 1,
    }];
    let sampler_vars = [BindGroupVariable {
        var_type: BindGroupVariableType::SampledImage,
        binding: 1,
        count: 1,
    }];
    let shader_info = [
        dashi::ShaderInfo {
            shader_type: ShaderType::All,
            variables: &uniform_vars,
        },
        dashi::ShaderInfo {
            shader_type: ShaderType::Fragment,
            variables: &sampler_vars,
        },
    ];
    ctx.make_bind_group_layout(&BindGroupLayoutInfo {
        debug_name: "preview_bind_group",
        shaders: &shader_info,
    })
    .map_err(|err| format!("failed to create bind group layout: {err}"))
}

fn create_bind_group(
    ctx: &mut Context,
    layout: Handle<BindGroupLayout>,
    uniform_buffer: Handle<dashi::Buffer>,
    sampler: Handle<dashi::Sampler>,
    view: ImageView,
) -> Result<Handle<dashi::BindGroup>, String> {
    let bindings = [
        dashi::BindingInfo {
            resource: dashi::ShaderResource::ConstBuffer(dashi::BufferView::new(uniform_buffer)),
            binding: 0,
        },
        dashi::BindingInfo {
            resource: dashi::ShaderResource::SampledImage(view, sampler),
            binding: 1,
        },
    ];
    ctx.make_bind_group(&dashi::BindGroupInfo {
        debug_name: "preview_bind_group",
        layout,
        bindings: &bindings,
        set: 0,
    })
    .map_err(|err| format!("failed to create bind group: {err}"))
}
