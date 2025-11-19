use std::{
    collections::HashMap,
    f32::consts::PI,
    hash::{Hash, Hasher},
    path::PathBuf,
};

use bytemuck::{Pod, Zeroable};
use dashi::gpu::execution::CommandRing;
use dashi::{
    AspectMask, AttachmentDescription, BindGroupLayout, BindGroupLayoutInfo, BindGroupVariable,
    BindGroupVariableType, BindTableLayout, BindTableLayoutInfo, BufferInfo, BufferUsage,
    ClearValue, CommandQueueInfo2, Context, ContextInfo, DepthInfo, Format,
    GraphicsPipelineDetails, GraphicsPipelineLayoutInfo, Handle, ImageInfo, ImageView,
    MemoryVisibility, PipelineShaderInfo, SampleCount, ShaderPrimitiveType, ShaderType,
    VertexDescriptionInfo, VertexEntryInfo, VertexRate, Viewport,
    driver::command::{BeginRenderPass, CopyImageBuffer, DrawIndexed},
    gpu::CommandStream,
};
use eframe::egui::{self, Color32, ColorImage};
use glam::{Mat3, Mat4, Vec3};
use noren::datatypes::{
    DatabaseEntry, ImageDB, ShaderDB, ShaderModule, leak_database_entry, primitives::Vertex,
    render_pass::RenderPassDB,
};
use shaderc::{Compiler, ShaderKind};

use noren::parsing::{GraphicsShaderLayout, RenderPassLayout};

use crate::{
    material_bindings::{TextureBindingKind, TextureBindingSlot, texture_binding_slots},
    material_editor::project::{GraphTexture, MaterialEditorProjectState},
    material_editor_types::{
        MaterialEditorDatabaseLayout, MaterialEditorGraphicsShader, MaterialEditorMaterial,
    },
};

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
                    let [width_px, height_px] = self.renderer.preview_size;
                    let aspect = height_px as f32 / width_px as f32;
                    let width = ui.available_width().min(width_px as f32);
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
    preview_size: [usize; 2],
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
            assets: PreviewAssetCache::new(state),
            image: ColorImage::new([320, 240], Color32::BLACK),
            preview_size: [320, 240],
            gpu,
            gpu_error,
        }
    }

    fn sync_with_state(&mut self, state: &MaterialEditorProjectState) {
        self.assets.reset(state);
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
            self.assets.ensure_render_passes(state);
            self.assets.ensure_imagery(state, gpu_for_assets);
        }
        self.assets.ensure_shaders(state);

        let mut shader_layout: Option<GraphicsShaderLayout> = None;
        let mut render_pass_layout: Option<RenderPassLayout> = None;
        let mut render_pass_key: Option<String> = None;
        let mut shader_modules: Option<PreviewShaderModules> = None;
        let mut shader_label = "builtin_preview".to_string();

        if let Some(shader_id) = material.shader.as_deref() {
            shader_label = shader_id.to_string();
            match state.graph.shaders.get(shader_id) {
                Some(shader) => {
                    let layout: GraphicsShaderLayout = shader.resource.data.clone().into();
                    render_pass_key = layout.render_pass.clone();
                    if let Some(pass_key) = &render_pass_key {
                        if let Some(pass_layout) = self.assets.render_pass_layouts.get(pass_key) {
                            render_pass_layout = Some(pass_layout.clone());
                        } else {
                            warnings.push(format!("Render pass '{pass_key}' is missing"));
                        }
                    }
                    shader_layout = Some(layout);
                    match self.assets.shader_modules(shader_id, &shader.resource.data) {
                        Ok(mods) => shader_modules = Some(mods),
                        Err(err) => warnings.push(err),
                    }
                }
                None => warnings.push(format!("Shader '{shader_id}' is missing")),
            }
        }

        let resolved =
            match self.resolve_textures(shader_layout.as_ref(), material, state, &mut warnings) {
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
            shader_layout.as_ref(),
            render_pass_key,
            render_pass_layout.as_ref(),
            &shader_label,
            shader_modules
                .as_ref()
                .map(|modules| (&modules.vertex, &modules.fragment)),
            &mut self.assets,
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

        self.preview_size = self.image.size;

        PreviewResult {
            warnings,
            image_changed: true,
        }
    }

    fn resolve_textures(
        &mut self,
        layout: Option<&GraphicsShaderLayout>,
        material: &MaterialEditorMaterial,
        state: &MaterialEditorProjectState,
        warnings: &mut Vec<String>,
    ) -> Option<ResolvedTextureBindings> {
        let layout = layout?;
        let slots = texture_binding_slots(layout);
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
    target: Option<PreviewTarget>,
    target_signature: Option<RenderTargetSignature>,
    pipeline: Option<PreviewPipeline>,
    pipeline_signature: Option<PreviewShaderSignature>,
    sampler: Handle<dashi::Sampler>,
    uniform_buffer: Handle<dashi::Buffer>,
    fallback_texture: PreviewTextureHandle,
    bind_groups: HashMap<String, Handle<dashi::BindGroup>>,
    builtin_shader: BuiltinShader,
    fallback_bind_group_layout: Handle<BindGroupLayout>,
    bind_group_layouts: HashMap<u64, Handle<BindGroupLayout>>,
    bind_table_layouts: HashMap<u64, Handle<BindTableLayout>>,
    builtin_render_pass: RenderPassLayout,
    builtin_render_pass_db: RenderPassDB,
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

        let fallback_bind_group_layout = create_bind_group_layout(&mut ctx)?;
        let fallback_texture = PreviewTextureHandle::solid_color(&mut ctx, [200, 120, 240, 255])?;
        let bind_group = create_bind_group(
            &mut ctx,
            fallback_bind_group_layout,
            uniform_buffer,
            sampler,
            fallback_texture.view,
        )?;
        let mut bind_groups = HashMap::new();
        bind_groups.insert("fallback".to_string(), bind_group);

        let builtin_shader = builtin_shader_modules()?;
        let builtin_render_pass = builtin_render_pass_layout();
        let mut render_passes = HashMap::new();
        render_passes.insert("builtin_preview".to_string(), builtin_render_pass.clone());
        let builtin_render_pass_db = RenderPassDB::new(render_passes);

        Ok(Self {
            ctx,
            ring,
            target: None,
            target_signature: None,
            pipeline: None,
            pipeline_signature: None,
            sampler,
            uniform_buffer,
            fallback_texture,
            bind_groups,
            builtin_shader,
            fallback_bind_group_layout,
            bind_group_layouts: HashMap::new(),
            bind_table_layouts: HashMap::new(),
            builtin_render_pass,
            builtin_render_pass_db,
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
        shader_layout: Option<&GraphicsShaderLayout>,
        render_pass_key: Option<String>,
        render_pass_layout: Option<&RenderPassLayout>,
        shader_label: &str,
        shader_modules: Option<(&ShaderModule, &ShaderModule)>,
        assets: &mut PreviewAssetCache,
    ) -> Result<(), String> {
        let render_pass_key = render_pass_key.unwrap_or_else(|| "builtin_preview".to_string());
        let render_pass_layout = render_pass_layout
            .cloned()
            .unwrap_or_else(|| self.builtin_render_pass.clone());
        self.ensure_pipeline(
            shader_label,
            shader_layout,
            &render_pass_key,
            &render_pass_layout,
            shader_modules,
            assets,
        )?;
        self.ensure_target(&render_pass_key, &render_pass_layout)?;
        let pipeline_snapshot = {
            let pipeline = self
                .pipeline
                .as_ref()
                .ok_or_else(|| "preview pipeline unavailable".to_string())?;
            (
                pipeline.pipeline,
                pipeline.bind_group_layout,
                pipeline.layout_hash,
                pipeline.render_pass,
                pipeline.viewport,
            )
        };
        let has_texture = texture.is_some();
        let target_snapshot = {
            let target = self
                .target
                .as_ref()
                .ok_or_else(|| "preview target unavailable".to_string())?;
            let color_image = *target
                .color_images
                .first()
                .ok_or_else(|| "preview target missing color image".to_string())?;
            (
                color_image,
                target.readback,
                target.color_views,
                target.depth_view,
                target.size,
                target.format,
            )
        };
        let (color_image, readback, color_views, depth_view, target_size, target_format) =
            target_snapshot;
        let (pipeline_handle, bind_group_layout, layout_hash, render_pass_handle, viewport) =
            pipeline_snapshot;
        self.update_uniforms(config, light_dir, has_texture, target_size)?;
        let resolved_texture = match texture {
            Some(handle) => handle,
            None => self.fallback_texture.clone(),
        };
        let bind_group = self.bind_group_for(&resolved_texture, bind_group_layout, layout_hash)?;

        let clear = ClearValue::Color([
            background.r() as f32 / 255.0,
            background.g() as f32 / 255.0,
            background.b() as f32 / 255.0,
            1.0,
        ]);

        let depthclear = depth_view.map(|_| ClearValue::DepthStencil {
            depth: 1.0,
            stencil: 0,
        });

        self.ring
            .record(|cmd| {
                let stream = CommandStream::new().begin();
                let begin_pass = BeginRenderPass {
                    viewport,
                    render_pass: render_pass_handle,
                    color_attachments: color_views,
                    depth_attachment: depth_view,
                    clear_values: [Some(clear), depthclear, None, None],
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
                    src: color_image,
                    dst: readback,
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

        self.readback(readback, target_size, target_format, image)?;
        Ok(())
    }

    fn ctx_ptr(&mut self) -> *mut Context {
        &mut self.ctx as *mut _
    }

    fn ensure_pipeline(
        &mut self,
        shader_label: &str,
        shader_layout: Option<&GraphicsShaderLayout>,
        render_pass_key: &str,
        render_pass_layout: &RenderPassLayout,
        shader_modules: Option<(&ShaderModule, &ShaderModule)>,
        assets: &mut PreviewAssetCache,
    ) -> Result<(), String> {
        let (vertex, fragment, label) = match shader_modules {
            Some((vertex, fragment)) => {
                (vertex.clone(), fragment.clone(), shader_label.to_string())
            }
            None => (
                self.builtin_shader.vertex.clone(),
                self.builtin_shader.fragment.clone(),
                "builtin_preview".to_string(),
            ),
        };

        let layout_hash = hash_graphics_layout(shader_layout);
        let signature = PreviewShaderSignature::from_modules(
            &label,
            &vertex,
            &fragment,
            render_pass_key,
            layout_hash,
        );
        if self.pipeline_signature.as_ref() != Some(&signature) {
            let (bg_layouts, bt_layouts, active_bind_group) =
                self.build_layout_handles(shader_layout)?;
            let vertex_info = VertexDescriptionInfo {
                entries: &VERTEX_ENTRIES,
                stride: std::mem::size_of::<Vertex>(),
                rate: VertexRate::Vertex,
            };

            let mut details = GraphicsPipelineDetails::default();
            let subpass_id = shader_layout.map(|layout| layout.subpass).unwrap_or(0);
            if render_pass_layout
                .subpasses
                .get(subpass_id as usize)
                .and_then(|subpass| subpass.depth_stencil_attachment.as_ref())
                .is_some()
            {
                details.depth_test = Some(DepthInfo {
                    should_test: true,
                    should_write: true,
                });
            }

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

            let layout_info = GraphicsPipelineLayoutInfo {
                debug_name: label.as_str(),
                vertex_info,
                bg_layouts,
                bt_layouts,
                shaders: &shaders,
                details,
            };

            let pipeline_layout = self
                .ctx
                .make_graphics_pipeline_layout(&layout_info)
                .map_err(|_| "failed to build pipeline layout".to_string())?;

            let pass_db = if render_pass_key == "builtin_preview" {
                &mut self.builtin_render_pass_db
            } else {
                assets
                    .render_passes
                    .as_mut()
                    .ok_or_else(|| "render pass database unavailable".to_string())?
            };

            let pipeline_info = pass_db
                .pipeline_info(
                    render_pass_key,
                    subpass_id,
                    pipeline_layout,
                    label.as_str(),
                    &mut self.ctx,
                )
                .map_err(|err| format!("failed to prepare pipeline info: {err}"))?;

            let pipeline = self
                .ctx
                .make_graphics_pipeline(&pipeline_info)
                .map_err(|_| "failed to build graphics pipeline".to_string())?;

            self.pipeline = Some(PreviewPipeline {
                pipeline,
                render_pass: pipeline_info.render_pass,
                viewport: render_pass_layout.viewport,
                bind_group_layout: active_bind_group,
                layout_hash,
            });
            self.pipeline_signature = Some(signature);
        }
        Ok(())
    }

    fn build_layout_handles(
        &mut self,
        layout: Option<&GraphicsShaderLayout>,
    ) -> Result<
        (
            [Option<Handle<BindGroupLayout>>; 4],
            [Option<Handle<BindTableLayout>>; 4],
            Handle<BindGroupLayout>,
        ),
        String,
    > {
        let mut bg_layouts: [Option<Handle<BindGroupLayout>>; 4] = Default::default();
        let mut bt_layouts: [Option<Handle<BindTableLayout>>; 4] = Default::default();
        let mut active = self.fallback_bind_group_layout;

        if let Some(layout) = layout {
            for (idx, cfg_opt) in layout.bind_group_layouts.iter().enumerate().take(4) {
                if let Some(cfg) = cfg_opt {
                    let (handle, _) = self.bind_group_layout_handle(cfg)?;
                    if idx == 0 {
                        active = handle;
                    }
                    bg_layouts[idx] = Some(handle);
                }
            }

            for (idx, cfg_opt) in layout.bind_table_layouts.iter().enumerate().take(4) {
                if let Some(cfg) = cfg_opt {
                    let handle = self.bind_table_layout_handle(cfg)?;
                    bt_layouts[idx] = Some(handle);
                }
            }
        }

        if bg_layouts[0].is_none() {
            bg_layouts[0] = Some(self.fallback_bind_group_layout);
            active = self.fallback_bind_group_layout;
        }

        Ok((bg_layouts, bt_layouts, active))
    }

    fn bind_group_layout_handle(
        &mut self,
        cfg: &dashi::cfg::BindGroupLayoutCfg,
    ) -> Result<(Handle<BindGroupLayout>, u64), String> {
        let hash = hash_json(cfg)?;
        if let Some(existing) = self.bind_group_layouts.get(&hash) {
            return Ok((*existing, hash));
        }
        let borrowed = cfg.borrow();
        let info: BindGroupLayoutInfo<'_> = borrowed.info();
        let handle = self
            .ctx
            .make_bind_group_layout(&info)
            .map_err(|err| format!("failed to create bind group layout: {err}"))?;
        self.bind_group_layouts.insert(hash, handle);
        Ok((handle, hash))
    }

    fn bind_table_layout_handle(
        &mut self,
        cfg: &dashi::cfg::BindTableLayoutCfg,
    ) -> Result<Handle<BindTableLayout>, String> {
        let hash = hash_json(cfg)?;
        if let Some(existing) = self.bind_table_layouts.get(&hash) {
            return Ok(*existing);
        }
        let borrowed = cfg.borrow();
        let info: BindTableLayoutInfo<'_> = borrowed.info();
        let handle = self
            .ctx
            .make_bind_table_layout(&info)
            .map_err(|err| format!("failed to create bind table layout: {err}"))?;
        self.bind_table_layouts.insert(hash, handle);
        Ok(handle)
    }

    fn bind_group_for(
        &mut self,
        texture: &PreviewTextureHandle,
        layout: Handle<BindGroupLayout>,
        layout_hash: u64,
    ) -> Result<Handle<dashi::BindGroup>, String> {
        let key = format!("{}:{layout_hash}", texture.name);
        if let Some(handle) = self.bind_groups.get(&key) {
            return Ok(*handle);
        }
        let handle = create_bind_group(
            &mut self.ctx,
            layout,
            self.uniform_buffer,
            self.sampler,
            texture.view,
        )?;
        self.bind_groups.insert(key, handle);
        Ok(handle)
    }

    fn ensure_target(
        &mut self,
        render_pass_key: &str,
        render_pass_layout: &RenderPassLayout,
    ) -> Result<(), String> {
        let signature = RenderTargetSignature::from_layout(render_pass_key, render_pass_layout)?;
        if self.target_signature.as_ref() != Some(&signature) {
            let target = PreviewTarget::new(&mut self.ctx, &signature, render_pass_layout)?;
            self.target = Some(target);
            self.target_signature = Some(signature);
        }
        Ok(())
    }

    fn update_uniforms(
        &mut self,
        config: &PreviewConfig,
        light_dir: Vec3,
        has_texture: bool,
        target_size: [u32; 2],
    ) -> Result<(), String> {
        let view = config.camera.view_matrix();
        let aspect = target_size[0] as f32 / target_size[1].max(1) as f32;
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

    fn readback(
        &mut self,
        readback: Handle<dashi::Buffer>,
        size: [u32; 2],
        format: Format,
        image: &mut ColorImage,
    ) -> Result<(), String> {
        let bpp = bytes_per_pixel(format)
            .ok_or_else(|| format!("Unsupported preview format {:?}", format))?;
        let mapped: &[u8] = self
            .ctx
            .map_buffer(readback)
            .map_err(|_| "failed to map readback buffer".to_string())?;
        let mut pixels = Vec::with_capacity((size[0] * size[1]) as usize);
        for chunk in mapped.chunks_exact(bpp) {
            let color = decode_pixel(format, chunk)?;
            pixels.push(color);
        }
        self.ctx
            .unmap_buffer(readback)
            .map_err(|_| "failed to unmap readback buffer".to_string())?;
        image.pixels = pixels;
        image.size = [size[0] as usize, size[1] as usize];
        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq)]
struct RenderTargetSignature {
    render_pass: String,
    width: u32,
    height: u32,
    color_format: Format,
    depth_format: Option<Format>,
    samples: SampleCount,
}

impl RenderTargetSignature {
    fn from_layout(render_pass_key: &str, layout: &RenderPassLayout) -> Result<Self, String> {
        let subpass = layout
            .subpasses
            .first()
            .ok_or_else(|| "render pass has no subpasses".to_string())?;
        let color_attachment = subpass
            .color_attachments
            .first()
            .ok_or_else(|| "render pass has no color attachments".to_string())?;
        let width = layout.viewport.area.w.max(1.0) as u32;
        let height = layout.viewport.area.h.max(1.0) as u32;

        Ok(Self {
            render_pass: render_pass_key.to_string(),
            width,
            height,
            color_format: color_attachment.format,
            depth_format: subpass.depth_stencil_attachment.as_ref().map(|d| d.format),
            samples: color_attachment.samples,
        })
    }
}

struct PreviewTarget {
    color_images: Vec<Handle<dashi::Image>>,
    color_views: [Option<ImageView>; 4],
    _depth_image: Option<Handle<dashi::Image>>,
    depth_view: Option<ImageView>,
    readback: Handle<dashi::Buffer>,
    _viewport: Viewport,
    size: [u32; 2],
    format: Format,
    _samples: SampleCount,
}

impl PreviewTarget {
    fn new(
        ctx: &mut Context,
        signature: &RenderTargetSignature,
        layout: &RenderPassLayout,
    ) -> Result<Self, String> {
        if signature.samples != SampleCount::S1 {
            return Err("Preview only supports sample count 1".to_string());
        }

        if bytes_per_pixel(signature.color_format).is_none() {
            return Err(format!(
                "Unsupported color format {:?} for preview",
                signature.color_format
            ));
        }

        let mut color_views: [Option<ImageView>; 4] = Default::default();
        let mut color_images = Vec::new();
        let subpass = layout
            .subpasses
            .first()
            .ok_or_else(|| "render pass has no subpasses".to_string())?;

        for (idx, attachment) in subpass.color_attachments.iter().take(4).enumerate() {
            let img = ctx
                .make_image(&ImageInfo {
                    debug_name: "preview_color",
                    dim: [signature.width, signature.height, 1],
                    layers: 1,
                    format: attachment.format,
                    mip_levels: 1,
                    samples: attachment.samples,
                    initial_data: None,
                })
                .map_err(|_| "failed to create color target".to_string())?;
            let view = ImageView {
                img,
                range: Default::default(),
                aspect: AspectMask::Color,
            };
            color_images.push(img);
            color_views[idx] = Some(view);
        }

        let depth_view = if let Some(depth_attachment) = &subpass.depth_stencil_attachment {
            let img = ctx
                .make_image(&ImageInfo {
                    debug_name: "preview_depth",
                    dim: [signature.width, signature.height, 1],
                    layers: 1,
                    format: depth_attachment.format,
                    mip_levels: 1,
                    samples: depth_attachment.samples,
                    initial_data: None,
                })
                .map_err(|_| "failed to create depth target".to_string())?;
            Some(ImageView {
                img,
                range: Default::default(),
                aspect: AspectMask::DepthStencil,
            })
        } else {
            None
        };

        let depth_image = depth_view.map(|view| view.img);

        let readback = ctx
            .make_buffer(&BufferInfo {
                debug_name: "preview_readback",
                byte_size: (signature.width
                    * signature.height
                    * bytes_per_pixel(signature.color_format).unwrap() as u32),
                visibility: MemoryVisibility::CpuAndGpu,
                usage: BufferUsage::ALL,
                initial_data: None,
            })
            .map_err(|_| "failed to allocate readback buffer".to_string())?;

        let viewport = Viewport {
            area: layout.viewport.area,
            scissor: layout.viewport.scissor,
            min_depth: layout.viewport.min_depth,
            max_depth: layout.viewport.max_depth,
        };

        Ok(Self {
            color_images,
            color_views,
            _depth_image: depth_image,
            depth_view,
            readback,
            _viewport: viewport,
            size: [signature.width, signature.height],
            format: signature.color_format,
            _samples: signature.samples,
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
struct PreviewShaderSignature {
    key: String,
    vertex_hash: u64,
    fragment_hash: u64,
    render_pass: String,
    layout_hash: u64,
}

impl PreviewShaderSignature {
    fn from_modules(
        label: &str,
        vertex: &ShaderModule,
        fragment: &ShaderModule,
        render_pass: &str,
        layout_hash: u64,
    ) -> Self {
        Self {
            key: label.to_string(),
            vertex_hash: hash_words(vertex.words()),
            fragment_hash: hash_words(fragment.words()),
            render_pass: render_pass.to_string(),
            layout_hash,
        }
    }
}

struct PreviewPipeline {
    pipeline: Handle<dashi::GraphicsPipeline>,
    render_pass: Handle<dashi::RenderPass>,
    viewport: Viewport,
    bind_group_layout: Handle<BindGroupLayout>,
    layout_hash: u64,
}

fn hash_graphics_layout(layout: Option<&GraphicsShaderLayout>) -> u64 {
    layout
        .and_then(|layout| hash_json(layout).ok())
        .unwrap_or_default()
}

fn hash_json<T: serde::Serialize>(value: &T) -> Result<u64, String> {
    let serialized = serde_json::to_string(value).map_err(|err| err.to_string())?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    serialized.hash(&mut hasher);
    Ok(hasher.finish())
}

fn bytes_per_pixel(format: Format) -> Option<usize> {
    match format {
        Format::RGBA8 | Format::RGBA8Unorm | Format::BGRA8Unorm => Some(4),
        _ => None,
    }
}

fn decode_pixel(format: Format, bytes: &[u8]) -> Result<Color32, String> {
    match format {
        Format::RGBA8 | Format::RGBA8Unorm => Ok(Color32::from_rgba_unmultiplied(
            bytes[0], bytes[1], bytes[2], bytes[3],
        )),
        Format::BGRA8Unorm => Ok(Color32::from_rgba_unmultiplied(
            bytes[2], bytes[1], bytes[0], bytes[3],
        )),
        other => Err(format!("Unsupported preview format {:?}", other)),
    }
}

fn builtin_render_pass_layout() -> RenderPassLayout {
    use noren::parsing::RenderSubpassLayout;

    let color_attachment = AttachmentDescription {
        format: Format::RGBA8,
        ..Default::default()
    };
    let depth_attachment = AttachmentDescription {
        format: Format::D24S8,
        ..Default::default()
    };

    RenderPassLayout {
        debug_name: Some("builtin_preview".to_string()),
        viewport: Viewport {
            area: dashi::FRect2D {
                x: 0.0,
                y: 0.0,
                w: 320.0,
                h: 240.0,
            },
            scissor: dashi::Rect2D {
                x: 0,
                y: 0,
                w: 320,
                h: 240,
            },
            min_depth: 0.0,
            max_depth: 1.0,
        },
        subpasses: vec![RenderSubpassLayout {
            color_attachments: vec![color_attachment],
            depth_stencil_attachment: Some(depth_attachment),
            subpass_dependencies: Vec::new(),
        }],
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
    render_passes: Option<RenderPassDB>,
    render_pass_layouts: HashMap<String, RenderPassLayout>,
    render_pass_count: usize,
}

impl PreviewAssetCache {
    fn new(state: &MaterialEditorProjectState) -> Self {
        Self {
            project_root: state.root().to_path_buf(),
            layout: state.layout.clone(),
            imagery: None,
            shaders: None,
            leaks: HashMap::new(),
            textures: HashMap::new(),
            render_passes: None,
            render_pass_layouts: HashMap::new(),
            render_pass_count: 0,
        }
    }

    fn reset(&mut self, state: &MaterialEditorProjectState) {
        self.project_root = state.root().to_path_buf();
        self.layout = state.layout.clone();
        self.imagery = None;
        self.shaders = None;
        self.leaks.clear();
        self.textures.clear();
        self.render_passes = None;
        self.render_pass_layouts.clear();
        self.render_pass_count = 0;
    }

    fn ensure_imagery(&mut self, state: &MaterialEditorProjectState, gpu: &mut PreviewGpu) {
        if self.project_root != state.root() || self.layout_changed(&state.layout) {
            self.reset(state);
        }
        if self.imagery.is_none() {
            let path = self.project_root.join(&self.layout.imagery);
            if let Some(str_path) = path.to_str() {
                let ptr = gpu.ctx_ptr();
                self.imagery = Some(ImageDB::new(ptr, str_path));
            }
        }
    }

    fn ensure_shaders(&mut self, state: &MaterialEditorProjectState) {
        if self.project_root != state.root() || self.layout_changed(&state.layout) {
            self.reset(state);
        }
        if self.shaders.is_none() {
            let path = self.project_root.join(&self.layout.shaders);
            if let Some(str_path) = path.to_str() {
                self.shaders = Some(ShaderDB::new(str_path));
            }
        }
    }

    fn ensure_render_passes(&mut self, state: &MaterialEditorProjectState) {
        if self.project_root != state.root() || self.layout_changed(&state.layout) {
            self.reset(state);
        }

        if self.render_passes.is_some() && self.render_pass_count == state.graph.render_passes.len()
        {
            return;
        }

        self.render_pass_layouts = state
            .graph
            .render_passes
            .iter()
            .map(|(id, pass)| (id.clone(), pass.resource.data.clone().into()))
            .collect();
        self.render_pass_count = state.graph.render_passes.len();
        self.render_passes = Some(RenderPassDB::new(self.render_pass_layouts.clone()));
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
