//! GUI-friendly types for editing material databases.
//!
//! ## Editable structures
//! The GUI must be able to read and write the following logical groups that are
//! serialized into the JSON layout files:
//! - **Database paths** (`MaterialEditorDatabaseLayout`) describing where
//!   geometry/imagery/material manifests are stored.
//! - **Texture, material, mesh, model, and shader manifests**
//!   (`MaterialEditorProject`). These mirror [`ModelLayoutFile`] entries and are
//!   responsible for associating named resources with the binary payloads stored
//!   in the RDB files.
//! - **Render passes and subpasses** (`MaterialEditorRenderPassFile`) which wrap
//!   [`RenderPassLayoutFile`].
//!
//! ## Compatibility guarantee
//! These types *only* provide serialization helpers and conversion traits to the
//! canonical database layouts defined in [`crate::parsing`]. The core database
//! code never depends on GUI-specific structs, keeping the DB crate
//! self-sufficient. Future GUI tooling can evolve independently while the
//! conversion traits make sure data continues to be exchanged via the stable
//! parsing structures.

use std::collections::HashMap;

use noren::parsing::{
    DatabaseLayoutFile, GraphicsShaderLayout, MaterialLayout, MeshLayout, ModelLayout,
    ModelLayoutFile, RenderPassLayout, RenderPassLayoutFile, RenderSubpassLayout, TextureLayout,
};
use dashi::{AttachmentDescription, SubpassDependency, Viewport, cfg};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorDatabaseLayout {
    pub geometry: String,
    pub imagery: String,
    pub models: String,
    pub materials: String,
    pub render_passes: String,
    pub shaders: String,
}

impl MaterialEditorDatabaseLayout {
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn to_json_string_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

impl From<DatabaseLayoutFile> for MaterialEditorDatabaseLayout {
    fn from(value: DatabaseLayoutFile) -> Self {
        Self {
            geometry: value.geometry,
            imagery: value.imagery,
            models: value.models,
            materials: value.materials,
            render_passes: value.render_passes,
            shaders: value.shaders,
        }
    }
}

impl From<MaterialEditorDatabaseLayout> for DatabaseLayoutFile {
    fn from(value: MaterialEditorDatabaseLayout) -> Self {
        Self {
            geometry: value.geometry,
            imagery: value.imagery,
            models: value.models,
            materials: value.materials,
            render_passes: value.render_passes,
            shaders: value.shaders,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorProject {
    pub textures: HashMap<String, MaterialEditorTexture>,
    pub materials: HashMap<String, MaterialEditorMaterial>,
    pub meshes: HashMap<String, MaterialEditorMesh>,
    pub models: HashMap<String, MaterialEditorModel>,
    pub shaders: HashMap<String, MaterialEditorGraphicsShader>,
    pub render_passes: HashMap<String, MaterialEditorRenderPass>,
}

impl MaterialEditorProject {
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn to_json_string_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

impl From<ModelLayoutFile> for MaterialEditorProject {
    fn from(value: ModelLayoutFile) -> Self {
        Self {
            textures: value
                .textures
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            materials: value
                .materials
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            meshes: value
                .meshes
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            models: value
                .models
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            shaders: value
                .shaders
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            render_passes: value
                .render_passes
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
        }
    }
}

impl From<MaterialEditorProject> for ModelLayoutFile {
    fn from(value: MaterialEditorProject) -> Self {
        Self {
            textures: value
                .textures
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            materials: value
                .materials
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            meshes: value
                .meshes
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            models: value
                .models
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            shaders: value
                .shaders
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
            render_passes: value
                .render_passes
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorRenderPassFile {
    pub render_passes: HashMap<String, MaterialEditorRenderPass>,
}

impl MaterialEditorRenderPassFile {
    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn to_json_string_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

impl From<RenderPassLayoutFile> for MaterialEditorRenderPassFile {
    fn from(value: RenderPassLayoutFile) -> Self {
        Self {
            render_passes: value
                .render_passes
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
        }
    }
}

impl From<MaterialEditorRenderPassFile> for RenderPassLayoutFile {
    fn from(value: MaterialEditorRenderPassFile) -> Self {
        Self {
            render_passes: value
                .render_passes
                .into_iter()
                .map(|(key, layout)| (key, layout.into()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorTexture {
    pub image: String,
    pub name: Option<String>,
}

impl From<TextureLayout> for MaterialEditorTexture {
    fn from(value: TextureLayout) -> Self {
        Self {
            image: value.image,
            name: value.name,
        }
    }
}

impl From<MaterialEditorTexture> for TextureLayout {
    fn from(value: MaterialEditorTexture) -> Self {
        Self {
            image: value.image,
            name: value.name,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorMaterial {
    pub name: Option<String>,
    pub textures: Vec<String>,
    pub shader: Option<String>,
}

impl From<MaterialLayout> for MaterialEditorMaterial {
    fn from(value: MaterialLayout) -> Self {
        Self {
            name: value.name,
            textures: value.textures,
            shader: value.shader,
        }
    }
}

impl From<MaterialEditorMaterial> for MaterialLayout {
    fn from(value: MaterialEditorMaterial) -> Self {
        Self {
            name: value.name,
            textures: value.textures,
            shader: value.shader,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorMesh {
    pub name: Option<String>,
    pub geometry: String,
    pub material: Option<String>,
    pub textures: Vec<String>,
}

impl From<MeshLayout> for MaterialEditorMesh {
    fn from(value: MeshLayout) -> Self {
        Self {
            name: value.name,
            geometry: value.geometry,
            material: value.material,
            textures: value.textures,
        }
    }
}

impl From<MaterialEditorMesh> for MeshLayout {
    fn from(value: MaterialEditorMesh) -> Self {
        Self {
            name: value.name,
            geometry: value.geometry,
            material: value.material,
            textures: value.textures,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorModel {
    pub name: Option<String>,
    pub meshes: Vec<String>,
}

impl From<ModelLayout> for MaterialEditorModel {
    fn from(value: ModelLayout) -> Self {
        Self {
            name: value.name,
            meshes: value.meshes,
        }
    }
}

impl From<MaterialEditorModel> for ModelLayout {
    fn from(value: MaterialEditorModel) -> Self {
        Self {
            name: value.name,
            meshes: value.meshes,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorGraphicsShader {
    pub name: Option<String>,
    pub vertex: Option<String>,
    pub fragment: Option<String>,
    pub geometry: Option<String>,
    #[serde(rename = "tessellation_control")]
    pub tessellation_control: Option<String>,
    #[serde(rename = "tessellation_evaluation")]
    pub tessellation_evaluation: Option<String>,
    pub bind_group_layouts: Vec<Option<cfg::BindGroupLayoutCfg>>,
    pub bind_table_layouts: Vec<Option<cfg::BindTableLayoutCfg>>,
    pub subpass: u8,
    pub render_pass: Option<String>,
}

impl From<GraphicsShaderLayout> for MaterialEditorGraphicsShader {
    fn from(value: GraphicsShaderLayout) -> Self {
        Self {
            name: value.name,
            vertex: value.vertex,
            fragment: value.fragment,
            geometry: value.geometry,
            tessellation_control: value.tessellation_control,
            tessellation_evaluation: value.tessellation_evaluation,
            bind_group_layouts: value.bind_group_layouts,
            bind_table_layouts: value.bind_table_layouts,
            subpass: value.subpass,
            render_pass: value.render_pass,
        }
    }
}

impl From<MaterialEditorGraphicsShader> for GraphicsShaderLayout {
    fn from(value: MaterialEditorGraphicsShader) -> Self {
        Self {
            name: value.name,
            vertex: value.vertex,
            fragment: value.fragment,
            geometry: value.geometry,
            tessellation_control: value.tessellation_control,
            tessellation_evaluation: value.tessellation_evaluation,
            bind_group_layouts: value.bind_group_layouts,
            bind_table_layouts: value.bind_table_layouts,
            subpass: value.subpass,
            render_pass: value.render_pass,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorRenderPass {
    pub debug_name: Option<String>,
    pub viewport: Viewport,
    pub subpasses: Vec<MaterialEditorRenderSubpass>,
}

impl From<RenderPassLayout> for MaterialEditorRenderPass {
    fn from(value: RenderPassLayout) -> Self {
        Self {
            debug_name: value.debug_name,
            viewport: value.viewport,
            subpasses: value.subpasses.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<MaterialEditorRenderPass> for RenderPassLayout {
    fn from(value: MaterialEditorRenderPass) -> Self {
        Self {
            debug_name: value.debug_name,
            viewport: value.viewport,
            subpasses: value.subpasses.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaterialEditorRenderSubpass {
    pub color_attachments: Vec<AttachmentDescription>,
    pub depth_stencil_attachment: Option<AttachmentDescription>,
    pub subpass_dependencies: Vec<SubpassDependency>,
}

impl From<RenderSubpassLayout> for MaterialEditorRenderSubpass {
    fn from(value: RenderSubpassLayout) -> Self {
        Self {
            color_attachments: value.color_attachments,
            depth_stencil_attachment: value.depth_stencil_attachment,
            subpass_dependencies: value.subpass_dependencies,
        }
    }
}

impl From<MaterialEditorRenderSubpass> for RenderSubpassLayout {
    fn from(value: MaterialEditorRenderSubpass) -> Self {
        Self {
            color_attachments: value.color_attachments,
            depth_stencil_attachment: value.depth_stencil_attachment,
            subpass_dependencies: value.subpass_dependencies,
        }
    }
}
