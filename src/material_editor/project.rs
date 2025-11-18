use std::{
    collections::HashMap,
    fmt,
    path::{Path, PathBuf},
};

use noren::parsing::{GraphicsShaderLayout, ModelLayoutFile, RenderPassLayoutFile};

use crate::{
    material_bindings::texture_binding_slots,
    material_editor::io::{ProjectIoError, read_json_file_blocking, write_json_file_blocking},
    material_editor_types::{
        MaterialEditorDatabaseLayout, MaterialEditorGraphicsShader, MaterialEditorMaterial,
        MaterialEditorMesh, MaterialEditorProject, MaterialEditorRenderPass, MaterialEditorTexture,
        MaterialTextureBinding,
    },
};

/// Load a material editor project from disk.
pub struct MaterialEditorProjectLoader;

impl MaterialEditorProjectLoader {
    pub async fn load(
        root: impl AsRef<Path>,
    ) -> Result<MaterialEditorProjectState, ProjectLoadError> {
        Self::load_sync(root.as_ref())
    }

    pub fn load_blocking(
        root: impl AsRef<Path>,
    ) -> Result<MaterialEditorProjectState, ProjectLoadError> {
        Self::load_sync(root.as_ref())
    }

    fn load_sync(root: &Path) -> Result<MaterialEditorProjectState, ProjectLoadError> {
        if !root.exists() {
            return Err(ProjectLoadError::MissingRoot(root.to_path_buf()));
        }

        let layout_path = root.join("layout.json");
        let layout: MaterialEditorDatabaseLayout =
            read_json_file_blocking(&layout_path).map_err(ProjectLoadError::Layout)?;

        let paths = ProjectPaths::from_layout(root, &layout);

        let models = load_model_file(&paths.models, AssetKind::Models)?;
        let materials = load_model_file(&paths.materials, AssetKind::Materials)?;
        let render_passes = load_render_pass_file(&paths.render_passes)?;

        let mut combined = models.unwrap_or_default();
        if let Some(materials_file) = materials {
            combined
                .textures
                .extend(materials_file.textures.into_iter());
            combined
                .materials
                .extend(materials_file.materials.into_iter());
            combined.shaders.extend(materials_file.shaders.into_iter());
            combined
                .render_passes
                .extend(materials_file.render_passes.into_iter());
        }

        if let Some(mut pass_file) = render_passes {
            combined
                .render_passes
                .extend(pass_file.render_passes.drain());
        }

        let project: MaterialEditorProject = combined.into();
        let graph = AssetGraph::from_project(&project);

        Ok(MaterialEditorProjectState {
            layout,
            project,
            graph,
            paths,
        })
    }
}

fn load_model_file(
    path: &Path,
    kind: AssetKind,
) -> Result<Option<ModelLayoutFile>, ProjectLoadError> {
    if !path.exists() {
        return Ok(None);
    }

    read_json_file_blocking(path)
        .map(Some)
        .map_err(|error| ProjectLoadError::Asset { kind, error })
}

fn load_render_pass_file(path: &Path) -> Result<Option<RenderPassLayoutFile>, ProjectLoadError> {
    if !path.exists() {
        return Ok(None);
    }

    read_json_file_blocking(path)
        .map(Some)
        .map_err(|error| ProjectLoadError::Asset {
            kind: AssetKind::RenderPasses,
            error,
        })
}

#[derive(Debug)]
pub struct MaterialEditorProjectState {
    pub layout: MaterialEditorDatabaseLayout,
    pub project: MaterialEditorProject,
    pub graph: AssetGraph,
    paths: ProjectPaths,
}

impl MaterialEditorProjectState {
    pub fn root(&self) -> &Path {
        &self.paths.root
    }

    pub fn is_dirty(&self) -> bool {
        self.graph.is_dirty()
    }

    pub fn mark_clean(&mut self) {
        self.graph.mark_clean();
    }

    pub async fn save(&mut self) -> Result<(), ProjectSaveError> {
        self.write_to_paths(&self.paths).await?;
        self.graph.mark_clean();
        Ok(())
    }

    pub fn save_blocking(&mut self) -> Result<(), ProjectSaveError> {
        self.write_to_paths_blocking(&self.paths)?;
        self.graph.mark_clean();
        Ok(())
    }

    pub async fn export_to(&self, root: impl AsRef<Path>) -> Result<(), ProjectSaveError> {
        let export_paths = ProjectPaths::from_layout(root, &self.layout);
        self.write_to_paths(&export_paths).await
    }

    pub fn export_to_blocking(&self, root: impl AsRef<Path>) -> Result<(), ProjectSaveError> {
        let export_paths = ProjectPaths::from_layout(root, &self.layout);
        self.write_to_paths_blocking(&export_paths)
    }

    async fn write_to_paths(&self, paths: &ProjectPaths) -> Result<(), ProjectSaveError> {
        self.write_to_paths_blocking(paths)
    }

    fn write_to_paths_blocking(&self, paths: &ProjectPaths) -> Result<(), ProjectSaveError> {
        write_json_file_blocking(&paths.layout, &self.layout).map_err(ProjectSaveError::Layout)?;

        let layout_file: ModelLayoutFile = self.project.clone().into();
        let mut models_only = ModelLayoutFile::default();
        models_only.meshes = layout_file.meshes.clone();
        models_only.models = layout_file.models.clone();

        write_json_file_blocking(&paths.models, &models_only).map_err(|error| {
            ProjectSaveError::Asset {
                kind: AssetKind::Models,
                error,
            }
        })?;

        let mut materials_only = ModelLayoutFile::default();
        materials_only.textures = layout_file.textures.clone();
        materials_only.materials = layout_file.materials.clone();
        materials_only.shaders = layout_file.shaders.clone();

        write_json_file_blocking(&paths.materials, &materials_only).map_err(|error| {
            ProjectSaveError::Asset {
                kind: AssetKind::Materials,
                error,
            }
        })?;

        let render_pass_file = RenderPassLayoutFile {
            render_passes: layout_file.render_passes,
        };

        write_json_file_blocking(&paths.render_passes, &render_pass_file).map_err(|error| {
            ProjectSaveError::Asset {
                kind: AssetKind::RenderPasses,
                error,
            }
        })
    }
}

#[derive(Debug, Clone)]
struct ProjectPaths {
    root: PathBuf,
    layout: PathBuf,
    models: PathBuf,
    materials: PathBuf,
    render_passes: PathBuf,
}

impl ProjectPaths {
    fn from_layout(root: impl AsRef<Path>, layout: &MaterialEditorDatabaseLayout) -> Self {
        let root_path = root.as_ref().to_path_buf();
        Self {
            layout: root_path.join("layout.json"),
            models: root_path.join(&layout.models),
            materials: root_path.join(&layout.materials),
            render_passes: root_path.join(&layout.render_passes),
            root: root_path,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AssetKind {
    Models,
    Materials,
    RenderPasses,
}

impl fmt::Display for AssetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetKind::Models => write!(f, "models"),
            AssetKind::Materials => write!(f, "materials"),
            AssetKind::RenderPasses => write!(f, "render passes"),
        }
    }
}

#[derive(Debug)]
pub enum ProjectLoadError {
    MissingRoot(PathBuf),
    Layout(ProjectIoError),
    Asset {
        kind: AssetKind,
        error: ProjectIoError,
    },
}

impl fmt::Display for ProjectLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProjectLoadError::MissingRoot(path) => {
                write!(f, "project root does not exist: {}", path.display())
            }
            ProjectLoadError::Layout(err) => {
                write!(f, "failed to read layout.json: {}", err)
            }
            ProjectLoadError::Asset { kind, error } => {
                write!(f, "failed to read {} file: {}", kind, error)
            }
        }
    }
}

impl std::error::Error for ProjectLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProjectLoadError::MissingRoot(_) => None,
            ProjectLoadError::Layout(err) => Some(err),
            ProjectLoadError::Asset { error, .. } => Some(error),
        }
    }
}

#[derive(Debug)]
pub enum ProjectSaveError {
    Layout(ProjectIoError),
    Asset {
        kind: AssetKind,
        error: ProjectIoError,
    },
}

impl fmt::Display for ProjectSaveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProjectSaveError::Layout(err) => {
                write!(f, "failed to save layout.json: {}", err)
            }
            ProjectSaveError::Asset { kind, error } => {
                write!(f, "failed to save {} file: {}", kind, error)
            }
        }
    }
}

impl std::error::Error for ProjectSaveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProjectSaveError::Layout(err) => Some(err),
            ProjectSaveError::Asset { error, .. } => Some(error),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditableResource<T> {
    pub data: T,
    pub dirty: bool,
    history: HistoryTracker<T>,
}

impl<T: Clone> EditableResource<T> {
    pub fn new(data: T) -> Self {
        Self {
            data,
            dirty: false,
            history: HistoryTracker::default(),
        }
    }

    pub fn update(&mut self, data: T) {
        self.history.push_snapshot(self.data.clone());
        self.data = data;
        self.dirty = true;
    }

    pub fn undo(&mut self) -> bool {
        if self.history.undo(&mut self.data) {
            self.dirty = true;
            return true;
        }
        false
    }

    pub fn redo(&mut self) -> bool {
        if self.history.redo(&mut self.data) {
            self.dirty = true;
            return true;
        }
        false
    }

    pub fn mark_clean(&mut self) {
        self.dirty = false;
        self.history.clear();
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }
}

#[derive(Debug, Clone)]
pub struct HistoryTracker<T> {
    past: Vec<T>,
    future: Vec<T>,
}

impl<T: Clone> HistoryTracker<T> {
    fn push_snapshot(&mut self, snapshot: T) {
        self.past.push(snapshot);
        self.future.clear();
    }

    fn undo(&mut self, current: &mut T) -> bool {
        if let Some(previous) = self.past.pop() {
            self.future.push(current.clone());
            *current = previous;
            return true;
        }
        false
    }

    fn redo(&mut self, current: &mut T) -> bool {
        if let Some(next) = self.future.pop() {
            self.past.push(current.clone());
            *current = next;
            return true;
        }
        false
    }

    fn clear(&mut self) {
        self.past.clear();
        self.future.clear();
    }

    fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }
}

impl<T> Default for HistoryTracker<T> {
    fn default() -> Self {
        Self {
            past: Vec::new(),
            future: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphMaterial {
    pub resource: EditableResource<MaterialEditorMaterial>,
    pub referenced_textures: Vec<String>,
    pub referenced_bindless: Vec<String>,
    pub referenced_shader: Option<String>,
    pub preview_meshes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GraphTexture {
    pub resource: EditableResource<MaterialEditorTexture>,
}

#[derive(Debug, Clone)]
pub struct GraphShader {
    pub resource: EditableResource<MaterialEditorGraphicsShader>,
}

#[derive(Debug, Clone)]
pub struct GraphMesh {
    pub resource: EditableResource<MaterialEditorMesh>,
}

#[derive(Debug, Clone)]
pub struct GraphRenderPass {
    pub resource: EditableResource<MaterialEditorRenderPass>,
}

#[derive(Debug, Clone, Default)]
pub struct AssetGraph {
    pub materials: HashMap<String, GraphMaterial>,
    pub textures: HashMap<String, GraphTexture>,
    pub shaders: HashMap<String, GraphShader>,
    pub meshes: HashMap<String, GraphMesh>,
    pub render_passes: HashMap<String, GraphRenderPass>,
}

impl AssetGraph {
    pub fn from_project(project: &MaterialEditorProject) -> Self {
        let textures = project
            .textures
            .iter()
            .map(|(id, tex)| {
                (
                    id.clone(),
                    GraphTexture {
                        resource: EditableResource::new(tex.clone()),
                    },
                )
            })
            .collect();

        let shaders = project
            .shaders
            .iter()
            .map(|(id, shader)| {
                (
                    id.clone(),
                    GraphShader {
                        resource: EditableResource::new(shader.clone()),
                    },
                )
            })
            .collect();

        let meshes = project
            .meshes
            .iter()
            .map(|(id, mesh)| {
                (
                    id.clone(),
                    GraphMesh {
                        resource: EditableResource::new(mesh.clone()),
                    },
                )
            })
            .collect();

        let render_passes = project
            .render_passes
            .iter()
            .map(|(id, pass)| {
                (
                    id.clone(),
                    GraphRenderPass {
                        resource: EditableResource::new(pass.clone()),
                    },
                )
            })
            .collect();

        let mut preview_lookup: HashMap<String, Vec<String>> = HashMap::new();
        for (mesh_id, mesh) in &project.meshes {
            if let Some(material_id) = &mesh.material {
                preview_lookup
                    .entry(material_id.clone())
                    .or_default()
                    .push(mesh_id.clone());
            }
        }

        let materials = project
            .materials
            .iter()
            .map(|(id, material)| {
                let mut material = material.clone();
                if let Some(shader_id) = &material.shader {
                    if let Some(shader) = project.shaders.get(shader_id) {
                        let layout: GraphicsShaderLayout = shader.clone().into();
                        let slots = texture_binding_slots(&layout);
                        for (binding, slot) in material.textures.iter_mut().zip(slots.iter()) {
                            binding.coerce_kind(&slot.kind);
                        }
                    }
                }
                (
                    id.clone(),
                    GraphMaterial {
                        referenced_textures: material
                            .textures
                            .iter()
                            .filter_map(MaterialTextureBinding::as_texture)
                            .map(str::to_string)
                            .collect(),
                        referenced_bindless: material
                            .textures
                            .iter()
                            .filter_map(MaterialTextureBinding::as_bindless)
                            .map(str::to_string)
                            .collect(),
                        referenced_shader: material.shader.clone(),
                        preview_meshes: preview_lookup.remove(id).unwrap_or_else(Vec::new),
                        resource: EditableResource::new(material.clone()),
                    },
                )
            })
            .collect();

        Self {
            materials,
            textures,
            shaders,
            meshes,
            render_passes,
        }
    }

    pub fn is_dirty(&self) -> bool {
        self.materials.values().any(|mat| mat.resource.dirty)
            || self.textures.values().any(|tex| tex.resource.dirty)
            || self.shaders.values().any(|shader| shader.resource.dirty)
            || self.meshes.values().any(|mesh| mesh.resource.dirty)
            || self.render_passes.values().any(|pass| pass.resource.dirty)
    }

    fn mark_clean(&mut self) {
        self.materials
            .values_mut()
            .for_each(|mat| mat.resource.mark_clean());
        self.textures
            .values_mut()
            .for_each(|tex| tex.resource.mark_clean());
        self.shaders
            .values_mut()
            .for_each(|shader| shader.resource.mark_clean());
        self.meshes
            .values_mut()
            .for_each(|mesh| mesh.resource.mark_clean());
        self.render_passes
            .values_mut()
            .for_each(|pass| pass.resource.mark_clean());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_sample_project() {
        let root = PathBuf::from("sample/db");
        let state =
            MaterialEditorProjectLoader::load_blocking(&root).expect("sample project loads");

        assert_eq!(state.project.materials.len(), 3);
        let material = state
            .graph
            .materials
            .get("material/quad")
            .expect("material present");
        assert!(material.referenced_textures.is_empty());
        assert!(material.preview_meshes.is_empty());

        let multi_bind_material = state
            .graph
            .materials
            .get("material/multi_bind")
            .expect("multi bind material present");
        assert_eq!(
            multi_bind_material.referenced_textures,
            vec![
                String::from("texture/quad_diffuse"),
                String::from("texture/quad_overlay"),
            ]
        );

        let bind_table_material = state
            .graph
            .materials
            .get("material/bind_table")
            .expect("bind table material present");
        assert_eq!(
            bind_table_material.referenced_textures,
            Vec::<String>::new()
        );
        assert_eq!(
            bind_table_material.referenced_bindless,
            vec![String::from("texture/quad_overlay")]
        );
    }
}
