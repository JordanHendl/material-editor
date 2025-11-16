use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use eframe::egui::{self, Color32, RichText};
use noren::parsing::GraphicsShaderLayout;

use crate::{
    material_bindings::{TextureBindingKind, TextureBindingSlot, texture_binding_slots},
    material_editor::{
        preview::MaterialPreviewPanel,
        project::{
            EditableResource, GraphMaterial, MaterialEditorProjectLoader,
            MaterialEditorProjectState,
        },
    },
    material_editor_types::MaterialEditorMaterial,
};

pub struct MaterialEditorApp {
    state: MaterialEditorProjectState,
    selected_material: Option<String>,
    new_material_id: String,
    status: Option<StatusMessage>,
    picker: Option<PickerDialog>,
    preview: MaterialPreviewPanel,
}

impl MaterialEditorApp {
    pub fn new(state: MaterialEditorProjectState) -> Self {
        let selected_material = Self::initial_selection(&state);
        let preview = MaterialPreviewPanel::new(&state);
        Self {
            state,
            selected_material,
            new_material_id: String::new(),
            status: None,
            picker: None,
            preview,
        }
    }

    fn initial_selection(state: &MaterialEditorProjectState) -> Option<String> {
        let mut keys: Vec<_> = state.graph.materials.keys().cloned().collect();
        keys.sort();
        keys.into_iter().next()
    }

    fn sorted_material_ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.state.graph.materials.keys().cloned().collect();
        ids.sort();
        ids
    }

    fn ensure_valid_selection(&mut self) {
        if let Some(current) = self.selected_material.clone() {
            if !self.state.graph.materials.contains_key(&current) {
                self.selected_material = Self::initial_selection(&self.state);
            }
        } else {
            self.selected_material = Self::initial_selection(&self.state);
        }
    }

    fn prune_status(&mut self) {
        if let Some(status) = &self.status {
            if status.expired() {
                self.status = None;
            }
        }
    }

    fn set_status(&mut self, kind: StatusKind, message: impl Into<String>) {
        self.status = Some(StatusMessage::new(kind, message));
    }

    fn save_project(&mut self) {
        match self.state.save_blocking() {
            Ok(()) => self.set_status(StatusKind::Info, "Project saved"),
            Err(err) => self.set_status(StatusKind::Error, format!("Failed to save: {err}")),
        }
    }

    fn discard_changes(&mut self) {
        let root = self.state.root().to_path_buf();
        match MaterialEditorProjectLoader::load_blocking(root) {
            Ok(new_state) => {
                self.state = new_state;
                self.ensure_valid_selection();
                self.set_status(StatusKind::Info, "Reloaded project");
                self.preview.sync_with_state(&self.state);
            }
            Err(err) => self.set_status(StatusKind::Error, format!("Failed to reload: {err}")),
        }
    }

    fn draw_top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Material Editor");
            ui.separator();
            ui.label(self.state.root().display().to_string());
            ui.separator();
            let dirty = self.state.is_dirty();
            let label = if dirty { "● Dirty" } else { "● Saved" };
            let color = if dirty {
                Color32::from_rgb(235, 111, 111)
            } else {
                Color32::from_rgb(116, 185, 120)
            };
            ui.label(RichText::new(label).color(color));
            if ui
                .add_enabled(dirty, egui::Button::new("Save"))
                .on_hover_text("Write changes to disk")
                .clicked()
            {
                self.save_project();
            }
            if ui
                .add_enabled(dirty, egui::Button::new("Discard"))
                .on_hover_text("Reload the project from disk")
                .clicked()
            {
                self.discard_changes();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(status) = &self.status {
                    let text = RichText::new(&status.text).color(status.color());
                    ui.label(text);
                }
            });
        });
    }

    fn draw_database_browser(&self, ui: &mut egui::Ui) {
        ui.heading("Database Browser");
        ui.label(format!("Textures: {}", self.state.graph.textures.len()));
        ui.label(format!("Shaders: {}", self.state.graph.shaders.len()));
        ui.label(format!("Meshes: {}", self.state.graph.meshes.len()));
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            self.draw_resource_list(
                ui,
                "Textures",
                self.state.graph.textures.keys().cloned().collect(),
                |id| format!("{} • {} material(s)", id, self.texture_usage(&id)),
            );
            self.draw_resource_list(
                ui,
                "Shaders",
                self.state.graph.shaders.keys().cloned().collect(),
                |id| format!("{} • {} material(s)", id, self.shader_usage(&id)),
            );
            self.draw_resource_list(
                ui,
                "Meshes",
                self.state.graph.meshes.keys().cloned().collect(),
                |id| format!("{} • {} preview material(s)", id, self.mesh_usage(&id)),
            );
        });
    }

    fn draw_resource_list(
        &self,
        ui: &mut egui::Ui,
        title: &str,
        mut entries: Vec<String>,
        render: impl Fn(&String) -> String,
    ) {
        entries.sort();
        egui::CollapsingHeader::new(format!("{} ({})", title, entries.len()))
            .default_open(true)
            .show(ui, |ui| {
                for entry in entries {
                    ui.label(RichText::new(render(&entry)).monospace());
                }
            });
    }

    fn draw_material_list(&mut self, ui: &mut egui::Ui) {
        ui.heading("Materials");
        ui.horizontal(|ui| {
            ui.label("ID");
            ui.add(egui::TextEdit::singleline(&mut self.new_material_id).hint_text("material/new"));
            if ui.button("Add").clicked() {
                self.add_material();
            }
        });
        ui.separator();

        let ids = self.sorted_material_ids();
        egui::ScrollArea::vertical().show(ui, |ui| {
            let mut to_remove: Option<String> = None;
            for id in ids {
                let selected = self
                    .selected_material
                    .as_ref()
                    .map(|current| current == &id)
                    .unwrap_or(false);
                let dirty = self
                    .state
                    .graph
                    .materials
                    .get(&id)
                    .map(|mat| mat.resource.dirty)
                    .unwrap_or(false);
                ui.horizontal(|ui| {
                    if ui.selectable_label(selected, &id).clicked() {
                        self.selected_material = Some(id.clone());
                    }
                    if dirty {
                        ui.label(RichText::new("●").color(Color32::from_rgb(235, 168, 75)));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("Delete").clicked() {
                            to_remove = Some(id.clone());
                        }
                    });
                });
            }
            if let Some(id) = to_remove {
                self.delete_material(&id);
            }
        });
    }

    fn add_material(&mut self) {
        let id = self.new_material_id.trim().to_string();
        if id.is_empty() {
            self.set_status(StatusKind::Error, "Material ID is required");
            return;
        }
        if self.state.graph.materials.contains_key(&id) {
            self.set_status(StatusKind::Error, format!("Material '{id}' already exists"));
            return;
        }

        let resource = EditableResource::new(MaterialEditorMaterial::default());
        let graph_material = GraphMaterial {
            referenced_textures: Vec::new(),
            referenced_shader: None,
            preview_meshes: Vec::new(),
            resource,
        };

        self.state
            .graph
            .materials
            .insert(id.clone(), graph_material);
        self.state
            .project
            .materials
            .insert(id.clone(), MaterialEditorMaterial::default());
        self.selected_material = Some(id.clone());
        self.new_material_id.clear();
        self.set_status(StatusKind::Info, format!("Added material '{id}'"));
    }

    fn delete_material(&mut self, id: &str) {
        if self.state.graph.materials.remove(id).is_some() {
            self.state.project.materials.remove(id);
            if self.selected_material.as_deref() == Some(id) {
                self.selected_material = None;
            }
            self.ensure_valid_selection();
            self.set_status(StatusKind::Info, format!("Removed material '{id}'"));
        }
    }

    fn draw_material_inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inspector");
        let Some(material_id) = self.selected_material.clone() else {
            ui.label("Select a material to edit");
            return;
        };
        let (mut working, dirty, can_undo, can_redo, preview_meshes) =
            match self.state.graph.materials.get(&material_id) {
                Some(graph_material) => (
                    graph_material.resource.data.clone(),
                    graph_material.resource.dirty,
                    graph_material.resource.can_undo(),
                    graph_material.resource.can_redo(),
                    graph_material.preview_meshes.clone(),
                ),
                None => {
                    ui.label("Material missing from graph");
                    return;
                }
            };

        ui.horizontal(|ui| {
            ui.heading(&material_id);
            if dirty {
                ui.label(RichText::new("● Dirty").color(Color32::from_rgb(235, 111, 111)));
            }
            if ui
                .add_enabled(can_undo, egui::Button::new("Undo"))
                .on_hover_text("Undo the previous change")
                .clicked()
            {
                self.undo_selected_material();
            }
            if ui
                .add_enabled(can_redo, egui::Button::new("Redo"))
                .on_hover_text("Redo the last undone change")
                .clicked()
            {
                self.redo_selected_material();
            }
        });

        let mut changed = false;
        let (schema, normalized) = self.normalize_material(&mut working);
        if normalized {
            changed = true;
        }
        ui.separator();
        ui.label("Display name");
        let mut display_name = working.name.clone().unwrap_or_default();
        if ui.text_edit_singleline(&mut display_name).changed() {
            let trimmed = display_name.trim().to_string();
            working.name = if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            };
            changed = true;
        }

        ui.separator();
        ui.label("Shader");
        ui.horizontal(|ui| {
            let shader_label = working.shader.as_deref().unwrap_or("None");
            ui.label(RichText::new(shader_label).monospace());
            if ui.button("Pick").clicked() {
                self.open_picker(PickerKind::Shader, working.shader.clone());
            }
            if working.shader.is_some() && ui.button("Clear").clicked() {
                working.shader = None;
                changed = true;
            }
        });

        ui.separator();
        ui.label("Texture bindings");
        if let Some(slots) = &schema {
            if slots.is_empty() {
                ui.label("Selected shader does not expose texture bindings");
            } else {
                for (index, slot) in slots.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(self.binding_slot_label(slot));
                        let texture = working
                            .textures
                            .get_mut(index)
                            .expect("binding slots sync ensures length");
                        let label = if texture.trim().is_empty() {
                            "<default>".to_string()
                        } else {
                            texture.clone()
                        };
                        ui.label(RichText::new(label).monospace());
                        if ui.small_button("Pick").clicked() {
                            let initial = if texture.trim().is_empty() {
                                None
                            } else {
                                Some(texture.clone())
                            };
                            self.open_picker(PickerKind::Texture { slot: index }, initial);
                        }
                        if !matches!(slot.kind, TextureBindingKind::BindGroup { .. })
                            && ui.small_button("Clear").clicked()
                        {
                            texture.clear();
                            changed = true;
                        }
                    });
                }
            }
        } else {
            ui.label("Assign a shader to configure texture bindings");
        }

        if !preview_meshes.is_empty() {
            ui.separator();
            ui.label("Preview meshes");
            for mesh in preview_meshes {
                ui.label(RichText::new(mesh).monospace());
            }
        }

        self.preview.ui(ui, &self.state, &material_id, &working);

        if changed {
            self.persist_material_data(&material_id, working);
        }
    }

    fn draw_validation_panel(&self, ui: &mut egui::Ui) {
        ui.heading("Validation");
        let Some(material_id) = self.selected_material.as_ref() else {
            ui.label("No material selected");
            return;
        };
        let messages = self.collect_validation_messages(material_id);
        if messages.is_empty() {
            ui.label(
                RichText::new("No validation issues detected")
                    .color(Color32::from_rgb(116, 185, 120)),
            );
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            for message in messages {
                ui.label(RichText::new(message).color(Color32::from_rgb(235, 168, 75)));
            }
        });
    }

    fn collect_validation_messages(&self, material_id: &str) -> Vec<String> {
        let Some(graph_material) = self.state.graph.materials.get(material_id) else {
            return vec!["Unknown material".to_string()];
        };
        let data = &graph_material.resource.data;
        let mut messages = Vec::new();

        match data.shader.as_ref() {
            Some(shader) if self.state.graph.shaders.contains_key(shader) => {}
            Some(shader) => messages.push(format!("Shader '{shader}' does not exist")),
            None => messages.push("Material does not reference a shader".to_string()),
        }

        if let Some(slots) = self.binding_schema(data) {
            let mut seen: HashSet<String> = HashSet::new();
            for (index, slot) in slots.iter().enumerate() {
                let value = data
                    .textures
                    .get(index)
                    .map(|entry| entry.trim())
                    .unwrap_or("");
                if value.is_empty() {
                    if matches!(slot.kind, TextureBindingKind::BindGroup { .. }) {
                        messages.push(format!("{} is unassigned", self.binding_slot_label(slot)));
                    }
                    continue;
                }
                if !self.state.graph.textures.contains_key(value) {
                    messages.push(format!("Texture '{}' is missing", value));
                }
                if !seen.insert(value.to_string()) {
                    messages.push(format!("Texture '{}' is bound multiple times", value));
                }
            }
        } else if data.shader.is_some() {
            messages.push("Unable to derive texture bindings for the selected shader".to_string());
        }

        messages
    }

    fn persist_material_data(&mut self, id: &str, mut data: MaterialEditorMaterial) {
        let _ = self.normalize_material(&mut data);
        let snapshot = if let Some(graph_material) = self.state.graph.materials.get_mut(id) {
            graph_material.resource.update(data);
            graph_material.referenced_textures = graph_material.resource.data.textures.clone();
            graph_material.referenced_shader = graph_material.resource.data.shader.clone();
            Some(graph_material.resource.data.clone())
        } else {
            None
        };
        if let Some(updated) = snapshot {
            self.state.project.materials.insert(id.to_string(), updated);
        }
    }

    fn undo_selected_material(&mut self) {
        if let Some(id) = self.selected_material.clone() {
            let mut snapshot = None;
            if let Some(graph_material) = self.state.graph.materials.get_mut(&id) {
                if graph_material.resource.undo() {
                    graph_material.referenced_textures =
                        graph_material.resource.data.textures.clone();
                    graph_material.referenced_shader = graph_material.resource.data.shader.clone();
                    snapshot = Some(graph_material.resource.data.clone());
                }
            }
            if let Some(data) = snapshot {
                self.state.project.materials.insert(id, data);
            }
        }
    }

    fn redo_selected_material(&mut self) {
        if let Some(id) = self.selected_material.clone() {
            let mut snapshot = None;
            if let Some(graph_material) = self.state.graph.materials.get_mut(&id) {
                if graph_material.resource.redo() {
                    graph_material.referenced_textures =
                        graph_material.resource.data.textures.clone();
                    graph_material.referenced_shader = graph_material.resource.data.shader.clone();
                    snapshot = Some(graph_material.resource.data.clone());
                }
            }
            if let Some(data) = snapshot {
                self.state.project.materials.insert(id, data);
            }
        }
    }

    fn open_picker(&mut self, kind: PickerKind, selected: Option<String>) {
        self.picker = Some(PickerDialog::new(kind, selected));
    }

    fn show_picker_dialog(&mut self, ctx: &egui::Context) {
        if let Some(mut dialog) = self.picker.take() {
            let mut apply_choice: Option<Option<String>> = None;
            let entries = self.picker_entries(&dialog.kind);
            egui::Window::new(dialog.title())
                .open(&mut dialog.open)
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.label("Filter");
                    ui.text_edit_singleline(&mut dialog.filter);
                    let mut filtered = entries.clone();
                    filtered.retain(|entry| {
                        dialog.filter.is_empty()
                            || entry
                                .to_ascii_lowercase()
                                .contains(&dialog.filter.to_ascii_lowercase())
                    });
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for entry in filtered {
                            let selected = dialog
                                .selected
                                .as_ref()
                                .map(|current| current == &entry)
                                .unwrap_or(false);
                            let response =
                                ui.selectable_label(selected, RichText::new(&entry).monospace());
                            if response.clicked() {
                                dialog.selected = Some(entry.clone());
                            }
                            if response.double_clicked() {
                                apply_choice = Some(Some(entry));
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Clear binding").clicked() {
                            apply_choice = Some(None);
                        }
                        let select_enabled = dialog.selected.is_some();
                        if ui
                            .add_enabled(select_enabled, egui::Button::new("Select"))
                            .clicked()
                        {
                            apply_choice = Some(dialog.selected.clone());
                        }
                    });
                });
            if let Some(choice) = apply_choice {
                let kind = dialog.kind.clone();
                self.apply_picker_choice(kind, choice);
            } else if dialog.open {
                self.picker = Some(dialog);
            }
        }
    }

    fn picker_entries(&self, kind: &PickerKind) -> Vec<String> {
        let mut entries: Vec<String> = match kind {
            PickerKind::Shader => self.state.graph.shaders.keys().cloned().collect(),
            PickerKind::Texture { .. } => self.state.graph.textures.keys().cloned().collect(),
        };
        entries.sort();
        entries
    }

    fn apply_picker_choice(&mut self, kind: PickerKind, choice: Option<String>) {
        let Some(material_id) = self.selected_material.clone() else {
            return;
        };
        let Some(mut material) = self
            .state
            .graph
            .materials
            .get(&material_id)
            .map(|mat| mat.resource.data.clone())
        else {
            return;
        };
        let mut changed = false;
        match kind {
            PickerKind::Shader => {
                let new_value = choice.filter(|value| !value.is_empty());
                if material.shader != new_value {
                    material.shader = new_value;
                    changed = true;
                }
            }
            PickerKind::Texture { slot } => {
                if slot >= material.textures.len() {
                    return;
                }
                if let Some(value) = choice.filter(|value| !value.is_empty()) {
                    if material.textures[slot] != value {
                        material.textures[slot] = value;
                        changed = true;
                    }
                } else if !material.textures[slot].is_empty() {
                    material.textures[slot].clear();
                    changed = true;
                }
            }
        }
        if changed {
            self.persist_material_data(&material_id, material);
        }
    }

    fn binding_schema(&self, material: &MaterialEditorMaterial) -> Option<Vec<TextureBindingSlot>> {
        let shader_id = material.shader.as_ref()?;
        let graph_shader = self.state.graph.shaders.get(shader_id)?;
        let layout: GraphicsShaderLayout = graph_shader.resource.data.clone().into();
        Some(texture_binding_slots(&layout))
    }

    fn normalize_material(
        &self,
        material: &mut MaterialEditorMaterial,
    ) -> (Option<Vec<TextureBindingSlot>>, bool) {
        let schema = self.binding_schema(material);
        let changed = self.sync_binding_slots(material, schema.as_ref());
        (schema, changed)
    }

    fn sync_binding_slots(
        &self,
        material: &mut MaterialEditorMaterial,
        schema: Option<&Vec<TextureBindingSlot>>,
    ) -> bool {
        let Some(slots) = schema else {
            return false;
        };
        let mut changed = false;
        if material.textures.len() > slots.len() {
            material.textures.truncate(slots.len());
            changed = true;
        }
        while material.textures.len() < slots.len() {
            material.textures.push(String::new());
            changed = true;
        }
        changed
    }

    fn binding_slot_label(&self, slot: &TextureBindingSlot) -> String {
        match slot.kind {
            TextureBindingKind::BindGroup { group, binding } => {
                format!("Set {} binding {} [{}]", group, binding, slot.element)
            }
            TextureBindingKind::BindTable { table, binding } => {
                format!("Table {} binding {} [{}]", table, binding, slot.element)
            }
        }
    }

    fn texture_usage(&self, texture_id: &str) -> usize {
        self.state
            .graph
            .materials
            .values()
            .filter(|mat| mat.referenced_textures.iter().any(|tex| tex == texture_id))
            .count()
    }

    fn shader_usage(&self, shader_id: &str) -> usize {
        self.state
            .graph
            .materials
            .values()
            .filter(|mat| mat.referenced_shader.as_deref() == Some(shader_id))
            .count()
    }

    fn mesh_usage(&self, mesh_id: &str) -> usize {
        self.state
            .graph
            .materials
            .values()
            .filter(|mat| mat.preview_meshes.iter().any(|mesh| mesh == mesh_id))
            .count()
    }
}

impl eframe::App for MaterialEditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.prune_status();

        egui::TopBottomPanel::top("material_editor_top").show(ctx, |ui| self.draw_top_bar(ui));

        egui::SidePanel::left("database_browser")
            .resizable(true)
            .min_width(240.0)
            .show(ctx, |ui| self.draw_database_browser(ui));

        egui::SidePanel::left("material_list")
            .resizable(true)
            .default_width(260.0)
            .show(ctx, |ui| self.draw_material_list(ui));

        egui::CentralPanel::default().show(ctx, |ui| self.draw_material_inspector(ui));

        egui::TopBottomPanel::bottom("validation_panel")
            .resizable(true)
            .default_height(140.0)
            .show(ctx, |ui| self.draw_validation_panel(ui));

        self.show_picker_dialog(ctx);

        ctx.request_repaint_after(Duration::from_millis(200));
    }
}

#[derive(Clone)]
enum PickerKind {
    Shader,
    Texture { slot: usize },
}

struct PickerDialog {
    kind: PickerKind,
    filter: String,
    selected: Option<String>,
    open: bool,
}

impl PickerDialog {
    fn new(kind: PickerKind, selected: Option<String>) -> Self {
        Self {
            kind,
            filter: String::new(),
            selected,
            open: true,
        }
    }

    fn title(&self) -> &'static str {
        match self.kind {
            PickerKind::Shader => "Select Shader",
            PickerKind::Texture { .. } => "Select Texture",
        }
    }
}

struct StatusMessage {
    text: String,
    kind: StatusKind,
    created_at: Instant,
}

impl StatusMessage {
    fn new(kind: StatusKind, text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind,
            created_at: Instant::now(),
        }
    }

    fn expired(&self) -> bool {
        self.created_at.elapsed() > Duration::from_secs(6)
    }

    fn color(&self) -> Color32 {
        match self.kind {
            StatusKind::Info => Color32::from_rgb(116, 185, 120),
            StatusKind::Error => Color32::from_rgb(235, 111, 111),
        }
    }
}

#[derive(Clone, Copy)]
enum StatusKind {
    Info,
    Error,
}
