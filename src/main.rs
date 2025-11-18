mod material_bindings;
mod material_editor;
mod material_editor_types;

use eframe::{Renderer, egui};
use material_editor::{project::MaterialEditorProjectLoader, ui::MaterialEditorApp};
use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let project_root = parse_project_root();
    let state = match MaterialEditorProjectLoader::load_blocking(&project_root) {
        Ok(state) => state,
        Err(err) => {
            eprintln!(
                "Failed to load project at {}: {err}",
                project_root.display()
            );
            std::process::exit(1);
        }
    };

    let options = eframe::NativeOptions {
        renderer: preferred_renderer(),
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(1280.0, 800.0))
            .with_min_inner_size(egui::vec2(960.0, 600.0)),
        ..Default::default()
    };

    let mut initial_state = Some(state);
    eframe::run_native(
        "Noren Material Editor",
        options,
        Box::new(move |_cc| {
            let state = initial_state
                .take()
                .expect("material editor state already taken");
            Box::new(MaterialEditorApp::new(state))
        }),
    )
}

fn parse_project_root() -> PathBuf {
    std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sample/db"))
}

fn preferred_renderer() -> Renderer {
    match std::env::var("NOREN_EDITOR_RENDERER") {
        Ok(value) if value.eq_ignore_ascii_case("wgpu") => Renderer::Wgpu,
        _ => Renderer::Glow,
    }
}
