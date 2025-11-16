# Material Editor

A desktop GUI for authoring Noren material databases. The tool is built on top of `eframe`/`egui` and reuses the runtime JSON schemas defined in `noren` so that authored data can be consumed by renderers without extra conversion steps. The editor reads the existing database manifests, lets you inspect and tweak materials, and can export the changes back to disk.

## Requirements

- Rust 1.84+ (edition 2024) with the `cargo` toolchain.
- A system GL or Vulkan stack that can run either `glow` or `wgpu` backends. The editor selects Glow by default and can opt into WGPU via the `NOREN_EDITOR_RENDERER=wgpu` environment variable.
- Existing database manifests that follow the layout produced by `noren` (`layout.json` plus the `models`, `materials`, and `render_passes` JSON files). See [Project layout](#project-layout) for details.

## Building and running

```bash
# Build and run with the default sample path (./sample/db)
cargo run

# Run the editor against a specific project root
cargo run -- /path/to/material/db

# Switch to the WGPU renderer if you need a modern backend
NOREN_EDITOR_RENDERER=wgpu cargo run -- /path/to/material/db
```

If you omit the project path the binary looks for `sample/db` relative to the current working directory. Passing a path explicitly is recommended for real projects.

## Project layout

The editor expects each project root to contain:

- `layout.json` &mdash; serialized [`MaterialEditorDatabaseLayout`] describing the relative paths of the other manifests (geometry, imagery, models, materials, render passes, and shaders).
- `models`, `materials`, and `render_passes` manifests &mdash; JSON files that mirror the `noren::parsing::ModelLayoutFile` / `RenderPassLayoutFile` structures. The loader reads each file (when present), merges the contents, and converts them into the unified [`MaterialEditorProject`] state used by the GUI.

When you press **Save** the editor writes a fresh `layout.json` plus split `models`, `materials`, and `render_passes` files by calling `write_json_file_blocking`. These files stay compatible with downstream tooling because they are converted back into `ModelLayoutFile` / `RenderPassLayoutFile` before being persisted.

## Editor workflow

The GUI is composed of four primary regions:

1. **Top bar** &mdash; Shows the project path, indicates whether the graph has unsaved edits, and exposes **Save** and **Discard** buttons. Discard reloads the project from disk and re-syncs the preview renderer.
2. **Database browser** &mdash; Provides counts and usage summaries for textures, shaders, and meshes so you can audit dependencies before editing a material.
3. **Material list** &mdash; Lets you create new material IDs, select an existing one, or delete it. Dirty markers highlight modified resources before they are committed.
4. **Inspector + Validation** &mdash; Editing pane where you can rename the material, pick or clear a shader, configure every texture binding slot (respecting the shader layout), review preview meshes, and run validation. Undo/redo buttons are provided via the `EditableResource` history tracker, so experimentation stays non-destructive.

### Asset pickers and binding schema

The inspector derives binding slots directly from the referenced shader layout (`GraphicsShaderLayout`). Whenever you assign a shader, the editor normalizes the `MaterialEditorMaterial` texture array to match the slot count. Texture and shader pickers present the available database entries and support clearing optional bindings, ensuring the material always matches the shader contract.

### Validation messages

The validation panel checks that the selected material references existing shaders and textures and that bindings satisfy uniqueness/required rules. Warnings are surfaced inline so you can fix issues before exporting the database.

### GPU preview panel

A collapsible preview pane renders the selected material on analytic meshes using the lightweight GPU wrapper in `dashi`. You can change the mesh, orbit camera, background color, and toggle wireframe (currently emits a warning while the feature is being implemented). When GPU initialization fails, the renderer captures the error and continues to show explanatory warnings instead of crashing.

## Repository structure

```
src/
├── main.rs                  # Native window bootstrap + renderer selection
├── material_bindings.rs     # Texture binding discovery from shader layouts
├── material_editor/         # High-level editor logic (IO, project graph, preview, UI)
└── material_editor_types/   # GUI-friendly serde types mirrored from noren::parsing
```

- `material_editor::project` owns loading/saving logic, editable graph tracking, and clean/dirty bookkeeping so both the GUI and potential CLI tools can reuse the same persistence layer.
- `material_editor::io` wraps JSON serialization/deserialization with richer errors for GUI presentation.
- `material_bindings` extracts all texture binding slots from `GraphicsShaderLayout`, including bindless tables, allowing the inspector to enforce correctness automatically.
- `material_editor::preview` handles the GPU-backed thumbnail renderer embedded in the inspector.

## Exporting data

Besides saving in-place, the project state can be exported to another directory. `MaterialEditorProjectState::export_to` / `export_to_blocking` rebuild the same manifest structure under a new root, making it easy to branch a database or prepare assets for CI builds. After an export or save, the dirty flag is cleared so the UI reflects the synced state.

## Developing

Run `cargo check` (or `cargo test` once tests are added) to validate changes. The project currently emits a few `dead_code` warnings for helper APIs that are only exercised by the GUI at runtime.
