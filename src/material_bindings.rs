use std::collections::BTreeMap;

use dashi::{BindGroupVariable, BindGroupVariableType, cfg};

use noren::parsing::GraphicsShaderLayout;

/// Indicates where a texture binding originates from inside the shader layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TextureBindingKind {
    /// A traditional descriptor set slot.
    BindGroup { group: usize, binding: u32 },
    /// A bindless table slot.
    BindTable { table: usize, binding: u32 },
}

/// Description of a single texture slot that a material must provide.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextureBindingSlot {
    pub kind: TextureBindingKind,
    pub element: u32,
    pub required: bool,
}

/// Enumerates all texture slots required by the provided shader layout.
pub fn texture_binding_slots(layout: &GraphicsShaderLayout) -> Vec<TextureBindingSlot> {
    let mut slots = Vec::new();

    for (group, cfg_opt) in layout.bind_group_layouts.iter().enumerate() {
        if let Some(cfg) = cfg_opt {
            append_slots(
                &mut slots,
                TextureBindingKind::BindGroup { group, binding: 0 },
                &cfg.shaders,
                true,
            );
        }
    }

    for (table, cfg_opt) in layout.bind_table_layouts.iter().enumerate() {
        if let Some(cfg) = cfg_opt {
            append_slots(
                &mut slots,
                TextureBindingKind::BindTable { table, binding: 0 },
                &cfg.shaders,
                false,
            );
        }
    }

    slots
}

fn append_slots(
    slots: &mut Vec<TextureBindingSlot>,
    template: TextureBindingKind,
    shaders: &[cfg::ShaderInfoCfg],
    required: bool,
) {
    for variable in unique_texture_bindings(shaders) {
        for element in 0..variable.count.max(1) {
            let kind = match template {
                TextureBindingKind::BindGroup { group, .. } => TextureBindingKind::BindGroup {
                    group,
                    binding: variable.binding,
                },
                TextureBindingKind::BindTable { table, .. } => TextureBindingKind::BindTable {
                    table,
                    binding: variable.binding,
                },
            };
            slots.push(TextureBindingSlot {
                kind,
                element,
                required,
            });
        }
    }
}

fn unique_texture_bindings(shader_infos: &[cfg::ShaderInfoCfg]) -> Vec<BindGroupVariable> {
    let mut bindings: BTreeMap<u32, BindGroupVariable> = BTreeMap::new();
    for shader in shader_infos {
        for variable in &shader.variables {
            if matches!(
                variable.var_type,
                BindGroupVariableType::SampledImage | BindGroupVariableType::StorageImage
            ) {
                bindings
                    .entry(variable.binding)
                    .or_insert_with(|| variable.clone());
            }
        }
    }
    bindings.into_values().collect()
}
