This directory is a staging database, and shows how a 'staging db' gets built into a fully fledged database.

To do this, run `dbgen sample_pre/norenbuild.json` and it'll create the output `db` which is usable with samples. Use `dbgen --append sample_pre/norenbuild.json` if you only want to add new entries to existing `.rdb` files without rebuilding them from scratch.

Materials, shaders, and textures now live in `db/materials.json` so the `models.json` file only needs to describe meshes and their composition inside higher level models. Each material declares the textures it binds and the shader it expects to use, which keeps the staging data aligned with how the runtime assembles models.

Render passes are authored separately inside `db/render_passes.json`. Each entry follows the same structure consumed by `dashi::builders::RenderPassBuilder`, so the data can be copied directly from tooling or hand-authored. Every shader in `db/materials.json` must specify the `render_pass` it targets, and that value must match one of the names inside `db/render_passes.json`. The loader will look up the matching pass definition before it finalizes the pipeline, which keeps asset authors in control of how attachments, viewports, and subpasses line up with the shader code.

When you just need to add a single resource, you can skip `norenbuild.json` entirely and write straight into an `.rdb`:

```
dbgen append geometry --rdb db/geometry.rdb --entry geometry/new_quad --gltf sample_pre/gltf/quad.gltf --mesh Quad
dbgen append imagery --rdb db/imagery.rdb --entry imagery/peppers --image sample_pre/imagery/peppers.png --format rgba8
dbgen append shader --rdb db/shaders.rdb --entry shader/quad.frag --stage fragment --shader sample_pre/shaders/quad.frag
```

