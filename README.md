# eggenvector

eggenvector (like eigenvector haha) is a math library backed by the zig SIMD `@Vector` types, and does not contain any external dependencies. 

The library defaults to [vitellus](https://github.com/eggyengine/vitellus) (and so Slang shader) conventions, as is used for the eggy engine. The convention is set once at compile time, so the same code works for OpenGL, Vulkan, DirectX, Metal and WebGPU (see [conventions](#conventions)).

- Vector2|3|4|any
- Mat|2x2|3x3|4x4|any_rowXany_col (column-major, each column is a SIMD `@Vector`)
- Quaternion, UnitQuaternion
- UnitComplex (2D rotation from an euler angle)
- Unit(T) — algebraic entities with a norm equal to one, e.g. `Unit(Vec3)`
- Isometry2|3 (translation ⨯ rotation)
- Similarity2|3 (translation ⨯ rotation ⨯ uniform scale)
- Affine2|3, Projective2|3, Transform2|3 (homogeneous-matrix transformations)
- Perspective3, Orthographic3 (3D projections for computer graphics)
- Transform
- Angle
- Config (global clip-space convention)

every matrix function lives under `emath.mat`, including the explicit projection variants (`perspectiveRH_ZO`, `frustumLH_NO`, `lookAtRH`, ...), which ignore the configured convention.

## add to project
requires zig `0.16.0` (have not tested for other zig versions, however likely works fine. please open a PR to reduce down the minimum version). 

to use this with the zig build system, import as so:
```bash
zig fetch --save git+https://github.com/eggyengine/eggenvector
```

and then in `build.zig`:
```zig
const emath = b.dependency("eggenvector", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("eggenvector", emath.module("eggenvector"));
```

and lastly in your library/executable:
```zig
const emath = @import("eggenvector");
```

### conventions
`lookAt4x4`, `perspective4x4`, `orthographic4x4` (and `lookAt`, `perspective`, `Perspective3`, `Orthographic3`, `mat.projectPoint`, `mat.unprojectPoint`) follow the global convention:

| convention | handedness | clip-space Y | depth range |
|---|---|---|---|
| `.vitellus` (default) | right | up | [0, 1] |
| `.opengl` | right | up | [-1, 1] |
| `.vulkan` | right | down | [0, 1] |
| `.directx` | left | up | [0, 1] |
| `.metal` | right | up | [0, 1] |
| `.webgpu` | right | up | [0, 1] |

the convention is fixed at compile time. to change it, declare `eggenvector_config` in your root source file (the one containing `main`):
```zig
const emath = @import("eggenvector");

pub const eggenvector_config: emath.Config = .{ .convention = .opengl };
```

it can only be declared there, and assigning to `emath.config` anywhere is a compile error. since the convention is comptime-known, only the chosen branch of each convention switch is compiled in.
