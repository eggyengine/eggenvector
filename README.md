# eggenvector

eggenvector (like eigenvector haha) is a math library backed by the zig SIMD `@Vector` types, and does not contain any external dependencies. 

The library specifically targets vulkan-based mathematics, however a few tweaks can allow to be fit for anything (speciically with matrices). 

- Vector2|3|4|any
- Mat|2x2|3x3|4x4|any_rowXany_col
- Transform
- Angle

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