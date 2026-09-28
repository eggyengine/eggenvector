const std = @import("std");
const vec = @import("vec.zig");

/// Scalar types a `Matrix` may be built from.
pub const PermittedTypes = enum {
    f16,
    f32,
    f64,
    f128,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
};

/// Whether `T` is a numeric type usable as a matrix scalar.
pub fn isPermittedType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float => true,
        .int => true,
        .comptime_float => true,
        .comptime_int => true,
        else => false,
    };
}

/// Whether `T` is a floating point type.
pub fn isFloatType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float, .comptime_float => true,
        else => false,
    };
}

fn assertPermittedType(comptime T: type) void {
    if (!isPermittedType(T)) {
        @compileError("Matrix type '" ++ @typeName(T) ++ "' is not permitted. Use a numeric type (f16, f32, f64, f128, i8-i64, u8-u64).");
    }
}

/// A generic MxN matrix type, specifically column-major order for compatability with the GPU (specifically vulkan).
/// Each column is a SIMD `@Vector(rows, T)`, so column-wise operations vectorise.
/// T must be a permitted numeric type (validated at comptime).
pub fn Matrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    comptime assertPermittedType(T);

    return struct {
        data: [cols]Column,

        /// A single column, as a SIMD vector.
        pub const Column = @Vector(rows, T);
        /// Number of rows.
        pub const Row = rows;
        /// Number of columns.
        pub const Col = cols;
        /// The scalar type.
        pub const Scalar = T;
    };
}

/// Create a matrix with all elements set to a value.
pub fn splat(comptime T: type, comptime rows: usize, comptime cols: usize, value: T) Matrix(T, rows, cols) {
    return .{ .data = @splat(@splat(value)) };
}

/// Create a zero matrix.
pub fn zero(comptime T: type, comptime rows: usize, comptime cols: usize) Matrix(T, rows, cols) {
    return splat(T, rows, cols, 0);
}

/// Create an identity matrix (requires square matrix).
///
/// The size is provided by `row:column = size:size`
pub fn identity(comptime T: type, comptime size: usize) Matrix(T, size, size) {
    var result = zero(T, size, size);
    inline for (0..size) |i| {
        result.data[i][i] = 1;
    }
    return result;
}

/// Create a matrix from a 2D array in column-major order (each inner array is one column).
pub fn fromArray(comptime T: type, comptime rows: usize, comptime cols: usize, data: [cols][rows]T) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    inline for (0..cols) |c| result.data[c] = data[c];
    return result;
}

/// Get element at row r, column c.
pub fn get(comptime T: type, comptime rows: usize, comptime cols: usize, m: Matrix(T, rows, cols), r: usize, c: usize) T {
    const col: [rows]T = m.data[c];
    return col[r];
}

/// Set element at row r, column c.
pub fn set(comptime T: type, comptime rows: usize, comptime cols: usize, m: *Matrix(T, rows, cols), r: usize, c: usize, value: T) void {
    var col: [rows]T = m.data[c];
    col[r] = value;
    m.data[c] = col;
}

/// Matrix addition (SIMD, per column).
pub fn add(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols)) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    inline for (0..cols) |c| result.data[c] = a.data[c] + b.data[c];
    return result;
}

/// Matrix subtraction (SIMD, per column).
pub fn sub(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols)) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    inline for (0..cols) |c| result.data[c] = a.data[c] - b.data[c];
    return result;
}

/// Scalar multiplication (SIMD, per column).
pub fn scale(comptime T: type, comptime rows: usize, comptime cols: usize, m: Matrix(T, rows, cols), scalar: T) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    inline for (0..cols) |c| result.data[c] = m.data[c] * @as(@Vector(rows, T), @splat(scalar));
    return result;
}

/// Multiply a matrix by a column vector: a linear combination of `m`'s columns (SIMD).
pub fn mulVec(comptime T: type, comptime rows: usize, comptime cols: usize, m: Matrix(T, rows, cols), v: @Vector(cols, T)) @Vector(rows, T) {
    var sum: @Vector(rows, T) = @splat(0);
    inline for (0..cols) |k| sum += m.data[k] * @as(@Vector(rows, T), @splat(v[k]));
    return sum;
}

/// Matrix multiplication (SIMD: each result column is `a` times a column of `b`).
pub fn mul(
    comptime T: type,
    comptime a_rows: usize,
    comptime a_cols: usize,
    comptime b_cols: usize,
    a: Matrix(T, a_rows, a_cols),
    b: Matrix(T, a_cols, b_cols),
) Matrix(T, a_rows, b_cols) {
    var result: Matrix(T, a_rows, b_cols) = undefined;
    inline for (0..b_cols) |c| result.data[c] = mulVec(T, a_rows, a_cols, a, b.data[c]);
    return result;
}

/// Transpose the matrix.
pub fn transpose(comptime T: type, comptime rows: usize, comptime cols: usize, m: Matrix(T, rows, cols)) Matrix(T, cols, rows) {
    var result: Matrix(T, cols, rows) = undefined;
    inline for (0..cols) |c| {
        inline for (0..rows) |r| {
            result.data[r][c] = m.data[c][r];
        }
    }
    return result;
}

/// Compute determinant of a 2x2 matrix.
pub fn determinant2x2(comptime T: type, m: Matrix(T, 2, 2)) T {
    return m.data[0][0] * m.data[1][1] - m.data[1][0] * m.data[0][1];
}

/// Compute inverse of a 2x2 matrix (returns null if singular).
/// Requires floating point type.
pub fn inverse2x2(comptime T: type, m: Matrix(T, 2, 2)) ?Matrix(T, 2, 2) {
    comptime if (!isFloatType(T)) @compileError("inverse2x2 requires a floating point type");
    const det = determinant2x2(T, m);
    if (det == 0) return null;
    const inv_det = 1.0 / det;
    return Matrix(T, 2, 2){
        .data = .{
            .{ m.data[1][1] * inv_det, -m.data[0][1] * inv_det },
            .{ -m.data[1][0] * inv_det, m.data[0][0] * inv_det },
        },
    };
}

/// Create 2x2 rotation matrix.
/// Requires floating point type.
pub fn rotation2x2(comptime T: type, angle: T) Matrix(T, 2, 2) {
    comptime if (!isFloatType(T)) @compileError("rotation2x2 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    return Matrix(T, 2, 2){
        .data = .{
            .{ c, s },
            .{ -s, c },
        },
    };
}

/// Create 2x2 scale matrix
pub fn scaling2x2(comptime T: type, sx: T, sy: T) Matrix(T, 2, 2) {
    return Matrix(T, 2, 2){
        .data = .{
            .{ sx, 0 },
            .{ 0, sy },
        },
    };
}

/// Compute determinant of a 3x3 matrix.
pub fn determinant3x3(comptime T: type, m: Matrix(T, 3, 3)) T {
    return m.data[0][0] * (m.data[1][1] * m.data[2][2] - m.data[2][1] * m.data[1][2]) -
        m.data[1][0] * (m.data[0][1] * m.data[2][2] - m.data[2][1] * m.data[0][2]) +
        m.data[2][0] * (m.data[0][1] * m.data[1][2] - m.data[1][1] * m.data[0][2]);
}

/// Create 3x3 rotation matrix around X axis.
/// Requires floating point type.
pub fn rotationX3x3(comptime T: type, angle: T) Matrix(T, 3, 3) {
    comptime if (!isFloatType(T)) @compileError("rotationX3x3 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    return Matrix(T, 3, 3){
        .data = .{
            .{ 1, 0, 0 },
            .{ 0, c, s },
            .{ 0, -s, c },
        },
    };
}

/// Create 3x3 rotation matrix around Y axis.
/// Requires floating point type.
pub fn rotationY3x3(comptime T: type, angle: T) Matrix(T, 3, 3) {
    comptime if (!isFloatType(T)) @compileError("rotationY3x3 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    return Matrix(T, 3, 3){
        .data = .{
            .{ c, 0, -s },
            .{ 0, 1, 0 },
            .{ s, 0, c },
        },
    };
}

/// Create 3x3 rotation matrix around Z axis.
/// Requires floating point type.
pub fn rotationZ3x3(comptime T: type, angle: T) Matrix(T, 3, 3) {
    comptime if (!isFloatType(T)) @compileError("rotationZ3x3 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    return Matrix(T, 3, 3){
        .data = .{
            .{ c, s, 0 },
            .{ -s, c, 0 },
            .{ 0, 0, 1 },
        },
    };
}

/// Create 3x3 scale matrix.
pub fn scaling3x3(comptime T: type, sx: T, sy: T, sz: T) Matrix(T, 3, 3) {
    return Matrix(T, 3, 3){
        .data = .{
            .{ sx, 0, 0 },
            .{ 0, sy, 0 },
            .{ 0, 0, sz },
        },
    };
}

/// Transform a 3D vector by a 3x3 matrix.
pub fn transformVec3by3x3(comptime T: type, m: Matrix(T, 3, 3), v: vec.Vector3(T)) vec.Vector3(T) {
    const r = mulVec(T, 3, 3, m, .{ v.x, v.y, v.z });
    return .{ .x = r[0], .y = r[1], .z = r[2] };
}

/// Create 4x4 translation matrix.
pub fn translation4x4(comptime T: type, tx: T, ty: T, tz: T) Matrix(T, 4, 4) {
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1, 0, 0, 0 },
            .{ 0, 1, 0, 0 },
            .{ 0, 0, 1, 0 },
            .{ tx, ty, tz, 1 },
        },
    };
}

/// Create 4x4 translation matrix from vector.
pub fn translationVec4x4(comptime T: type, v: vec.Vector3(T)) Matrix(T, 4, 4) {
    return translation4x4(T, v.x, v.y, v.z);
}

/// Create 4x4 scale matrix.
pub fn scaling4x4(comptime T: type, sx: T, sy: T, sz: T) Matrix(T, 4, 4) {
    return Matrix(T, 4, 4){
        .data = .{
            .{ sx, 0, 0, 0 },
            .{ 0, sy, 0, 0 },
            .{ 0, 0, sz, 0 },
            .{ 0, 0, 0, 1 },
        },
    };
}

/// Create 4x4 uniform scale matrix.
pub fn uniformScaling4x4(comptime T: type, s: T) Matrix(T, 4, 4) {
    return scaling4x4(T, s, s, s);
}

/// Create 4x4 rotation matrix around X axis.
/// Requires floating point type.
pub fn rotationX4x4(comptime T: type, angle: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("rotationX4x4 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1, 0, 0, 0 },
            .{ 0, c, s, 0 },
            .{ 0, -s, c, 0 },
            .{ 0, 0, 0, 1 },
        },
    };
}

/// Create 4x4 rotation matrix around Y axis.
/// Requires floating point type.
pub fn rotationY4x4(comptime T: type, angle: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("rotationY4x4 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    return Matrix(T, 4, 4){
        .data = .{
            .{ c, 0, -s, 0 },
            .{ 0, 1, 0, 0 },
            .{ s, 0, c, 0 },
            .{ 0, 0, 0, 1 },
        },
    };
}

/// Create 4x4 rotation matrix around Z axis.
/// Requires floating point type.
pub fn rotationZ4x4(comptime T: type, angle: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("rotationZ4x4 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    return Matrix(T, 4, 4){
        .data = .{
            .{ c, s, 0, 0 },
            .{ -s, c, 0, 0 },
            .{ 0, 0, 1, 0 },
            .{ 0, 0, 0, 1 },
        },
    };
}

/// Create 4x4 rotation matrix around arbitrary axis.
/// Requires floating point type.
pub fn rotationAxis4x4(comptime T: type, axis: vec.Vector3(T), angle: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("rotationAxis4x4 requires a floating point type");
    const c = @cos(angle);
    const s = @sin(angle);
    const t = 1 - c;
    const n = axis.normalize();
    const x = n.x;
    const y = n.y;
    const z = n.z;

    return Matrix(T, 4, 4){
        .data = .{
            .{ t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0 },
            .{ t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0 },
            .{ t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0 },
            .{ 0, 0, 0, 1 },
        },
    };
}

/// Global math settings, fixed at compile time. Declare `pub const eggenvector_config: Config`
/// in your root source file (the one with `main`) to override the defaults.
pub const Config = struct {
    convention: Convention = .vitellus,

    /// Coordinate/clip-space convention used by `lookAt4x4`, `perspective4x4`, `orthographic4x4`,
    /// `projectPoint` and `unprojectPoint`.
    pub const Convention = enum {
        /// Right-handed, Y up, depth [0, 1]. Vitellus/Slang (D3D/Metal clip space on every backend,
        /// the Vulkan backend flips the viewport for you).
        vitellus,
        /// Right-handed, Y up, depth [-1, 1].
        opengl,
        /// Right-handed, Y down, depth [0, 1]. Raw Vulkan without a negative-height viewport.
        vulkan,
        /// Left-handed, Y up, depth [0, 1].
        directx,
        /// Right-handed, Y up, depth [0, 1].
        metal,
        /// Right-handed, Y up, depth [0, 1].
        webgpu,
    };
};

const root = @import("root");

/// The global config: the root file's `eggenvector_config` if it declares one, otherwise the defaults.
/// Comptime-known, so every convention `switch` below compiles down to a single branch.
pub const config: Config = if (@hasDecl(root, "eggenvector_config")) root.eggenvector_config else .{};

/// Create a look-at view matrix for `config.convention` (left-handed for `.directx`, right-handed otherwise).
/// Requires floating point type.
pub fn lookAt4x4(comptime T: type, eye: vec.Vector3(T), target: vec.Vector3(T), up: vec.Vector3(T)) Matrix(T, 4, 4) {
    return lookAtFor(config.convention, T, eye, target, up);
}

fn lookAtFor(comptime c: Config.Convention, comptime T: type, eye: vec.Vector3(T), target: vec.Vector3(T), up: vec.Vector3(T)) Matrix(T, 4, 4) {
    return switch (c) {
        .directx => lookAtLH(T, eye, target, up),
        else => lookAtRH(T, eye, target, up),
    };
}

/// Create a perspective projection matrix for `config.convention`.
///
/// fov_y: vertical field of view in radians
/// aspect: width / height
/// near, far: near and far clipping planes
/// Requires floating point type.
pub fn perspective4x4(comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    return perspectiveFor(config.convention, T, fov_y, aspect, near, far);
}

fn perspectiveFor(comptime c: Config.Convention, comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    return switch (c) {
        .vitellus, .metal, .webgpu => perspectiveRH_ZO(T, fov_y, aspect, near, far),
        .opengl => perspectiveRH_NO(T, fov_y, aspect, near, far),
        .vulkan => perspectiveVulkan(T, fov_y, aspect, near, far),
        .directx => perspectiveLH_ZO(T, fov_y, aspect, near, far),
    };
}

/// Create an orthographic projection matrix for `config.convention`.
/// Requires floating point type.
pub fn orthographic4x4(comptime T: type, left: T, right: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    return orthographicFor(config.convention, T, left, right, bottom, top, near, far);
}

fn orthographicFor(comptime c: Config.Convention, comptime T: type, left: T, right: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    return switch (c) {
        .vitellus, .metal, .webgpu => orthographicRH_ZO(T, left, right, bottom, top, near, far),
        .opengl => orthographicRH_NO(T, left, right, bottom, top, near, far),
        .vulkan => orthographicVulkan(T, left, right, bottom, top, near, far),
        .directx => orthographicLH_ZO(T, left, right, bottom, top, near, far),
    };
}

/// Transform a 4D vector by a 4x4 matrix.
pub fn transformVec4by4x4(comptime T: type, m: Matrix(T, 4, 4), v: vec.Vector4(T)) vec.Vector4(T) {
    const r = mulVec(T, 4, 4, m, .{ v.x, v.y, v.z, v.w });
    return .{ .x = r[0], .y = r[1], .z = r[2], .w = r[3] };
}

/// Transform a 3D point (w=1) by a 4x4 matrix.
/// Requires floating point type.
pub fn transformPoint4x4(comptime T: type, m: Matrix(T, 4, 4), v: vec.Vector3(T)) vec.Vector3(T) {
    comptime if (!isFloatType(T)) @compileError("transformPoint4x4 requires a floating point type");
    const r = mulVec(T, 4, 4, m, .{ v.x, v.y, v.z, 1 });
    const p = r / @as(@Vector(4, T), @splat(r[3]));
    return .{ .x = p[0], .y = p[1], .z = p[2] };
}

/// Transform a 3D direction (w=0, no translation) by a 4x4 matrix.
pub fn transformDirection4x4(comptime T: type, m: Matrix(T, 4, 4), v: vec.Vector3(T)) vec.Vector3(T) {
    const r = mulVec(T, 4, 4, m, .{ v.x, v.y, v.z, 0 });
    return .{ .x = r[0], .y = r[1], .z = r[2] };
}

/// Multiply two 4x4 matrices.
pub fn multiply4x4(comptime T: type, a: Matrix(T, 4, 4), b: Matrix(T, 4, 4)) Matrix(T, 4, 4) {
    return mul(T, 4, 4, 4, a, b);
}

/// Get the inverse of a 4x4 matrix (returns null if singular).
/// Requires floating point type.
pub fn inverse4x4(comptime T: type, m: Matrix(T, 4, 4)) ?Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("inverse4x4 requires a floating point type");

    const Vec4 = vec.Vector4(T);

    // Compute cofactors
    const c00 = m.data[2][2] * m.data[3][3] - m.data[3][2] * m.data[2][3];
    const c02 = m.data[1][2] * m.data[3][3] - m.data[3][2] * m.data[1][3];
    const c03 = m.data[1][2] * m.data[2][3] - m.data[2][2] * m.data[1][3];

    const c04 = m.data[2][1] * m.data[3][3] - m.data[3][1] * m.data[2][3];
    const c06 = m.data[1][1] * m.data[3][3] - m.data[3][1] * m.data[1][3];
    const c07 = m.data[1][1] * m.data[2][3] - m.data[2][1] * m.data[1][3];

    const c08 = m.data[2][1] * m.data[3][2] - m.data[3][1] * m.data[2][2];
    const c10 = m.data[1][1] * m.data[3][2] - m.data[3][1] * m.data[1][2];
    const c11 = m.data[1][1] * m.data[2][2] - m.data[2][1] * m.data[1][2];

    const c12 = m.data[2][0] * m.data[3][3] - m.data[3][0] * m.data[2][3];
    const c14 = m.data[1][0] * m.data[3][3] - m.data[3][0] * m.data[1][3];
    const c15 = m.data[1][0] * m.data[2][3] - m.data[2][0] * m.data[1][3];

    const c16 = m.data[2][0] * m.data[3][2] - m.data[3][0] * m.data[2][2];
    const c18 = m.data[1][0] * m.data[3][2] - m.data[3][0] * m.data[1][2];
    const c19 = m.data[1][0] * m.data[2][2] - m.data[2][0] * m.data[1][2];

    const c20 = m.data[2][0] * m.data[3][1] - m.data[3][0] * m.data[2][1];
    const c22 = m.data[1][0] * m.data[3][1] - m.data[3][0] * m.data[1][1];
    const c23 = m.data[1][0] * m.data[2][1] - m.data[2][0] * m.data[1][1];

    const f0 = Vec4{ .x = c00, .y = c00, .z = c02, .w = c03 };
    const f1 = Vec4{ .x = c04, .y = c04, .z = c06, .w = c07 };
    const f2 = Vec4{ .x = c08, .y = c08, .z = c10, .w = c11 };
    const f3 = Vec4{ .x = c12, .y = c12, .z = c14, .w = c15 };
    const f4 = Vec4{ .x = c16, .y = c16, .z = c18, .w = c19 };
    const f5 = Vec4{ .x = c20, .y = c20, .z = c22, .w = c23 };

    const v0 = Vec4{ .x = m.data[1][0], .y = m.data[0][0], .z = m.data[0][0], .w = m.data[0][0] };
    const v1 = Vec4{ .x = m.data[1][1], .y = m.data[0][1], .z = m.data[0][1], .w = m.data[0][1] };
    const v2 = Vec4{ .x = m.data[1][2], .y = m.data[0][2], .z = m.data[0][2], .w = m.data[0][2] };
    const v3 = Vec4{ .x = m.data[1][3], .y = m.data[0][3], .z = m.data[0][3], .w = m.data[0][3] };

    const sign_a = Vec4{ .x = 1, .y = -1, .z = 1, .w = -1 };
    const sign_b = Vec4{ .x = -1, .y = 1, .z = -1, .w = 1 };

    const adj0 = v1.mul(f0).sub(v2.mul(f1)).add(v3.mul(f2));
    const adj1 = v0.mul(f0).sub(v2.mul(f3)).add(v3.mul(f4));
    const adj2 = v0.mul(f1).sub(v1.mul(f3)).add(v3.mul(f5));
    const adj3 = v0.mul(f2).sub(v1.mul(f4)).add(v2.mul(f5));

    const inv0 = adj0.mul(sign_a);
    const inv1 = adj1.mul(sign_b);
    const inv2 = adj2.mul(sign_a);
    const inv3 = adj3.mul(sign_b);

    // det = first column of m · first row of the adjugate
    const adj_row0 = Vec4{ .x = inv0.x, .y = inv1.x, .z = inv2.x, .w = inv3.x };
    const col0 = Vec4{ .x = m.data[0][0], .y = m.data[0][1], .z = m.data[0][2], .w = m.data[0][3] };
    const det = col0.dot(adj_row0);

    if (@abs(det) < 1e-10) return null;

    const inv_det = 1 / det;

    return Matrix(T, 4, 4){
        .data = .{
            .{ inv0.x * inv_det, inv0.y * inv_det, inv0.z * inv_det, inv0.w * inv_det },
            .{ inv1.x * inv_det, inv1.y * inv_det, inv1.z * inv_det, inv1.w * inv_det },
            .{ inv2.x * inv_det, inv2.y * inv_det, inv2.z * inv_det, inv2.w * inv_det },
            .{ inv3.x * inv_det, inv3.y * inv_det, inv3.z * inv_det, inv3.w * inv_det },
        },
    };
}

/// Frustum projection: right-handed, depth [0,1] (Vulkan, Metal).
pub fn frustumRH_ZO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("frustumRH_ZO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ (2 * near) / width, 0, 0, 0 },
            .{ 0, (2 * near) / height, 0, 0 },
            .{ (right_ + left) / width, (top + bottom) / height, -far / depth, -1 },
            .{ 0, 0, -(far * near) / depth, 0 },
        },
    };
}

/// Frustum projection: right-handed, depth [-1,1] (OpenGL default).
pub fn frustumRH_NO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("frustumRH_NO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ (2 * near) / width, 0, 0, 0 },
            .{ 0, (2 * near) / height, 0, 0 },
            .{ (right_ + left) / width, (top + bottom) / height, -(far + near) / depth, -1 },
            .{ 0, 0, -(2 * far * near) / depth, 0 },
        },
    };
}

/// Frustum projection: left-handed, depth [0,1] (Direct3D).
pub fn frustumLH_ZO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("frustumLH_ZO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ (2 * near) / width, 0, 0, 0 },
            .{ 0, (2 * near) / height, 0, 0 },
            .{ -(right_ + left) / width, -(top + bottom) / height, far / depth, 1 },
            .{ 0, 0, -(far * near) / depth, 0 },
        },
    };
}

/// Frustum projection: left-handed, depth [-1,1].
pub fn frustumLH_NO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("frustumLH_NO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ (2 * near) / width, 0, 0, 0 },
            .{ 0, (2 * near) / height, 0, 0 },
            .{ -(right_ + left) / width, -(top + bottom) / height, (far + near) / depth, 1 },
            .{ 0, 0, -(2 * far * near) / depth, 0 },
        },
    };
}

/// Perspective projection: right-handed, depth [0,1] (Vulkan, Metal).
pub fn perspectiveRH_ZO(comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("perspectiveRH_ZO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, far / (near - far), -1 },
            .{ 0, 0, -(far * near) / (far - near), 0 },
        },
    };
}

/// Perspective projection: right-handed, depth [-1,1] (OpenGL default).
pub fn perspectiveRH_NO(comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("perspectiveRH_NO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, -(far + near) / (far - near), -1 },
            .{ 0, 0, -(2 * far * near) / (far - near), 0 },
        },
    };
}

/// Perspective projection: left-handed, depth [0,1] (Direct3D).
pub fn perspectiveLH_ZO(comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("perspectiveLH_ZO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, far / (far - near), 1 },
            .{ 0, 0, -(far * near) / (far - near), 0 },
        },
    };
}

/// Perspective projection: left-handed, depth [-1,1].
pub fn perspectiveLH_NO(comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("perspectiveLH_NO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, (far + near) / (far - near), 1 },
            .{ 0, 0, -(2 * far * near) / (far - near), 0 },
        },
    };
}

/// Vulkan-specific perspective (RH, ZO, Y-flipped).
/// Used by `perspective4x4` under the `.vulkan` convention.
pub fn perspectiveVulkan(comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("perspectiveVulkan requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, -1 / tan_half_fov, 0, 0 },
            .{ 0, 0, far / (near - far), -1 },
            .{ 0, 0, -(far * near) / (far - near), 0 },
        },
    };
}

/// Infinite perspective: right-handed, depth [0,1].
pub fn infinitePerspectiveRH_ZO(comptime T: type, fov_y: T, aspect: T, near: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("infinitePerspectiveRH_ZO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, -1, -1 },
            .{ 0, 0, -near, 0 },
        },
    };
}

/// Infinite perspective: right-handed, depth [-1,1].
pub fn infinitePerspectiveRH_NO(comptime T: type, fov_y: T, aspect: T, near: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("infinitePerspectiveRH_NO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, -1, -1 },
            .{ 0, 0, -2 * near, 0 },
        },
    };
}

/// Infinite perspective: left-handed, depth [0,1].
pub fn infinitePerspectiveLH_ZO(comptime T: type, fov_y: T, aspect: T, near: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("infinitePerspectiveLH_ZO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, 1, 1 },
            .{ 0, 0, -near, 0 },
        },
    };
}

/// Infinite perspective: left-handed, depth [-1,1].
pub fn infinitePerspectiveLH_NO(comptime T: type, fov_y: T, aspect: T, near: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("infinitePerspectiveLH_NO requires a floating point type");
    const tan_half_fov = @tan(fov_y / 2);
    return Matrix(T, 4, 4){
        .data = .{
            .{ 1 / (aspect * tan_half_fov), 0, 0, 0 },
            .{ 0, 1 / tan_half_fov, 0, 0 },
            .{ 0, 0, 1, 1 },
            .{ 0, 0, -2 * near, 0 },
        },
    };
}

/// Orthographic projection: right-handed, depth [0,1] (Vulkan, Metal).
pub fn orthographicRH_ZO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("orthographicRH_ZO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ 2 / width, 0, 0, 0 },
            .{ 0, 2 / height, 0, 0 },
            .{ 0, 0, -1 / depth, 0 },
            .{ -(right_ + left) / width, -(top + bottom) / height, -near / depth, 1 },
        },
    };
}

/// Orthographic projection: right-handed, depth [-1,1] (OpenGL default).
pub fn orthographicRH_NO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("orthographicRH_NO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ 2 / width, 0, 0, 0 },
            .{ 0, 2 / height, 0, 0 },
            .{ 0, 0, -2 / depth, 0 },
            .{ -(right_ + left) / width, -(top + bottom) / height, -(far + near) / depth, 1 },
        },
    };
}

/// Orthographic projection: left-handed, depth [0,1] (Direct3D).
pub fn orthographicLH_ZO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("orthographicLH_ZO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ 2 / width, 0, 0, 0 },
            .{ 0, 2 / height, 0, 0 },
            .{ 0, 0, 1 / depth, 0 },
            .{ -(right_ + left) / width, -(top + bottom) / height, -near / depth, 1 },
        },
    };
}

/// Orthographic projection: left-handed, depth [-1,1].
pub fn orthographicLH_NO(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("orthographicLH_NO requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ 2 / width, 0, 0, 0 },
            .{ 0, 2 / height, 0, 0 },
            .{ 0, 0, 2 / depth, 0 },
            .{ -(right_ + left) / width, -(top + bottom) / height, -(far + near) / depth, 1 },
        },
    };
}

/// Vulkan-specific orthographic (RH, ZO, Y-flipped).
/// Used by `orthographic4x4` under the `.vulkan` convention.
pub fn orthographicVulkan(comptime T: type, left: T, right_: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("orthographicVulkan requires a floating point type");
    const width = right_ - left;
    const height = top - bottom;
    const depth = far - near;
    return Matrix(T, 4, 4){
        .data = .{
            .{ 2 / width, 0, 0, 0 },
            .{ 0, -2 / height, 0, 0 },
            .{ 0, 0, -1 / depth, 0 },
            .{ -(right_ + left) / width, (top + bottom) / height, -near / depth, 1 },
        },
    };
}

/// Right-handed look-at view matrix.
pub fn lookAtRH(comptime T: type, eye: vec.Vector3(T), target: vec.Vector3(T), up: vec.Vector3(T)) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("lookAtRH requires a floating point type");
    const f = target.sub(eye).normalize();
    const s = f.cross(up).normalize();
    const u = s.cross(f);
    return Matrix(T, 4, 4){
        .data = .{
            .{ s.x, u.x, -f.x, 0 },
            .{ s.y, u.y, -f.y, 0 },
            .{ s.z, u.z, -f.z, 0 },
            .{ -s.dot(eye), -u.dot(eye), f.dot(eye), 1 },
        },
    };
}

/// Left-handed look-at view matrix.
pub fn lookAtLH(comptime T: type, eye: vec.Vector3(T), target: vec.Vector3(T), up: vec.Vector3(T)) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("lookAtLH requires a floating point type");
    const f = target.sub(eye).normalize();
    const s = up.cross(f).normalize();
    const u = f.cross(s);
    return Matrix(T, 4, 4){
        .data = .{
            .{ s.x, u.x, f.x, 0 },
            .{ s.y, u.y, f.y, 0 },
            .{ s.z, u.z, f.z, 0 },
            .{ -s.dot(eye), -u.dot(eye), -f.dot(eye), 1 },
        },
    };
}

/// Compute determinant of a 4x4 matrix.
pub fn determinant4x4(comptime T: type, m: Matrix(T, 4, 4)) T {
    const s0 = m.data[0][0] * m.data[1][1] - m.data[1][0] * m.data[0][1];
    const s1 = m.data[0][0] * m.data[2][1] - m.data[2][0] * m.data[0][1];
    const s2 = m.data[0][0] * m.data[3][1] - m.data[3][0] * m.data[0][1];
    const s3 = m.data[1][0] * m.data[2][1] - m.data[2][0] * m.data[1][1];
    const s4 = m.data[1][0] * m.data[3][1] - m.data[3][0] * m.data[1][1];
    const s5 = m.data[2][0] * m.data[3][1] - m.data[3][0] * m.data[2][1];

    const c5 = m.data[2][2] * m.data[3][3] - m.data[3][2] * m.data[2][3];
    const c4 = m.data[1][2] * m.data[3][3] - m.data[3][2] * m.data[1][3];
    const c3 = m.data[1][2] * m.data[2][3] - m.data[2][2] * m.data[1][3];
    const c2 = m.data[0][2] * m.data[3][3] - m.data[3][2] * m.data[0][3];
    const c1 = m.data[0][2] * m.data[2][3] - m.data[2][2] * m.data[0][3];
    const c0 = m.data[0][2] * m.data[1][3] - m.data[1][2] * m.data[0][3];

    return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
}

/// Compute inverse of a 3x3 matrix (returns null if singular).
/// Requires floating point type.
pub fn inverse3x3(comptime T: type, m: Matrix(T, 3, 3)) ?Matrix(T, 3, 3) {
    comptime if (!isFloatType(T)) @compileError("inverse3x3 requires a floating point type");
    const det = determinant3x3(T, m);
    if (@abs(det) < 1e-10) return null;
    const inv_det = 1 / det;

    return Matrix(T, 3, 3){
        .data = .{
            .{
                (m.data[1][1] * m.data[2][2] - m.data[2][1] * m.data[1][2]) * inv_det,
                (m.data[2][1] * m.data[0][2] - m.data[0][1] * m.data[2][2]) * inv_det,
                (m.data[0][1] * m.data[1][2] - m.data[1][1] * m.data[0][2]) * inv_det,
            },
            .{
                (m.data[2][0] * m.data[1][2] - m.data[1][0] * m.data[2][2]) * inv_det,
                (m.data[0][0] * m.data[2][2] - m.data[2][0] * m.data[0][2]) * inv_det,
                (m.data[1][0] * m.data[0][2] - m.data[0][0] * m.data[1][2]) * inv_det,
            },
            .{
                (m.data[1][0] * m.data[2][1] - m.data[2][0] * m.data[1][1]) * inv_det,
                (m.data[2][0] * m.data[0][1] - m.data[0][0] * m.data[2][1]) * inv_det,
                (m.data[0][0] * m.data[1][1] - m.data[1][0] * m.data[0][1]) * inv_det,
            },
        },
    };
}

/// Extract the upper-left 3x3 portion of a 4x4 matrix.
pub fn extractMat3(comptime T: type, m: Matrix(T, 4, 4)) Matrix(T, 3, 3) {
    return Matrix(T, 3, 3){
        .data = .{
            .{ m.data[0][0], m.data[0][1], m.data[0][2] },
            .{ m.data[1][0], m.data[1][1], m.data[1][2] },
            .{ m.data[2][0], m.data[2][1], m.data[2][2] },
        },
    };
}

/// Compute the normal matrix (inverse transpose of upper-left 3x3).
/// Returns null if the 3x3 portion is singular.
/// Requires floating point type.
pub fn normalMatrix(comptime T: type, m: Matrix(T, 4, 4)) ?Matrix(T, 3, 3) {
    comptime if (!isFloatType(T)) @compileError("normalMatrix requires a floating point type");
    const m3 = extractMat3(T, m);
    const inv = inverse3x3(T, m3) orelse return null;
    return transpose(T, 3, 3, inv);
}

/// Project a 3D object-space point to window coordinates.
/// model: model-view matrix, proj: projection matrix.
/// viewport: vec4(x, y, width, height).
pub fn projectPoint(comptime T: type, obj: vec.Vector3(T), model: Matrix(T, 4, 4), proj: Matrix(T, 4, 4), viewport: vec.Vector4(T)) vec.Vector3(T) {
    comptime if (!isFloatType(T)) @compileError("projectPoint requires a floating point type");
    const tmp = vec.Vector4(T){ .x = obj.x, .y = obj.y, .z = obj.z, .w = 1 };
    var result = transformVec4by4x4(T, model, tmp);
    result = transformVec4by4x4(T, proj, result);

    // Perspective divide
    result.x /= result.w;
    result.y /= result.w;
    result.z /= result.w;

    // Map to [0, 1]
    result.x = result.x * 0.5 + 0.5;
    result.y = result.y * 0.5 + 0.5;
    if (comptime config.convention == .opengl) result.z = result.z * 0.5 + 0.5;

    // Map to viewport
    return vec.Vector3(T){
        .x = result.x * viewport.z + viewport.x,
        .y = result.y * viewport.w + viewport.y,
        .z = result.z,
    };
}

/// Unproject window coordinates back to object space.
/// win: vec3(winX, winY, depth).
/// model: model-view matrix, proj: projection matrix.
/// viewport: vec4(x, y, width, height).
/// Returns null if the combined matrix is singular.
pub fn unprojectPoint(comptime T: type, win: vec.Vector3(T), model: Matrix(T, 4, 4), proj: Matrix(T, 4, 4), viewport: vec.Vector4(T)) ?vec.Vector3(T) {
    comptime if (!isFloatType(T)) @compileError("unprojectPoint requires a floating point type");
    const combined = multiply4x4(T, proj, model);
    const inv = inverse4x4(T, combined) orelse return null;

    // Map from viewport to [-1, 1]
    var tmp = vec.Vector4(T){
        .x = (win.x - viewport.x) / viewport.z * 2 - 1,
        .y = (win.y - viewport.y) / viewport.w * 2 - 1,
        .z = if (comptime config.convention == .opengl) win.z * 2 - 1 else win.z,
        .w = 1,
    };

    tmp = transformVec4by4x4(T, inv, tmp);
    if (tmp.w == 0) return null;

    return vec.Vector3(T){
        .x = tmp.x / tmp.w,
        .y = tmp.y / tmp.w,
        .z = tmp.z / tmp.w,
    };
}

/// Check if two matrices are approximately equal (element-wise, SIMD).
pub fn approxEql(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols), epsilon: T) bool {
    inline for (0..cols) |c| {
        if (@reduce(.Or, @abs(a.data[c] - b.data[c]) > @as(@Vector(rows, T), @splat(epsilon)))) return false;
    }
    return true;
}

/// Check if two matrices are exactly equal (SIMD).
pub fn eql(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols)) bool {
    inline for (0..cols) |c| {
        if (@reduce(.Or, a.data[c] != b.data[c])) return false;
    }
    return true;
}

/// Compute the trace of a square matrix (sum of diagonal elements).
pub fn trace(comptime T: type, comptime size: usize, m: Matrix(T, size, size)) T {
    var result: T = 0;
    inline for (0..size) |i| {
        result += m.data[i][i];
    }
    return result;
}

/// A 2x2 `f32` matrix (column-major).
pub const Mat2 = Matrix(f32, 2, 2);
/// A 3x3 `f32` matrix (column-major).
pub const Mat3 = Matrix(f32, 3, 3);
/// A 4x4 `f32` matrix (column-major).
pub const Mat4 = Matrix(f32, 4, 4);

/// A 2x2 `f64` matrix (column-major).
pub const Mat2d = Matrix(f64, 2, 2);
/// A 3x3 `f64` matrix (column-major).
pub const Mat3d = Matrix(f64, 3, 3);
/// A 4x4 `f64` matrix (column-major).
pub const Mat4d = Matrix(f64, 4, 4);

/// A 2x2 `i32` matrix (column-major).
pub const Mat2i = Matrix(i32, 2, 2);
/// A 3x3 `i32` matrix (column-major).
pub const Mat3i = Matrix(i32, 3, 3);
/// A 4x4 `i32` matrix (column-major).
pub const Mat4i = Matrix(i32, 4, 4);

test "each convention's projection and lookAt" {
    const V3 = vec.Vector3(f32);
    inline for (.{
        .{ Config.Convention.vitellus, 0, 1 },
        .{ Config.Convention.metal, 0, 1 },
        .{ Config.Convention.webgpu, 0, 1 },
        .{ Config.Convention.opengl, -1, 1 },
        .{ Config.Convention.vulkan, 0, -1 },
        .{ Config.Convention.directx, 0, 1 },
    }) |case| {
        const c = case[0];
        const forward: f32 = if (c == .directx) 1 else -1; // view-space forward is +Z only in left-handed
        const p = perspectiveFor(c, f32, 1.0, 1.0, 0.5, 50.0);
        const o = orthographicFor(c, f32, -1, 1, -1, 1, 0.5, 50.0);
        try std.testing.expectApproxEqAbs(@as(f32, case[1]), transformPoint4x4(f32, p, V3.init(0, 0, 0.5 * forward)).z, 1e-5);
        try std.testing.expectApproxEqAbs(1.0, transformPoint4x4(f32, p, V3.init(0, 0, 50 * forward)).z, 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, case[1]), transformPoint4x4(f32, o, V3.init(0, 0, 0.5 * forward)).z, 1e-5);
        // clip-space Y direction: up for everyone but raw vulkan
        try std.testing.expect(transformPoint4x4(f32, p, V3.init(0, 1, forward)).y * @as(f32, case[2]) > 0);
        // looking straight down the convention's forward axis is the identity view
        const view = lookAtFor(c, f32, V3.zero, V3.init(0, 0, forward), V3.up);
        try std.testing.expect(approxEql(f32, 4, 4, view, identity(f32, 4), 1e-6));
    }
}

test "default config is vitellus" {
    try std.testing.expectEqual(Config.Convention.vitellus, config.convention);
}
