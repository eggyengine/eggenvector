const std = @import("std");
const vec = @import("vec.zig");

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

pub fn isPermittedType(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .float => true,
        .int => true,
        .comptime_float => true,
        .comptime_int => true,
        else => false,
    };
}

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
/// T must be a permitted numeric type (validated at comptime).
pub fn Matrix(comptime T: type, comptime rows: usize, comptime cols: usize) type {
    comptime assertPermittedType(T);

    return struct {
        data: [cols][rows]T,

        pub const Row = rows;
        pub const Col = cols;
        pub const Scalar = T;
    };
}

/// Create a matrix with all elements set to a value.
pub fn splat(comptime T: type, comptime rows: usize, comptime cols: usize, value: T) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    for (0..cols) |c| {
        for (0..rows) |r| {
            result.data[c][r] = value;
        }
    }
    return result;
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
    for (0..size) |i| {
        result.data[i][i] = 1;
    }
    return result;
}

/// Create a matrix from a 2D array.
pub fn fromArray(comptime T: type, comptime rows: usize, comptime cols: usize, data: [rows][cols]T) Matrix(T, rows, cols) {
    return Matrix(T, rows, cols){ .data = data };
}

/// Get element at row r, column c.
pub fn get(comptime T: type, comptime rows: usize, comptime cols: usize, m: Matrix(T, rows, cols), r: usize, c: usize) T {
    return m.data[c][r];
}

/// Set element at row r, column c.
pub fn set(comptime T: type, comptime rows: usize, comptime cols: usize, m: *Matrix(T, rows, cols), r: usize, c: usize, value: T) void {
    m.data[c][r] = value;
}

/// Matrix addition.
pub fn add(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols)) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    for (0..cols) |c| {
        for (0..rows) |r| {
            result.data[c][r] = a.data[c][r] + b.data[c][r];
        }
    }
    return result;
}

/// Matrix subtraction.
pub fn sub(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols)) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    for (0..cols) |c| {
        for (0..rows) |r| {
            result.data[c][r] = a.data[c][r] - b.data[c][r];
        }
    }
    return result;
}

/// Scalar multiplication.
pub fn scale(comptime T: type, comptime rows: usize, comptime cols: usize, m: Matrix(T, rows, cols), scalar: T) Matrix(T, rows, cols) {
    var result: Matrix(T, rows, cols) = undefined;
    for (0..cols) |c| {
        for (0..rows) |r| {
            result.data[c][r] = m.data[c][r] * scalar;
        }
    }
    return result;
}

/// Matrix multiplication.
pub fn mul(
    comptime T: type,
    comptime a_rows: usize,
    comptime a_cols: usize,
    comptime b_cols: usize,
    a: Matrix(T, a_rows, a_cols),
    b: Matrix(T, a_cols, b_cols),
) Matrix(T, a_rows, b_cols) {
    var result = zero(T, a_rows, b_cols);
    for (0..a_rows) |r| {
        for (0..b_cols) |c| {
            var sum: T = 0;
            for (0..a_cols) |k| {
                sum += a.data[k][r] * b.data[c][k];
            }
            result.data[c][r] = sum;
        }
    }
    return result;
}

/// Transpose the matrix.
pub fn transpose(comptime T: type, comptime rows: usize, comptime cols: usize, m: Matrix(T, rows, cols)) Matrix(T, cols, rows) {
    var result: Matrix(T, cols, rows) = undefined;
    for (0..cols) |c| {
        for (0..rows) |r| {
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
    return vec.Vector3(T){
        .x = m.data[0][0] * v.x + m.data[1][0] * v.y + m.data[2][0] * v.z,
        .y = m.data[0][1] * v.x + m.data[1][1] * v.y + m.data[2][1] * v.z,
        .z = m.data[0][2] * v.x + m.data[1][2] * v.y + m.data[2][2] * v.z,
    };
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

/// Create a look-at view matrix.
/// Requires floating point type.
pub fn lookAt4x4(comptime T: type, eye: vec.Vector3(T), target: vec.Vector3(T), up: vec.Vector3(T)) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("lookAt4x4 requires a floating point type");
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

/// Create a perspective projection matrix (right-handed, zero-to-one depth, column-major).
///
/// fov_y: vertical field of view in radians
/// aspect: width / height
/// near, far: near and far clipping planes
/// Requires floating point type.
pub fn perspective4x4(comptime T: type, fov_y: T, aspect: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("perspective4x4 requires a floating point type");
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

/// Create an orthographic projection matrix.
/// Requires floating point type.
///
/// Supports vulkan-based contexts.
pub fn orthographic4x4(comptime T: type, left: T, right: T, bottom: T, top: T, near: T, far: T) Matrix(T, 4, 4) {
    comptime if (!isFloatType(T)) @compileError("orthographic4x4 requires a floating point type");
    const width = right - left;
    const height = top - bottom;
    const depth = far - near;

    return Matrix(T, 4, 4){
        .data = .{
            .{ 2 / width, 0, 0, 0 },
            .{ 0, -2 / height, 0, 0 },
            .{ 0, 0, 1 / depth, 0 },
            .{ -(right + left) / width, (top + bottom) / height, -near / depth, 1 },
        },
    };
}

/// Transform a 4D vector by a 4x4 matrix.
pub fn transformVec4by4x4(comptime T: type, m: Matrix(T, 4, 4), v: vec.Vector4(T)) vec.Vector4(T) {
    return vec.Vector4(T){
        .x = m.data[0][0] * v.x + m.data[1][0] * v.y + m.data[2][0] * v.z + m.data[3][0] * v.w,
        .y = m.data[0][1] * v.x + m.data[1][1] * v.y + m.data[2][1] * v.z + m.data[3][1] * v.w,
        .z = m.data[0][2] * v.x + m.data[1][2] * v.y + m.data[2][2] * v.z + m.data[3][2] * v.w,
        .w = m.data[0][3] * v.x + m.data[1][3] * v.y + m.data[2][3] * v.z + m.data[3][3] * v.w,
    };
}

/// Transform a 3D point (w=1) by a 4x4 matrix.
/// Requires floating point type.
pub fn transformPoint4x4(comptime T: type, m: Matrix(T, 4, 4), v: vec.Vector3(T)) vec.Vector3(T) {
    comptime if (!isFloatType(T)) @compileError("transformPoint4x4 requires a floating point type");
    const w = m.data[0][3] * v.x + m.data[1][3] * v.y + m.data[2][3] * v.z + m.data[3][3];
    return vec.Vector3(T){
        .x = (m.data[0][0] * v.x + m.data[1][0] * v.y + m.data[2][0] * v.z + m.data[3][0]) / w,
        .y = (m.data[0][1] * v.x + m.data[1][1] * v.y + m.data[2][1] * v.z + m.data[3][1]) / w,
        .z = (m.data[0][2] * v.x + m.data[1][2] * v.y + m.data[2][2] * v.z + m.data[3][2]) / w,
    };
}

/// Transform a 3D direction (w=0, no translation) by a 4x4 matrix.
pub fn transformDirection4x4(comptime T: type, m: Matrix(T, 4, 4), v: vec.Vector3(T)) vec.Vector3(T) {
    return vec.Vector3(T){
        .x = m.data[0][0] * v.x + m.data[1][0] * v.y + m.data[2][0] * v.z,
        .y = m.data[0][1] * v.x + m.data[1][1] * v.y + m.data[2][1] * v.z,
        .z = m.data[0][2] * v.x + m.data[1][2] * v.y + m.data[2][2] * v.z,
    };
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

    const inv0 = Vec4{ .x = adj0.x * sign_a.x, .y = adj0.y * sign_b.y, .z = adj0.z * sign_a.z, .w = adj0.w * sign_b.w };
    const inv1 = Vec4{ .x = adj1.x * sign_b.x, .y = adj1.y * sign_a.y, .z = adj1.z * sign_b.z, .w = adj1.w * sign_a.w };
    const inv2 = Vec4{ .x = adj2.x * sign_a.x, .y = adj2.y * sign_b.y, .z = adj2.z * sign_a.z, .w = adj2.w * sign_b.w };
    const inv3 = Vec4{ .x = adj3.x * sign_b.x, .y = adj3.y * sign_a.y, .z = adj3.z * sign_b.z, .w = adj3.w * sign_a.w };

    const row0 = Vec4{ .x = m.data[0][0], .y = m.data[0][1], .z = m.data[0][2], .w = m.data[0][3] };
    const dot0 = Vec4{ .x = row0.x * inv0.x, .y = row0.y * inv0.y, .z = row0.z * inv0.z, .w = row0.w * inv0.w };
    const det = dot0.x + dot0.y + dot0.z + dot0.w;

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
/// This is the original perspective4x4 — kept for Vulkan convenience.
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
/// This is the original orthographic4x4 — kept for Vulkan convenience.
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
    result.z = result.z * 0.5 + 0.5;

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
        .z = win.z * 2 - 1,
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

/// Check if two matrices are approximately equal (element-wise).
pub fn approxEql(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols), epsilon: T) bool {
    for (0..cols) |c| {
        for (0..rows) |r| {
            if (@abs(a.data[c][r] - b.data[c][r]) > epsilon) return false;
        }
    }
    return true;
}

/// Check if two matrices are exactly equal.
pub fn eql(comptime T: type, comptime rows: usize, comptime cols: usize, a: Matrix(T, rows, cols), b: Matrix(T, rows, cols)) bool {
    for (0..cols) |c| {
        for (0..rows) |r| {
            if (a.data[c][r] != b.data[c][r]) return false;
        }
    }
    return true;
}

/// Compute the trace of a square matrix (sum of diagonal elements).
pub fn trace(comptime T: type, comptime size: usize, m: Matrix(T, size, size)) T {
    var result: T = 0;
    for (0..size) |i| {
        result += m.data[i][i];
    }
    return result;
}

pub const Mat2 = Matrix(f32, 2, 2);
pub const Mat3 = Matrix(f32, 3, 3);
pub const Mat4 = Matrix(f32, 4, 4);

pub const Mat2d = Matrix(f64, 2, 2);
pub const Mat3d = Matrix(f64, 3, 3);
pub const Mat4d = Matrix(f64, 4, 4);

pub const Mat2i = Matrix(i32, 2, 2);
pub const Mat3i = Matrix(i32, 3, 3);
pub const Mat4i = Matrix(i32, 4, 4);
