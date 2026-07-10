const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");

/// The category of a homogeneous-matrix-backed transformation.
pub const TransformCategory = enum {
    /// An arbitrary, possibly non-invertible transformation.
    general,
    /// An invertible transformation.
    projective,
    /// An affine transformation (the last row is `[0 ... 0 1]`).
    affine,
};

/// A transformation of `dim`-dimensional space stored as a `(dim+1)x(dim+1)`
/// homogeneous matrix, tagged with a comptime category.
fn HomogeneousBase(comptime T: type, comptime dim: usize, comptime category: TransformCategory) type {
    comptime std.debug.assert(dim == 2 or dim == 3);
    comptime if (!mat.isFloatType(T)) @compileError("transform types require a floating point type");
    const n = dim + 1;

    return struct {
        matrix: mat.Matrix(T, n, n),

        const Self = @This();
        pub const Point = if (dim == 2) vec.Vector2(T) else vec.Vector3(T);
        pub const Category = category;

        pub fn identity() Self {
            return .{ .matrix = mat.identity(T, n) };
        }

        /// Wrap a homogeneous matrix without validating the category invariant.
        pub fn fromMatrixUnchecked(m: mat.Matrix(T, n, n)) Self {
            return .{ .matrix = m };
        }

        /// The underlying homogeneous matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, n, n) {
            return self.matrix;
        }

        /// Compose two transformations (matrix product).
        pub fn mul(self: Self, rhs: Self) Self {
            return .{ .matrix = mat.mul(T, n, n, n, self.matrix, rhs.matrix) };
        }

        /// Attempt to invert; returns null if the matrix is singular.
        pub fn tryInverse(self: Self) ?Self {
            const inv = if (dim == 2)
                mat.inverse3x3(T, self.matrix)
            else
                mat.inverse4x4(T, self.matrix);
            return .{ .matrix = inv orelse return null };
        }

        /// Invert the transformation. Only available for the projective and affine
        /// categories, whose invariants guarantee invertibility.
        pub fn inverse(self: Self) Self {
            comptime if (category == .general)
                @compileError("a general Transform may be non-invertible; use tryInverse instead");
            return self.tryInverse() orelse @panic("singular matrix: transform category invariant violated");
        }

        /// Transform a point (homogeneous coordinate 1, with perspective divide).
        pub fn transformPoint(self: Self, p: Point) Point {
            if (dim == 2) {
                const d = self.matrix.data;
                const x = d[0][0] * p.x + d[1][0] * p.y + d[2][0];
                const y = d[0][1] * p.x + d[1][1] * p.y + d[2][1];
                const w = d[0][2] * p.x + d[1][2] * p.y + d[2][2];
                return .{ .x = x / w, .y = y / w };
            } else {
                return mat.transformPoint4x4(T, self.matrix, p);
            }
        }

        /// Transform a vector (homogeneous coordinate 0: no translation, no perspective divide).
        pub fn transformVector(self: Self, v: Point) Point {
            if (dim == 2) {
                const d = self.matrix.data;
                return .{
                    .x = d[0][0] * v.x + d[1][0] * v.y,
                    .y = d[0][1] * v.x + d[1][1] * v.y,
                };
            } else {
                return mat.transformDirection4x4(T, self.matrix, v);
            }
        }

        /// Forget the category, viewing this as a general Transform.
        pub fn toGeneral(self: Self) HomogeneousBase(T, dim, .general) {
            return .{ .matrix = self.matrix };
        }

        /// View an affine transformation as a projective one (affine maps are invertible).
        pub fn toProjective(self: Self) HomogeneousBase(T, dim, .projective) {
            comptime if (category == .general)
                @compileError("a general Transform is not guaranteed to be projective (invertible)");
            return .{ .matrix = self.matrix };
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return mat.approxEql(T, n, n, self.matrix, other.matrix, epsilon);
        }
    };
}

/// A general 2D transformation that does not have to be invertible, stored as a homogeneous 3x3 matrix.
pub fn Transform2(comptime T: type) type {
    return HomogeneousBase(T, 2, .general);
}

/// A general 3D transformation that does not have to be invertible, stored as a homogeneous 4x4 matrix.
pub fn Transform3(comptime T: type) type {
    return HomogeneousBase(T, 3, .general);
}

/// An invertible 2D transformation stored as a homogeneous 3x3 matrix.
pub fn Projective2(comptime T: type) type {
    return HomogeneousBase(T, 2, .projective);
}

/// An invertible 3D transformation stored as a homogeneous 4x4 matrix.
pub fn Projective3(comptime T: type) type {
    return HomogeneousBase(T, 3, .projective);
}

/// A 2D affine transformation stored as a homogeneous 3x3 matrix.
pub fn Affine2(comptime T: type) type {
    return HomogeneousBase(T, 2, .affine);
}

/// A 3D affine transformation stored as a homogeneous 4x4 matrix.
pub fn Affine3(comptime T: type) type {
    return HomogeneousBase(T, 3, .affine);
}

const testing = std.testing;
const V2 = vec.Vector2(f32);
const V3 = vec.Vector3(f32);

test "Affine2 translates a point" {
    var m = mat.identity(f32, 3);
    m.data[2] = .{ 5, -2, 1 }; // translation column
    const a = Affine2(f32).fromMatrixUnchecked(m);
    try testing.expect(a.transformPoint(V2.init(1, 1)).approxEql(V2.init(6, -1), 1e-6));
    // vectors are unaffected by translation
    try testing.expect(a.transformVector(V2.init(1, 1)).approxEql(V2.init(1, 1), 1e-6));
}

test "Affine2 inverse round-trips" {
    const a = Affine2(f32).fromMatrixUnchecked(mat.mul(
        f32,
        3,
        3,
        3,
        .{ .data = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 3, 4, 1 } } },
        mat.fromArray(f32, 3, 3, .{ .{ 2, 0, 0 }, .{ 0, 2, 0 }, .{ 0, 0, 1 } }),
    ));
    const p = V2.init(7, -3);
    try testing.expect(a.inverse().transformPoint(a.transformPoint(p)).approxEql(p, 1e-4));
}

test "Affine3 rotation matrix agrees with mat helpers" {
    const rot = mat.rotationZ4x4(f32, 0.9);
    const a = Affine3(f32).fromMatrixUnchecked(rot);
    const p = V3.init(1, 2, 3);
    try testing.expect(a.transformPoint(p).approxEql(mat.transformPoint4x4(f32, rot, p), 1e-6));
}

test "Projective3 perspective divide and inverse" {
    const proj = Projective3(f32).fromMatrixUnchecked(mat.perspective4x4(f32, 1.2, 16.0 / 9.0, 0.1, 100.0));
    const p = V3.init(0.5, -0.3, -10.0);
    const projected = proj.transformPoint(p);
    const back = proj.inverse().transformPoint(projected);
    try testing.expect(back.approxEql(p, 1e-3));
}

test "Transform2 tryInverse returns null for a singular matrix" {
    const t = Transform2(f32).fromMatrixUnchecked(mat.zero(f32, 3, 3));
    try testing.expect(t.tryInverse() == null);
    try testing.expect(Transform2(f32).identity().tryInverse() != null);
}

test "Transform3 composition matches matrix product" {
    const a = Transform3(f32).fromMatrixUnchecked(mat.translation4x4(f32, 1, 2, 3));
    const b = Transform3(f32).fromMatrixUnchecked(mat.scaling4x4(f32, 2, 2, 2));
    const p = V3.init(1, 1, 1);
    const composed = a.mul(b).transformPoint(p);
    const sequential = a.transformPoint(b.transformPoint(p));
    try testing.expect(composed.approxEql(sequential, 1e-5));
}

test "Affine3 upgrades to projective and general" {
    const a = Affine3(f32).fromMatrixUnchecked(mat.translation4x4(f32, 1, 0, 0));
    const p = a.toProjective();
    const g = a.toGeneral();
    const point = V3.init(0, 0, 0);
    try testing.expect(p.transformPoint(point).approxEql(a.transformPoint(point), 1e-6));
    try testing.expect(g.transformPoint(point).approxEql(a.transformPoint(point), 1e-6));
}
