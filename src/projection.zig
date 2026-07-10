const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const homogeneous = @import("homogeneous.zig");

/// A 3D perspective projection for computer graphics.
///
/// Stores the projection parameters and builds the homogeneous matrix on demand
/// using the library's Vulkan-flavoured convention (see `mat.perspective4x4`).
pub fn Perspective3(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float);

    return struct {
        aspect: T,
        fovy: T,
        znear: T,
        zfar: T,

        const Self = @This();

        pub fn init(aspect: T, fovy: T, znear: T, zfar: T) Self {
            std.debug.assert(znear != zfar);
            std.debug.assert(aspect != 0);
            return .{ .aspect = aspect, .fovy = fovy, .znear = znear, .zfar = zfar };
        }

        /// The homogeneous 4x4 projection matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 4, 4) {
            return mat.perspective4x4(T, self.fovy, self.aspect, self.znear, self.zfar);
        }

        /// View this projection as an invertible transformation.
        pub fn toProjective(self: Self) homogeneous.Projective3(T) {
            return homogeneous.Projective3(T).fromMatrixUnchecked(self.toHomogeneous());
        }

        /// Project a view-space point into normalized device coordinates (with perspective divide).
        pub fn projectPoint(self: Self, p: vec.Vector3(T)) vec.Vector3(T) {
            return mat.transformPoint4x4(T, self.toHomogeneous(), p);
        }

        /// Un-project a point from normalized device coordinates back into view space.
        pub fn unprojectPoint(self: Self, p: vec.Vector3(T)) vec.Vector3(T) {
            const inv = mat.inverse4x4(T, self.toHomogeneous()) orelse
                @panic("singular perspective matrix: invalid projection parameters");
            return mat.transformPoint4x4(T, inv, p);
        }
    };
}

/// A 3D orthographic projection for computer graphics.
///
/// Stores the view cuboid bounds and builds the homogeneous matrix on demand
/// using the library's Vulkan-flavoured convention (see `mat.orthographic4x4`).
pub fn Orthographic3(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float);

    return struct {
        left: T,
        right: T,
        bottom: T,
        top: T,
        znear: T,
        zfar: T,

        const Self = @This();

        pub fn init(left: T, right: T, bottom: T, top: T, znear: T, zfar: T) Self {
            std.debug.assert(left != right);
            std.debug.assert(bottom != top);
            std.debug.assert(znear != zfar);
            return .{ .left = left, .right = right, .bottom = bottom, .top = top, .znear = znear, .zfar = zfar };
        }

        /// The homogeneous 4x4 projection matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 4, 4) {
            return mat.orthographic4x4(T, self.left, self.right, self.bottom, self.top, self.znear, self.zfar);
        }

        /// View this projection as an invertible (in fact, affine) transformation.
        pub fn toProjective(self: Self) homogeneous.Projective3(T) {
            return homogeneous.Projective3(T).fromMatrixUnchecked(self.toHomogeneous());
        }

        /// Project a view-space point into normalized device coordinates.
        pub fn projectPoint(self: Self, p: vec.Vector3(T)) vec.Vector3(T) {
            return mat.transformPoint4x4(T, self.toHomogeneous(), p);
        }

        /// Un-project a point from normalized device coordinates back into view space.
        pub fn unprojectPoint(self: Self, p: vec.Vector3(T)) vec.Vector3(T) {
            const inv = mat.inverse4x4(T, self.toHomogeneous()) orelse
                @panic("singular orthographic matrix: invalid projection parameters");
            return mat.transformPoint4x4(T, inv, p);
        }
    };
}

const testing = std.testing;
const V3 = vec.Vector3(f32);

test "Perspective3 matches mat.perspective4x4" {
    const persp = Perspective3(f32).init(16.0 / 9.0, 1.2, 0.1, 100.0);
    const expected = mat.perspective4x4(f32, 1.2, 16.0 / 9.0, 0.1, 100.0);
    try testing.expect(mat.eql(f32, 4, 4, persp.toHomogeneous(), expected));
}

test "Perspective3 near plane center projects to depth zero" {
    const persp = Perspective3(f32).init(1.0, std.math.pi / 2.0, 0.5, 50.0);
    const projected = persp.projectPoint(V3.init(0, 0, -0.5));
    try testing.expectApproxEqAbs(0.0, projected.x, 1e-6);
    try testing.expectApproxEqAbs(0.0, projected.y, 1e-6);
    try testing.expectApproxEqAbs(0.0, projected.z, 1e-6);
}

test "Perspective3 project/unproject round-trips" {
    const persp = Perspective3(f32).init(16.0 / 9.0, 1.0, 0.1, 100.0);
    const p = V3.init(1.5, -0.7, -20.0);
    const back = persp.unprojectPoint(persp.projectPoint(p));
    try testing.expect(back.approxEql(p, 1e-3));
}

test "Orthographic3 matches mat.orthographic4x4" {
    const ortho = Orthographic3(f32).init(-2, 2, -1, 1, 0.1, 10.0);
    const expected = mat.orthographic4x4(f32, -2, 2, -1, 1, 0.1, 10.0);
    try testing.expect(mat.eql(f32, 4, 4, ortho.toHomogeneous(), expected));
}

test "Orthographic3 project/unproject round-trips" {
    const ortho = Orthographic3(f32).init(-10, 10, -5, 5, 0.5, 100.0);
    const p = V3.init(3, -2, 40);
    const back = ortho.unprojectPoint(ortho.projectPoint(p));
    try testing.expect(back.approxEql(p, 1e-4));
}
