const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const quat = @import("quat.zig");
const complex = @import("complex.zig");
const isometry = @import("isometry.zig");

/// A 2D similarity transformation: uniform scale, then rotation, then translation.
///
/// Preserves angles and shape, but not size.
pub fn Similarity2(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float);

    return struct {
        isometry: isometry.Isometry2(T),
        scaling: T,

        const Self = @This();
        const Iso = isometry.Isometry2(T);
        const V = vec.Vector2(T);

        pub fn identity() Self {
            return .{ .isometry = Iso.identity(), .scaling = 1 };
        }

        /// From a translation, a rotation angle in radians, and a uniform scale factor.
        pub fn init(translation: V, angle: T, scaling: T) Self {
            std.debug.assert(scaling != 0);
            return .{ .isometry = Iso.init(translation, angle), .scaling = scaling };
        }

        pub fn fromParts(iso: Iso, scaling: T) Self {
            std.debug.assert(scaling != 0);
            return .{ .isometry = iso, .scaling = scaling };
        }

        pub fn fromIsometry(iso: Iso) Self {
            return .{ .isometry = iso, .scaling = 1 };
        }

        pub fn fromScaling(scaling: T) Self {
            std.debug.assert(scaling != 0);
            return .{ .isometry = Iso.identity(), .scaling = scaling };
        }

        /// Compose: `(self.mul(rhs)).transformPoint(p) == self.transformPoint(rhs.transformPoint(p))`.
        pub fn mul(self: Self, rhs: Self) Self {
            return .{
                .isometry = .{
                    .rotation = self.isometry.rotation.mul(rhs.isometry.rotation),
                    .translation = self.isometry.translation.add(
                        self.isometry.rotation.rotateVector(rhs.isometry.translation).scale(self.scaling),
                    ),
                },
                .scaling = self.scaling * rhs.scaling,
            };
        }

        pub fn inverse(self: Self) Self {
            const inv_rot = self.isometry.rotation.inverse();
            const inv_scale = 1 / self.scaling;
            return .{
                .isometry = .{
                    .rotation = inv_rot,
                    .translation = inv_rot.rotateVector(self.isometry.translation.negate()).scale(inv_scale),
                },
                .scaling = inv_scale,
            };
        }

        /// Scale, rotate, then translate a point.
        pub fn transformPoint(self: Self, p: V) V {
            return self.isometry.rotation.rotateVector(p.scale(self.scaling)).add(self.isometry.translation);
        }

        /// Scale and rotate a vector (translation does not apply to vectors).
        pub fn transformVector(self: Self, v: V) V {
            return self.isometry.rotation.rotateVector(v.scale(self.scaling));
        }

        pub fn inverseTransformPoint(self: Self, p: V) V {
            return self.isometry.rotation.inverseRotateVector(p.sub(self.isometry.translation)).scale(1 / self.scaling);
        }

        pub fn inverseTransformVector(self: Self, v: V) V {
            return self.isometry.rotation.inverseRotateVector(v).scale(1 / self.scaling);
        }

        /// Convert to a 3x3 homogeneous matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 3, 3) {
            var m = self.isometry.toHomogeneous();
            for (0..2) |c| {
                for (0..2) |r| {
                    m.data[c][r] *= self.scaling;
                }
            }
            return m;
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return self.isometry.approxEql(other.isometry, epsilon) and
                @abs(self.scaling - other.scaling) <= epsilon;
        }
    };
}

/// A 3D similarity transformation: uniform scale, then rotation, then translation.
///
/// Preserves angles and shape, but not size.
pub fn Similarity3(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float);

    return struct {
        isometry: isometry.Isometry3(T),
        scaling: T,

        const Self = @This();
        const Iso = isometry.Isometry3(T);
        const V = vec.Vector3(T);

        pub fn identity() Self {
            return .{ .isometry = Iso.identity(), .scaling = 1 };
        }

        /// From a translation, a scaled rotation axis (axis * angle), and a uniform scale factor.
        pub fn init(translation: V, axis_angle: V, scaling: T) Self {
            std.debug.assert(scaling != 0);
            return .{ .isometry = Iso.init(translation, axis_angle), .scaling = scaling };
        }

        pub fn fromParts(iso: Iso, scaling: T) Self {
            std.debug.assert(scaling != 0);
            return .{ .isometry = iso, .scaling = scaling };
        }

        pub fn fromIsometry(iso: Iso) Self {
            return .{ .isometry = iso, .scaling = 1 };
        }

        pub fn fromScaling(scaling: T) Self {
            std.debug.assert(scaling != 0);
            return .{ .isometry = Iso.identity(), .scaling = scaling };
        }

        /// Compose: `(self.mul(rhs)).transformPoint(p) == self.transformPoint(rhs.transformPoint(p))`.
        pub fn mul(self: Self, rhs: Self) Self {
            return .{
                .isometry = .{
                    .rotation = self.isometry.rotation.mul(rhs.isometry.rotation),
                    .translation = self.isometry.translation.add(
                        self.isometry.rotation.rotateVector(rhs.isometry.translation).scale(self.scaling),
                    ),
                },
                .scaling = self.scaling * rhs.scaling,
            };
        }

        pub fn inverse(self: Self) Self {
            const inv_rot = self.isometry.rotation.inverse();
            const inv_scale = 1 / self.scaling;
            return .{
                .isometry = .{
                    .rotation = inv_rot,
                    .translation = inv_rot.rotateVector(self.isometry.translation.negate()).scale(inv_scale),
                },
                .scaling = inv_scale,
            };
        }

        /// Scale, rotate, then translate a point.
        pub fn transformPoint(self: Self, p: V) V {
            return self.isometry.rotation.rotateVector(p.scale(self.scaling)).add(self.isometry.translation);
        }

        /// Scale and rotate a vector (translation does not apply to vectors).
        pub fn transformVector(self: Self, v: V) V {
            return self.isometry.rotation.rotateVector(v.scale(self.scaling));
        }

        pub fn inverseTransformPoint(self: Self, p: V) V {
            return self.isometry.rotation.inverseRotateVector(p.sub(self.isometry.translation)).scale(1 / self.scaling);
        }

        pub fn inverseTransformVector(self: Self, v: V) V {
            return self.isometry.rotation.inverseRotateVector(v).scale(1 / self.scaling);
        }

        /// Convert to a 4x4 homogeneous matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 4, 4) {
            var m = self.isometry.toHomogeneous();
            for (0..3) |c| {
                for (0..3) |r| {
                    m.data[c][r] *= self.scaling;
                }
            }
            return m;
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return self.isometry.approxEql(other.isometry, epsilon) and
                @abs(self.scaling - other.scaling) <= epsilon;
        }
    };
}

const testing = std.testing;
const Sim2 = Similarity2(f32);
const Sim3 = Similarity3(f32);
const V2 = vec.Vector2(f32);
const V3 = vec.Vector3(f32);

test "Similarity2 transforms a point" {
    const sim = Sim2.init(V2.init(1, 0), std.math.pi / 2.0, 2.0);
    // (1,0) scaled -> (2,0), rotated 90° -> (0,2), translated -> (1,2)
    const p = sim.transformPoint(V2.unit_x);
    try testing.expect(p.approxEql(V2.init(1, 2), 1e-5));
}

test "Similarity2 inverse round-trips" {
    const sim = Sim2.init(V2.init(-2, 3), 0.7, 1.5);
    const p = V2.init(4, -1);
    try testing.expect(sim.inverseTransformPoint(sim.transformPoint(p)).approxEql(p, 1e-4));
    try testing.expect(sim.inverse().mul(sim).approxEql(Sim2.identity(), 1e-5));
}

test "Similarity2 composition matches sequential application" {
    const a = Sim2.init(V2.init(1, 1), 0.3, 2.0);
    const b = Sim2.init(V2.init(-1, 2), -0.5, 0.5);
    const p = V2.init(2, 3);
    const composed = a.mul(b).transformPoint(p);
    const sequential = a.transformPoint(b.transformPoint(p));
    try testing.expect(composed.approxEql(sequential, 1e-4));
}

test "Similarity3 transforms a point" {
    const sim = Sim3.init(V3.init(0, 0, 1), V3.init(0, 0, std.math.pi / 2.0), 3.0);
    // (1,0,0) scaled -> (3,0,0), rotated 90° about z -> (0,3,0), translated -> (0,3,1)
    const p = sim.transformPoint(V3.unit_x);
    try testing.expect(p.approxEql(V3.init(0, 3, 1), 1e-4));
}

test "Similarity3 inverse round-trips" {
    const sim = Sim3.init(V3.init(1, 2, 3), V3.init(0.4, -0.2, 0.9), 0.75);
    const p = V3.init(-3, 5, 2);
    try testing.expect(sim.inverseTransformPoint(sim.transformPoint(p)).approxEql(p, 1e-4));
    try testing.expect(sim.inverse().mul(sim).approxEql(Sim3.identity(), 1e-5));
}

test "Similarity3 homogeneous matrix agrees with transformPoint" {
    const sim = Sim3.init(V3.init(2, -1, 0.5), V3.init(1, 1, 0), 2.5);
    const p = V3.init(1, 2, -3);
    const via_matrix = mat.transformPoint4x4(f32, sim.toHomogeneous(), p);
    try testing.expect(via_matrix.approxEql(sim.transformPoint(p), 1e-3));
}
