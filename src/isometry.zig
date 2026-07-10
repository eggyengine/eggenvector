const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const quat = @import("quat.zig");
const complex = @import("complex.zig");

/// A 2D direct isometry: a rotation followed by a translation (translation ∘ rotation).
///
/// Also known as a rigid-body transformation; it preserves distances and orientation.
pub fn Isometry2(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float);

    return struct {
        rotation: complex.UnitComplex(T),
        translation: vec.Vector2(T),

        const Self = @This();
        const Rot = complex.UnitComplex(T);
        const V = vec.Vector2(T);

        pub fn identity() Self {
            return .{ .rotation = Rot.identity(), .translation = V.zero };
        }

        /// From a translation and a rotation angle in radians.
        pub fn init(translation: V, angle: T) Self {
            return .{ .rotation = Rot.fromAngle(angle), .translation = translation };
        }

        pub fn fromParts(translation: V, rotation: Rot) Self {
            return .{ .rotation = rotation, .translation = translation };
        }

        pub fn fromTranslation(translation: V) Self {
            return .{ .rotation = Rot.identity(), .translation = translation };
        }

        pub fn fromRotation(rotation: Rot) Self {
            return .{ .rotation = rotation, .translation = V.zero };
        }

        /// Compose: `(self.mul(rhs)).transformPoint(p) == self.transformPoint(rhs.transformPoint(p))`.
        pub fn mul(self: Self, rhs: Self) Self {
            return .{
                .rotation = self.rotation.mul(rhs.rotation),
                .translation = self.translation.add(self.rotation.rotateVector(rhs.translation)),
            };
        }

        pub fn inverse(self: Self) Self {
            const inv_rot = self.rotation.inverse();
            return .{
                .rotation = inv_rot,
                .translation = inv_rot.rotateVector(self.translation.negate()),
            };
        }

        /// Rotate then translate a point.
        pub fn transformPoint(self: Self, p: V) V {
            return self.rotation.rotateVector(p).add(self.translation);
        }

        /// Rotate a vector (translation does not apply to vectors).
        pub fn transformVector(self: Self, v: V) V {
            return self.rotation.rotateVector(v);
        }

        pub fn inverseTransformPoint(self: Self, p: V) V {
            return self.rotation.inverseRotateVector(p.sub(self.translation));
        }

        pub fn inverseTransformVector(self: Self, v: V) V {
            return self.rotation.inverseRotateVector(v);
        }

        /// Interpolate the translation linearly and the rotation spherically.
        pub fn lerpSlerp(self: Self, other: Self, t: T) Self {
            return .{
                .rotation = self.rotation.slerp(other.rotation, t),
                .translation = self.translation.lerp(other.translation, t),
            };
        }

        /// Convert to a 3x3 homogeneous matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 3, 3) {
            var m = self.rotation.toHomogeneous();
            m.data[2] = .{ self.translation.x, self.translation.y, 1 };
            return m;
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return self.rotation.approxEql(other.rotation, epsilon) and
                self.translation.approxEql(other.translation, epsilon);
        }
    };
}

/// A 3D direct isometry: a rotation followed by a translation (translation ∘ rotation).
///
/// Also known as a rigid-body transformation; it preserves distances and orientation.
pub fn Isometry3(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float);

    return struct {
        rotation: quat.UnitQuaternion(T),
        translation: vec.Vector3(T),

        const Self = @This();
        const Rot = quat.UnitQuaternion(T);
        const V = vec.Vector3(T);

        pub fn identity() Self {
            return .{ .rotation = Rot.identity(), .translation = V.zero };
        }

        /// From a translation and a scaled rotation axis (axis * angle).
        pub fn init(translation: V, axis_angle: V) Self {
            return .{ .rotation = Rot.fromScaledAxis(axis_angle), .translation = translation };
        }

        pub fn fromParts(translation: V, rotation: Rot) Self {
            return .{ .rotation = rotation, .translation = translation };
        }

        pub fn fromTranslation(translation: V) Self {
            return .{ .rotation = Rot.identity(), .translation = translation };
        }

        pub fn fromRotation(rotation: Rot) Self {
            return .{ .rotation = rotation, .translation = V.zero };
        }

        /// Compose: `(self.mul(rhs)).transformPoint(p) == self.transformPoint(rhs.transformPoint(p))`.
        pub fn mul(self: Self, rhs: Self) Self {
            return .{
                .rotation = self.rotation.mul(rhs.rotation),
                .translation = self.translation.add(self.rotation.rotateVector(rhs.translation)),
            };
        }

        pub fn inverse(self: Self) Self {
            const inv_rot = self.rotation.inverse();
            return .{
                .rotation = inv_rot,
                .translation = inv_rot.rotateVector(self.translation.negate()),
            };
        }

        /// Rotate then translate a point.
        pub fn transformPoint(self: Self, p: V) V {
            return self.rotation.rotateVector(p).add(self.translation);
        }

        /// Rotate a vector (translation does not apply to vectors).
        pub fn transformVector(self: Self, v: V) V {
            return self.rotation.rotateVector(v);
        }

        pub fn inverseTransformPoint(self: Self, p: V) V {
            return self.rotation.inverseRotateVector(p.sub(self.translation));
        }

        pub fn inverseTransformVector(self: Self, v: V) V {
            return self.rotation.inverseRotateVector(v);
        }

        /// Interpolate the translation linearly and the rotation spherically.
        pub fn lerpSlerp(self: Self, other: Self, t: T) Self {
            return .{
                .rotation = self.rotation.slerp(other.rotation, t),
                .translation = self.translation.lerp(other.translation, t),
            };
        }

        /// Convert to a 4x4 homogeneous matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 4, 4) {
            var m = self.rotation.toHomogeneous();
            m.data[3] = .{ self.translation.x, self.translation.y, self.translation.z, 1 };
            return m;
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return self.rotation.approxEql(other.rotation, epsilon) and
                self.translation.approxEql(other.translation, epsilon);
        }
    };
}

const testing = std.testing;
const Iso2 = Isometry2(f32);
const Iso3 = Isometry3(f32);
const V2 = vec.Vector2(f32);
const V3 = vec.Vector3(f32);

test "Isometry2 transforms a point" {
    const iso = Iso2.init(V2.init(1, 2), std.math.pi / 2.0);
    const p = iso.transformPoint(V2.unit_x);
    try testing.expect(p.approxEql(V2.init(1, 3), 1e-5));
}

test "Isometry2 inverse round-trips" {
    const iso = Iso2.init(V2.init(-3, 5), 0.8);
    const p = V2.init(2, 7);
    try testing.expect(iso.inverseTransformPoint(iso.transformPoint(p)).approxEql(p, 1e-4));
    try testing.expect(iso.inverse().transformPoint(iso.transformPoint(p)).approxEql(p, 1e-4));
}

test "Isometry2 composition matches sequential application" {
    const a = Iso2.init(V2.init(1, 0), 0.4);
    const b = Iso2.init(V2.init(0, 2), -0.9);
    const p = V2.init(3, 4);
    const composed = a.mul(b).transformPoint(p);
    const sequential = a.transformPoint(b.transformPoint(p));
    try testing.expect(composed.approxEql(sequential, 1e-4));
}

test "Isometry2 homogeneous matrix agrees with transformPoint" {
    const iso = Iso2.init(V2.init(4, -1), 0.6);
    const m = iso.toHomogeneous();
    const p = V2.init(2, 3);
    const expected = iso.transformPoint(p);
    const hx = m.data[0][0] * p.x + m.data[1][0] * p.y + m.data[2][0];
    const hy = m.data[0][1] * p.x + m.data[1][1] * p.y + m.data[2][1];
    try testing.expectApproxEqAbs(expected.x, hx, 1e-5);
    try testing.expectApproxEqAbs(expected.y, hy, 1e-5);
}

test "Isometry3 transforms a point" {
    const rot = quat.UnitQuat.fromAxisAngle(V3.unit_z, std.math.pi / 2.0);
    const iso = Iso3.fromParts(V3.init(0, 0, 1), rot);
    const p = iso.transformPoint(V3.unit_x);
    try testing.expect(p.approxEql(V3.init(0, 1, 1), 1e-5));
}

test "Isometry3 inverse round-trips" {
    const iso = Iso3.init(V3.init(1, -2, 3), V3.init(0.3, 0.5, -0.2));
    const p = V3.init(5, 6, -7);
    try testing.expect(iso.inverseTransformPoint(iso.transformPoint(p)).approxEql(p, 1e-4));
    try testing.expect(iso.inverse().mul(iso).approxEql(Iso3.identity(), 1e-5));
}

test "Isometry3 homogeneous matrix agrees with transformPoint" {
    const iso = Iso3.init(V3.init(2, 1, -4), V3.init(1, 0.5, 0.25));
    const p = V3.init(-1, 2, 0.5);
    const via_matrix = mat.transformPoint4x4(f32, iso.toHomogeneous(), p);
    try testing.expect(via_matrix.approxEql(iso.transformPoint(p), 1e-4));
}
