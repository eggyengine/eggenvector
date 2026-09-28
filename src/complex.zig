const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");

/// A 2D rotation represented as a unit complex number (cos θ + i·sin θ).
///
/// The 2D analogue of a unit quaternion; composition adds Euler angles.
pub fn UnitComplex(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float);

    return struct {
        re: T,
        im: T,

        const Self = @This();

        /// The identity rotation (angle 0).
        pub fn identity() Self {
            return .{ .re = 1, .im = 0 };
        }

        /// Create from a rotation angle in radians.
        pub fn fromAngle(angle_rads: T) Self {
            return .{ .re = @cos(angle_rads), .im = @sin(angle_rads) };
        }

        /// Create from cos/sin parts without checking the unit-norm invariant.
        pub fn newUnchecked(re: T, im: T) Self {
            return .{ .re = re, .im = im };
        }

        /// The rotation that maps direction `a` onto direction `b`.
        pub fn rotationBetween(a: vec.Vector2(T), b: vec.Vector2(T)) Self {
            const na = a.normalize();
            const nb = b.normalize();
            return .{ .re = na.dot(nb), .im = na.x * nb.y - na.y * nb.x };
        }

        /// The rotation angle in radians, in (-π, π].
        pub fn angle(self: Self) T {
            return std.math.atan2(self.im, self.re);
        }

        /// Compose two rotations (angles add).
        pub fn mul(self: Self, other: Self) Self {
            return .{
                .re = self.re * other.re - self.im * other.im,
                .im = self.re * other.im + self.im * other.re,
            };
        }

        /// The inverse rotation (the conjugate, since the norm is one).
        pub fn inverse(self: Self) Self {
            return .{ .re = self.re, .im = -self.im };
        }

        /// Rotate a 2D vector.
        pub fn rotateVector(self: Self, v: vec.Vector2(T)) vec.Vector2(T) {
            return .{
                .x = self.re * v.x - self.im * v.y,
                .y = self.im * v.x + self.re * v.y,
            };
        }

        /// Rotate a 2D vector by the inverse of this rotation.
        pub fn inverseRotateVector(self: Self, v: vec.Vector2(T)) vec.Vector2(T) {
            return self.inverse().rotateVector(v);
        }

        /// Spherical interpolation along the shortest arc.
        pub fn slerp(self: Self, other: Self, t: T) Self {
            const delta = other.mul(self.inverse()).angle();
            return fromAngle(delta * t).mul(self);
        }

        /// Raise to a power (a fraction of the rotation).
        pub fn powSelf(self: Self, t: T) Self {
            return fromAngle(self.angle() * t);
        }

        /// Re-normalize to correct accumulated floating point drift.
        pub fn renormalize(self: Self) Self {
            const len = @sqrt(self.re * self.re + self.im * self.im);
            return .{ .re = self.re / len, .im = self.im / len };
        }

        /// Convert to a 2x2 rotation matrix.
        pub fn toRotationMatrix(self: Self) mat.Matrix(T, 2, 2) {
            return .{
                .data = .{
                    .{ self.re, self.im }, // col 0
                    .{ -self.im, self.re }, // col 1
                },
            };
        }

        /// Convert to a 3x3 homogeneous matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 3, 3) {
            return .{
                .data = .{
                    .{ self.re, self.im, 0 }, // col 0
                    .{ -self.im, self.re, 0 }, // col 1
                    .{ 0, 0, 1 }, // col 2
                },
            };
        }

        /// Approximate equality: true when every component differs by at most `epsilon`.
        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.re - other.re) <= epsilon and @abs(self.im - other.im) <= epsilon;
        }
    };
}

const UC = UnitComplex(f32);
const V2 = vec.Vector2(f32);

test "UnitComplex rotates a vector" {
    const r = UC.fromAngle(std.math.pi / 2.0);
    const v = r.rotateVector(V2.unit_x);
    try std.testing.expectApproxEqAbs(0.0, v.x, 1e-6);
    try std.testing.expectApproxEqAbs(1.0, v.y, 1e-6);
}

test "UnitComplex composition adds angles" {
    const a = UC.fromAngle(0.3);
    const b = UC.fromAngle(0.5);
    try std.testing.expectApproxEqAbs(0.8, a.mul(b).angle(), 1e-6);
}

test "UnitComplex inverse round-trips" {
    const r = UC.fromAngle(1.2);
    const v = V2{ .x = 3, .y = -2 };
    const back = r.inverse().rotateVector(r.rotateVector(v));
    try std.testing.expect(back.approxEql(v, 1e-5));
}

test "UnitComplex rotationBetween" {
    const r = UC.rotationBetween(V2.unit_x, V2.unit_y);
    try std.testing.expectApproxEqAbs(std.math.pi / 2.0, r.angle(), 1e-6);
}

test "UnitComplex slerp midpoint" {
    const a = UC.fromAngle(0.0);
    const b = UC.fromAngle(1.0);
    try std.testing.expectApproxEqAbs(0.5, a.slerp(b, 0.5).angle(), 1e-6);
}

test "UnitComplex matches its rotation matrix" {
    const r = UC.fromAngle(0.7);
    const m = r.toRotationMatrix();
    const v = V2{ .x = 1, .y = 2 };
    const rotated = r.rotateVector(v);
    const mx = m.data[0][0] * v.x + m.data[1][0] * v.y;
    const my = m.data[0][1] * v.x + m.data[1][1] * v.y;
    try std.testing.expectApproxEqAbs(rotated.x, mx, 1e-6);
    try std.testing.expectApproxEqAbs(rotated.y, my, 1e-6);
}
