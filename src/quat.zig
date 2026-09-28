const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");

/// A quaternion (x, y, z, w) stored in a single 4-lane SIMD `@Vector`.
pub fn Quaternion(comptime T: type) type {
    return struct {
        const Vec4 = @Vector(4, T);
        const Self = @This();

        v: Vec4,

        /// Create quaternion from components (x, y, z, w).
        pub fn init(vx: T, vy: T, vz: T, vw: T) Self {
            return .{ .v = Vec4{ vx, vy, vz, vw } };
        }

        /// Create quaternion from vector and scalar.
        pub fn fromVecScalar(xyz: vec.Vector3(T), scalar_w: T) Self {
            return .{ .v = Vec4{ xyz.x, xyz.y, xyz.z, scalar_w } };
        }

        /// Create identity quaternion (no rotation).
        pub fn identity() Self {
            return .{ .v = Vec4{ 0, 0, 0, 1 } };
        }

        /// Create quaternion from axis-angle representation.
        pub fn fromAxisAngle(axis: vec.Vector3(T), angle: T) Self {
            const half_angle = angle / 2;
            const sin_half = @sin(half_angle);
            const cos_half = @cos(half_angle);
            const n = axis.normalize();
            return .{ .v = Vec4{ n.x * sin_half, n.y * sin_half, n.z * sin_half, cos_half } };
        }

        /// Create quaternion from Euler angles (pitch, yaw, roll in radians).
        /// Order: Z (roll) * Y (yaw) * X (pitch)
        pub fn fromEuler(pitch: T, yaw: T, roll: T) Self {
            const cp = @cos(pitch / 2);
            const sp = @sin(pitch / 2);
            const cy = @cos(yaw / 2);
            const sy = @sin(yaw / 2);
            const cr = @cos(roll / 2);
            const sr = @sin(roll / 2);

            return .{
                .v = Vec4{
                    sr * cp * cy - cr * sp * sy, // x
                    cr * sp * cy + sr * cp * sy, // y
                    cr * cp * sy - sr * sp * cy, // z
                    cr * cp * cy + sr * sp * sy, // w
                },
            };
        }

        /// Create quaternion that rotates from one direction to another.
        pub fn fromToRotation(from: vec.Vector3(T), to: vec.Vector3(T)) Self {
            const f = from.normalize();
            const t = to.normalize();
            const d = f.dot(t);

            if (d >= 1.0 - 1e-6) {
                return identity();
            }

            if (d <= -1.0 + 1e-6) {
                const x_axis = vec.Vector3(T){ .x = 1, .y = 0, .z = 0 };
                var axis = x_axis.cross(f);
                if (axis.lengthSquared() < 1e-6) {
                    const y_axis = vec.Vector3(T){ .x = 0, .y = 1, .z = 0 };
                    axis = y_axis.cross(f);
                }
                return fromAxisAngle(axis.normalize(), std.math.pi);
            }

            const axis = f.cross(t);
            const s = @sqrt((1 + d) * 2);
            const inv_s = 1 / s;

            return .{ .v = Vec4{
                axis.x * inv_s,
                axis.y * inv_s,
                axis.z * inv_s,
                s / 2,
            } };
        }

        /// Create quaternion from rotation matrix.
        pub fn fromMatrix(m: mat.Matrix(T, 4, 4)) Self {
            const trace = m.data[0][0] + m.data[1][1] + m.data[2][2];

            if (trace > 0) {
                const s = @sqrt(trace + 1) * 2;
                return .{ .v = Vec4{
                    (m.data[1][2] - m.data[2][1]) / s,
                    (m.data[2][0] - m.data[0][2]) / s,
                    (m.data[0][1] - m.data[1][0]) / s,
                    s / 4,
                } };
            } else if (m.data[0][0] > m.data[1][1] and m.data[0][0] > m.data[2][2]) {
                const s = @sqrt(1 + m.data[0][0] - m.data[1][1] - m.data[2][2]) * 2;
                return .{ .v = Vec4{
                    s / 4,
                    (m.data[1][0] + m.data[0][1]) / s,
                    (m.data[2][0] + m.data[0][2]) / s,
                    (m.data[1][2] - m.data[2][1]) / s,
                } };
            } else if (m.data[1][1] > m.data[2][2]) {
                const s = @sqrt(1 + m.data[1][1] - m.data[0][0] - m.data[2][2]) * 2;
                return .{ .v = Vec4{
                    (m.data[1][0] + m.data[0][1]) / s,
                    s / 4,
                    (m.data[2][1] + m.data[1][2]) / s,
                    (m.data[2][0] - m.data[0][2]) / s,
                } };
            } else {
                const s = @sqrt(1 + m.data[2][2] - m.data[0][0] - m.data[1][1]) * 2;
                return .{ .v = Vec4{
                    (m.data[2][0] + m.data[0][2]) / s,
                    (m.data[2][1] + m.data[1][2]) / s,
                    s / 4,
                    (m.data[0][1] - m.data[1][0]) / s,
                } };
            }
        }

        /// The x (imaginary i) component.
        pub fn x(self: Self) T {
            return self.v[0];
        }
        /// The y (imaginary j) component.
        pub fn y(self: Self) T {
            return self.v[1];
        }
        /// The z (imaginary k) component.
        pub fn z(self: Self) T {
            return self.v[2];
        }
        /// The w (real) component.
        pub fn w(self: Self) T {
            return self.v[3];
        }

        /// Get the vector (imaginary) part.
        pub fn vector(self: Self) vec.Vector3(T) {
            return .{ .x = self.v[0], .y = self.v[1], .z = self.v[2] };
        }

        /// Get the scalar (real) part.
        pub fn scalar(self: Self) T {
            return self.v[3];
        }

        /// SIMD dot product.
        inline fn dot4(a: Vec4, b: Vec4) T {
            const prod = a * b;
            return prod[0] + prod[1] + prod[2] + prod[3];
        }

        /// Quaternion dot product.
        pub fn dot(self: Self, other: Self) T {
            return dot4(self.v, other.v);
        }

        /// Squared length of quaternion (SIMD).
        pub fn lengthSquared(self: Self) T {
            return dot4(self.v, self.v);
        }

        /// Length (magnitude) of quaternion.
        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        /// Normalize the quaternion.
        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len < 1e-10) return identity();
            const inv_len: Vec4 = @splat(1 / len);
            return .{ .v = self.v * inv_len };
        }

        /// Quaternion conjugate (negate vector part).
        pub fn conjugate(self: Self) Self {
            return .{ .v = Vec4{ -self.v[0], -self.v[1], -self.v[2], self.v[3] } };
        }

        /// Quaternion inverse (conjugate / length²).
        pub fn inverse(self: Self) Self {
            const len_sq = self.lengthSquared();
            if (len_sq < 1e-10) return identity();
            const inv_len_sq: Vec4 = @splat(1 / len_sq);
            const conj = self.conjugate();
            return .{ .v = conj.v * inv_len_sq };
        }

        /// Negate quaternion.
        pub fn negate(self: Self) Self {
            return .{ .v = -self.v };
        }

        /// Quaternion addition (SIMD).
        pub fn add(self: Self, other: Self) Self {
            return .{ .v = self.v + other.v };
        }

        /// Quaternion subtraction (SIMD).
        pub fn sub(self: Self, other: Self) Self {
            return .{ .v = self.v - other.v };
        }

        /// Scalar multiplication (SIMD).
        pub fn scale(self: Self, s: T) Self {
            const sv: Vec4 = @splat(s);
            return .{ .v = self.v * sv };
        }

        /// Quaternion multiplication (Hamilton product).
        pub fn mul(self: Self, other: Self) Self {
            const a = self.v;
            const b = other.v;

            return .{
                .v = Vec4{
                    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], // x
                    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0], // y
                    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], // z
                    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2], // w
                },
            };
        }

        /// Rotate quaternion by an additional angle around an axis.
        ///
        /// Equivalent to: self * Quaternion.fromAxisAngle(axis, angle)
        pub fn rotate(self: Self, angle: T, axis: vec.Vector3(T)) Self {
            const len = axis.length();
            const normalized_axis = if (@abs(len - 1) > 0.001)
                vec.Vector3(T){
                    .x = axis.x / len,
                    .y = axis.y / len,
                    .z = axis.z / len,
                }
            else
                axis;

            const half_angle = angle * 0.5;
            const sin_half = @sin(half_angle);
            const cos_half = @cos(half_angle);

            const rotation = Self{
                .v = Vec4{
                    normalized_axis.x * sin_half,
                    normalized_axis.y * sin_half,
                    normalized_axis.z * sin_half,
                    cos_half,
                },
            };

            return self.mul(rotation);
        }

        /// Rotate a 3D vector by this quaternion.
        ///
        /// v' = q * v * q^-1
        pub fn rotateVector(self: Self, v: vec.Vector3(T)) vec.Vector3(T) {
            const u = self.vector();
            const s = self.v[3];

            const uv = u.cross(v);
            const uuv = u.cross(uv);

            return .{
                .x = v.x + 2 * (s * uv.x + uuv.x),
                .y = v.y + 2 * (s * uv.y + uuv.y),
                .z = v.z + 2 * (s * uv.z + uuv.z),
            };
        }

        /// Linear interpolation (not normalised, use nlerp for rotations).
        pub fn lerp(self: Self, other: Self, t: T) Self {
            const one_t: Vec4 = @splat(1 - t);
            const tv: Vec4 = @splat(t);
            return .{ .v = self.v * one_t + other.v * tv };
        }

        /// Normalized linear interpolation (fast approximation of slerp).
        pub fn nlerp(self: Self, other: Self, t: T) Self {
            const d = self.dot(other);
            const b = if (d < 0) other.negate() else other;
            return self.lerp(b, t).normalize();
        }

        /// Spherical linear interpolation (constant angular velocity).
        pub fn slerp(self: Self, other: Self, t: T) Self {
            var d = self.dot(other);
            var b = other;

            if (d < 0) {
                d = -d;
                b = other.negate();
            }

            if (d > 0.9995) {
                return self.nlerp(b, t);
            }

            const theta = std.math.acos(d);
            const sin_theta = @sin(theta);
            const s0 = @sin((1 - t) * theta) / sin_theta;
            const s1 = @sin(t * theta) / sin_theta;

            const s0v: Vec4 = @splat(s0);
            const s1v: Vec4 = @splat(s1);
            return .{ .v = self.v * s0v + b.v * s1v };
        }

        /// Convert to 4x4 rotation matrix.
        pub fn toMatrix(self: Self) mat.Matrix(T, 4, 4) {
            const q = self.normalize();
            const xx = q.v[0] * q.v[0];
            const yy = q.v[1] * q.v[1];
            const zz = q.v[2] * q.v[2];
            const xy = q.v[0] * q.v[1];
            const xz = q.v[0] * q.v[2];
            const yz = q.v[1] * q.v[2];
            const wx = q.v[3] * q.v[0];
            const wy = q.v[3] * q.v[1];
            const wz = q.v[3] * q.v[2];

            return mat.Matrix(T, 4, 4){
                .data = .{
                    .{ 1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0 }, // col 0
                    .{ 2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0 }, // col 1
                    .{ 2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0 }, // col 2
                    .{ 0, 0, 0, 1 }, // col 3
                },
            };
        }

        /// Convert to Euler angles (pitch, yaw, roll).
        /// Returns (pitch, yaw, roll) in radians.
        pub fn toEuler(self: Self) struct { pitch: T, yaw: T, roll: T } {
            const q = self.v;

            // Roll (x-axis rotation)
            const sinr_cosp = 2 * (q[3] * q[0] + q[1] * q[2]);
            const cosr_cosp = 1 - 2 * (q[0] * q[0] + q[1] * q[1]);
            const roll = std.math.atan2(sinr_cosp, cosr_cosp);

            // Pitch (y-axis rotation)
            const sinp = 2 * (q[3] * q[1] - q[2] * q[0]);
            const pitch = if (@abs(sinp) >= 1)
                std.math.copysign(std.math.pi / 2, sinp)
            else
                std.math.asin(sinp);

            // Yaw (z-axis rotation)
            const siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1]);
            const cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
            const yaw = std.math.atan2(siny_cosp, cosy_cosp);

            return .{ .pitch = pitch, .yaw = yaw, .roll = roll };
        }

        /// Get the axis of rotation.
        pub fn getAxis(self: Self) vec.Vector3(T) {
            const sin_sq = 1 - self.v[3] * self.v[3];
            if (sin_sq < 1e-10) {
                return .{ .x = 1, .y = 0, .z = 0 }; // No rotation
            }
            const inv_sin = 1 / @sqrt(sin_sq);
            return .{
                .x = self.v[0] * inv_sin,
                .y = self.v[1] * inv_sin,
                .z = self.v[2] * inv_sin,
            };
        }

        /// Get the angle of rotation in radians.
        pub fn getAngle(self: Self) T {
            return 2 * std.math.acos(@min(@max(self.v[3], -1), 1));
        }

        /// Check if two quaternions are approximately equal.
        pub fn approxEqual(self: Self, other: Self, epsilon: T) bool {
            const diff = self.v - other.v;
            const abs_diff = @abs(diff);
            return abs_diff[0] < epsilon and abs_diff[1] < epsilon and
                abs_diff[2] < epsilon and abs_diff[3] < epsilon;
        }

        /// Check if quaternion is normalized.
        pub fn isNormalized(self: Self, epsilon: T) bool {
            return @abs(self.lengthSquared() - 1) < epsilon;
        }
    };
}

/// An `f32` quaternion.
pub const Quat = Quaternion(f32);
/// An `f64` quaternion.
pub const Quatd = Quaternion(f64);

/// A quaternion with a norm equal to one, representing a 3D rotation.
///
/// All operations preserve the unit-norm invariant, so `inverse` is the
/// cheap conjugate rather than a full division by the squared norm.
pub fn UnitQuaternion(comptime T: type) type {
    return struct {
        quat: Quaternion(T),

        const Self = @This();
        const Q = Quaternion(T);

        /// The identity rotation.
        pub fn identity() Self {
            return .{ .quat = Q.identity() };
        }

        /// Normalize an arbitrary quaternion into a unit quaternion.
        pub fn fromQuaternion(q: Q) Self {
            return .{ .quat = q.normalize() };
        }

        /// Wrap without normalizing. The caller guarantees the unit-norm invariant.
        pub fn newUnchecked(q: Q) Self {
            return .{ .quat = q };
        }

        /// From a rotation axis (need not be normalized, must be non-zero) and an angle in radians.
        pub fn fromAxisAngle(rotation_axis: vec.Vector3(T), rotation_angle: T) Self {
            return .{ .quat = Q.fromAxisAngle(rotation_axis, rotation_angle) };
        }

        /// From a scaled axis (axis * angle). The zero vector maps to the identity.
        pub fn fromScaledAxis(v: vec.Vector3(T)) Self {
            const a = v.length();
            if (a < 1e-10) return identity();
            return fromAxisAngle(v.scale(1 / a), a);
        }

        /// From Euler angles (pitch, yaw, roll in radians).
        pub fn fromEuler(pitch: T, yaw: T, roll: T) Self {
            return .{ .quat = Q.fromEuler(pitch, yaw, roll).normalize() };
        }

        /// The rotation that maps direction `from` onto direction `to`.
        pub fn rotationBetween(from: vec.Vector3(T), to: vec.Vector3(T)) Self {
            return .{ .quat = Q.fromToRotation(from, to).normalize() };
        }

        /// From a rotation matrix.
        pub fn fromRotationMatrix(m: mat.Matrix(T, 4, 4)) Self {
            return .{ .quat = Q.fromMatrix(m).normalize() };
        }

        /// The underlying quaternion.
        pub fn inner(self: Self) Q {
            return self.quat;
        }

        /// The rotation angle in radians.
        pub fn angle(self: Self) T {
            return self.quat.getAngle();
        }

        /// The rotation axis.
        pub fn axis(self: Self) vec.Vector3(T) {
            return self.quat.getAxis();
        }

        /// The rotation axis scaled by the rotation angle.
        pub fn scaledAxis(self: Self) vec.Vector3(T) {
            return self.axis().scale(self.angle());
        }

        /// Compose two rotations.
        pub fn mul(self: Self, other: Self) Self {
            return .{ .quat = self.quat.mul(other.quat) };
        }

        /// The inverse rotation. For unit quaternions this is the conjugate.
        pub fn inverse(self: Self) Self {
            return .{ .quat = self.quat.conjugate() };
        }

        /// Rotate a 3D vector.
        pub fn rotateVector(self: Self, v: vec.Vector3(T)) vec.Vector3(T) {
            return self.quat.rotateVector(v);
        }

        /// Rotate a 3D vector by the inverse of this rotation.
        pub fn inverseRotateVector(self: Self, v: vec.Vector3(T)) vec.Vector3(T) {
            return self.quat.conjugate().rotateVector(v);
        }

        /// Spherical linear interpolation.
        pub fn slerp(self: Self, other: Self, t: T) Self {
            return .{ .quat = self.quat.slerp(other.quat, t) };
        }

        /// Normalized linear interpolation (fast approximation of slerp).
        pub fn nlerp(self: Self, other: Self, t: T) Self {
            return .{ .quat = self.quat.nlerp(other.quat, t) };
        }

        /// Raise to a power (a fraction of the rotation).
        pub fn powSelf(self: Self, t: T) Self {
            return fromScaledAxis(self.scaledAxis().scale(t));
        }

        /// Re-normalize to correct accumulated floating point drift.
        pub fn renormalize(self: Self) Self {
            return .{ .quat = self.quat.normalize() };
        }

        /// Convert to a 4x4 homogeneous rotation matrix.
        pub fn toHomogeneous(self: Self) mat.Matrix(T, 4, 4) {
            return self.quat.toMatrix();
        }

        /// Convert to a 4x4 rotation matrix.
        pub fn toRotationMatrix(self: Self) mat.Matrix(T, 4, 4) {
            return self.quat.toMatrix();
        }

        /// Convert to Euler angles (pitch, yaw, roll in radians).
        pub fn toEuler(self: Self) struct { pitch: T, yaw: T, roll: T } {
            const e = self.quat.toEuler();
            return .{ .pitch = e.pitch, .yaw = e.yaw, .roll = e.roll };
        }

        /// Check approximate equality, treating q and -q as the same rotation.
        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return self.quat.approxEqual(other.quat, epsilon) or
                self.quat.approxEqual(other.quat.negate(), epsilon);
        }
    };
}

/// An `f32` unit quaternion.
pub const UnitQuat = UnitQuaternion(f32);
/// An `f64` unit quaternion.
pub const UnitQuatd = UnitQuaternion(f64);

const testing = std.testing;
const V3 = vec.Vector3(f32);

test "UnitQuaternion rotates a vector" {
    const r = UnitQuat.fromAxisAngle(V3.unit_z, std.math.pi / 2.0);
    const v = r.rotateVector(V3.unit_x);
    try testing.expect(v.approxEql(V3.unit_y, 1e-6));
}

test "UnitQuaternion inverse round-trips" {
    const r = UnitQuat.fromAxisAngle(V3.init(1, 2, 3), 0.9);
    const v = V3.init(4, -5, 6);
    try testing.expect(r.inverse().rotateVector(r.rotateVector(v)).approxEql(v, 1e-4));
}

test "UnitQuaternion fromQuaternion normalizes" {
    const q = Quat.init(0, 0, 2, 0);
    const r = UnitQuat.fromQuaternion(q);
    try testing.expect(r.inner().isNormalized(1e-6));
}

test "UnitQuaternion composition matches sequential rotation" {
    const a = UnitQuat.fromAxisAngle(V3.unit_x, 0.4);
    const b = UnitQuat.fromAxisAngle(V3.unit_y, 0.7);
    const v = V3.init(1, 2, 3);
    const composed = a.mul(b).rotateVector(v);
    const sequential = a.rotateVector(b.rotateVector(v));
    try testing.expect(composed.approxEql(sequential, 1e-5));
}

test "UnitQuaternion homogeneous matrix agrees with rotateVector" {
    const r = UnitQuat.fromAxisAngle(V3.init(1, 1, 0), 1.1);
    const v = V3.init(2, -1, 0.5);
    const via_matrix = mat.transformDirection4x4(f32, r.toHomogeneous(), v);
    try testing.expect(via_matrix.approxEql(r.rotateVector(v), 1e-5));
}

test "UnitQuaternion slerp endpoints and axis" {
    const a = UnitQuat.identity();
    const b = UnitQuat.fromAxisAngle(V3.unit_y, 1.0);
    try testing.expect(a.slerp(b, 0).approxEql(a, 1e-6));
    try testing.expect(a.slerp(b, 1).approxEql(b, 1e-6));
    const mid = a.slerp(b, 0.5);
    try testing.expectApproxEqAbs(0.5, mid.angle(), 1e-5);
}

test "UnitQuaternion scaled axis round-trips" {
    const r = UnitQuat.fromAxisAngle(V3.unit_z, 0.8);
    const back = UnitQuat.fromScaledAxis(r.scaledAxis());
    try testing.expect(back.approxEql(r, 1e-5));
}
