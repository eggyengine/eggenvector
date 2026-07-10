const std = @import("std");
const vec = @import("vec.zig");

/// A wrapper for algebraic entities with a norm equal to one.
///
/// Works with any type providing `length()` and `scale()` methods,
/// e.g. `Unit(Vector3(f32))` or `Unit(Quaternion(f32))`.
pub fn Unit(comptime T: type) type {
    return struct {
        value: T,

        const Self = @This();
        pub const Scalar = @TypeOf(@as(T, undefined).length());

        /// Normalize `value` and wrap it. Asserts that the norm is non-zero.
        pub fn new(value: T) Self {
            const len = value.length();
            std.debug.assert(len > 0);
            return .{ .value = value.scale(1 / len) };
        }

        /// Normalize `value`, returning both the unit wrapper and the original norm.
        pub fn newAndGet(value: T) struct { unit: Self, norm: Scalar } {
            const len = value.length();
            std.debug.assert(len > 0);
            return .{ .unit = .{ .value = value.scale(1 / len) }, .norm = len };
        }

        /// Attempt to normalize; returns null if the norm is <= `min_norm`.
        pub fn tryNew(value: T, min_norm: Scalar) ?Self {
            const len = value.length();
            if (len <= min_norm) return null;
            return .{ .value = value.scale(1 / len) };
        }

        /// Wrap without normalizing. The caller guarantees the unit-norm invariant.
        pub fn newUnchecked(value: T) Self {
            return .{ .value = value };
        }

        /// The wrapped value.
        pub fn get(self: Self) T {
            return self.value;
        }

        /// Re-normalize to correct accumulated floating point drift.
        pub fn renormalize(self: Self) Self {
            return new(self.value);
        }
    };
}

test "Unit normalizes a vector" {
    const V3 = vec.Vector3(f32);
    const u = Unit(V3).new(.{ .x = 3, .y = 0, .z = 4 });
    try std.testing.expectApproxEqAbs(1.0, u.get().length(), 1e-6);
    try std.testing.expectApproxEqAbs(0.6, u.get().x, 1e-6);
    try std.testing.expectApproxEqAbs(0.8, u.get().z, 1e-6);
}

test "Unit newAndGet returns the original norm" {
    const V2 = vec.Vector2(f32);
    const r = Unit(V2).newAndGet(.{ .x = 0, .y = 5 });
    try std.testing.expectApproxEqAbs(5.0, r.norm, 1e-6);
    try std.testing.expectApproxEqAbs(1.0, r.unit.get().y, 1e-6);
}

test "Unit tryNew rejects near-zero values" {
    const V3 = vec.Vector3(f32);
    try std.testing.expect(Unit(V3).tryNew(V3.zero, 1e-6) == null);
    try std.testing.expect(Unit(V3).tryNew(V3.unit_x, 1e-6) != null);
}
