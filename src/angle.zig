const std = @import("std");

/// A type-safe description of an angle.
///
/// Internally stores the angle in radians, and dynamically converts on export.
pub fn Angle(comptime T: type) type {
    comptime std.debug.assert(@typeInfo(T) == .float); // only floats make sense

    return struct {
        const Self = @This();

        _rads: T,

        pub fn fromDegrees(degrees: T) Self {
            return Self{ ._rads = std.math.degreesToRadians(degrees) };
        }

        pub fn fromRadians(radians: T) Self {
            return Self{ ._rads = radians };
        }

        pub fn toRadians(self: Self) T {
            return self._rads;
        }

        pub fn toDegrees(self: Self) T {
            return std.math.radiansToDegrees(self._rads);
        }

        pub fn add(self: Self, other: Self) Self {
            return Self{ ._rads = self._rads + other._rads };
        }

        pub fn sub(self: Self, other: Self) Self {
            return Self{ ._rads = self._rads - other._rads };
        }

        pub fn scale(self: Self, factor: T) Self {
            return Self{ ._rads = self._rads * factor };
        }

        /// Normalize to [0, 2π)
        pub fn normalize(self: Self) Self {
            const tau = std.math.tau;
            return Self{ ._rads = @mod(self._rads, tau) };
        }

        /// Normalize to (-π, π]
        pub fn normalizeSigned(self: Self) Self {
            const tau = std.math.tau;
            return Self{ ._rads = self._rads - tau * @round(self._rads / tau) };
        }

        pub fn sin(self: Self) T {
            return std.math.sin(self._rads);
        }

        pub fn cos(self: Self) T {
            return std.math.cos(self._rads);
        }

        pub fn tan(self: Self) T {
            return std.math.tan(self._rads);
        }
    };
}
