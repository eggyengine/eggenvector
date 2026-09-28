const std = @import("std");

/// A 2D vector with SIMD-backed arithmetic. `extern` layout, safe to upload to the GPU.
pub fn Vector2(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,

        const Self = @This();
        /// The underlying SIMD `@Vector` type used for arithmetic.
        pub const Vec = @Vector(2, T);

        /// All components zero.
        pub const zero = Self{ .x = 0, .y = 0 };
        /// All components one.
        pub const one = Self{ .x = 1, .y = 1 };
        /// Unit vector along +X.
        pub const unit_x = Self{ .x = 1, .y = 0 };
        /// Unit vector along +Y.
        pub const unit_y = Self{ .x = 0, .y = 1 };

        /// Create a new value from its components.
        pub fn init(x: T, y: T) Self {
            return .{ .x = x, .y = y };
        }

        /// Create a vector with every component set to `value`.
        pub fn splat(value: T) Self {
            return .{ .x = value, .y = value };
        }

        /// Create a vector from an array of components.
        pub fn fromArray(arr: [2]T) Self {
            return .{ .x = arr[0], .y = arr[1] };
        }

        /// Convert to an array of components.
        pub fn toArray(self: Self) [2]T {
            return .{ self.x, self.y };
        }

        fn toVec(self: Self) Vec {
            return .{ self.x, self.y };
        }

        fn fromVec(v: Vec) Self {
            return .{ .x = v[0], .y = v[1] };
        }

        /// Component-wise addition (SIMD).
        pub fn add(self: Self, other: Self) Self {
            return fromVec(self.toVec() + other.toVec());
        }

        /// Component-wise subtraction (SIMD).
        pub fn sub(self: Self, other: Self) Self {
            return fromVec(self.toVec() - other.toVec());
        }

        /// Component-wise multiplication (SIMD).
        pub fn mul(self: Self, other: Self) Self {
            return fromVec(self.toVec() * other.toVec());
        }

        /// Component-wise division (SIMD).
        pub fn div(self: Self, other: Self) Self {
            return fromVec(self.toVec() / other.toVec());
        }

        /// Multiply every component by `scalar` (SIMD).
        pub fn scale(self: Self, scalar: T) Self {
            return fromVec(self.toVec() * @as(Vec, @splat(scalar)));
        }

        /// Negate every component (SIMD).
        pub fn negate(self: Self) Self {
            return fromVec(-self.toVec());
        }

        /// Dot product (SIMD multiply + reduce).
        pub fn dot(self: Self, other: Self) T {
            return @reduce(.Add, self.toVec() * other.toVec());
        }

        /// Squared length. Cheaper than `length` when only comparing magnitudes.
        pub fn lengthSquared(self: Self) T {
            return self.dot(self);
        }

        /// Euclidean length (magnitude).
        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        /// Unit-length copy of this vector. Returns `zero` for the zero vector.
        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) return zero;
            return self.scale(1.0 / len);
        }

        /// Euclidean distance to `other`.
        pub fn distance(self: Self, other: Self) T {
            return self.sub(other).length();
        }

        /// Squared Euclidean distance to `other`.
        pub fn distanceSquared(self: Self, other: Self) T {
            return self.sub(other).lengthSquared();
        }

        /// Linear interpolation towards `other` by `t` (0 = self, 1 = other).
        pub fn lerp(self: Self, other: Self, t: T) Self {
            return self.add(other.sub(self).scale(t));
        }

        /// Component-wise minimum (SIMD).
        pub fn min(self: Self, other: Self) Self {
            return fromVec(@min(self.toVec(), other.toVec()));
        }

        /// Component-wise maximum (SIMD).
        pub fn max(self: Self, other: Self) Self {
            return fromVec(@max(self.toVec(), other.toVec()));
        }

        /// Component-wise clamp between `min_val` and `max_val` (SIMD).
        pub fn clamp(self: Self, min_val: Self, max_val: Self) Self {
            return self.max(min_val).min(max_val);
        }

        /// Component-wise absolute value (SIMD).
        pub fn abs(self: Self) Self {
            return fromVec(@abs(self.toVec()));
        }

        /// Component-wise floor (SIMD).
        pub fn floor(self: Self) Self {
            return fromVec(@floor(self.toVec()));
        }

        /// Component-wise ceiling (SIMD).
        pub fn ceil(self: Self) Self {
            return fromVec(@ceil(self.toVec()));
        }

        /// Component-wise round to nearest (SIMD).
        pub fn round(self: Self) Self {
            return fromVec(@round(self.toVec()));
        }

        /// Perpendicular vector (rotated 90 degrees counter-clockwise)
        pub fn perpendicular(self: Self) Self {
            return .{ .x = -self.y, .y = self.x };
        }

        /// Angle in radians from positive x-axis
        pub fn angle(self: Self) T {
            return std.math.atan2(self.y, self.x);
        }

        /// Create from angle (in radians) and length
        pub fn fromAngle(a: T, len: T) Self {
            return .{ .x = @cos(a) * len, .y = @sin(a) * len };
        }

        /// Exact component-wise equality.
        pub fn eql(self: Self, other: Self) bool {
            return self.x == other.x and self.y == other.y;
        }

        /// Approximate equality: true when every component differs by at most `epsilon`.
        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.x - other.x) <= epsilon and @abs(self.y - other.y) <= epsilon;
        }
    };
}

/// A 3D vector with SIMD-backed arithmetic (computed in a 4-lane `@Vector`, stored as 3 fields). `extern` layout.
pub fn Vector3(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,
        z: T,

        const Self = @This();
        /// The underlying SIMD `@Vector` type used for arithmetic.
        pub const Vec = @Vector(4, T); // Use 4 for alignment, ignore w

        /// All components zero.
        pub const zero = Self{ .x = 0, .y = 0, .z = 0 };
        /// All components one.
        pub const one = Self{ .x = 1, .y = 1, .z = 1 };
        /// Unit vector along +X.
        pub const unit_x = Self{ .x = 1, .y = 0, .z = 0 };
        /// Unit vector along +Y.
        pub const unit_y = Self{ .x = 0, .y = 1, .z = 0 };
        /// Unit vector along +Z.
        pub const unit_z = Self{ .x = 0, .y = 0, .z = 1 };
        /// +Y.
        pub const up = unit_y;
        /// -Y.
        pub const down = Self{ .x = 0, .y = -1, .z = 0 };
        /// -Z (right-handed forward).
        pub const forward = Self{ .x = 0, .y = 0, .z = -1 };
        /// +Z.
        pub const back = unit_z;
        /// -X.
        pub const left = Self{ .x = -1, .y = 0, .z = 0 };
        /// +X.
        pub const right = unit_x;

        /// Create a new vector by defining each value
        pub fn init(x: T, y: T, z: T) Self {
            return .{ .x = x, .y = y, .z = z };
        }

        /// Create a new vector with all values set to T
        pub fn splat(value: T) Self {
            return .{ .x = value, .y = value, .z = value };
        }

        /// Create a new vector from an 3 element array.
        pub fn fromArray(arr: [3]T) Self {
            return .{ .x = arr[0], .y = arr[1], .z = arr[2] };
        }

        /// Convert to an array of components.
        pub fn toArray(self: Self) [3]T {
            return .{ self.x, self.y, self.z };
        }

        /// Convert to a SIMD zig-native `@Vector` type
        fn toVec(self: Self) Vec {
            return .{ self.x, self.y, self.z, 0 };
        }

        /// Create a new eggy Vector from `@Vector` type.
        fn fromVec(v: Vec) Self {
            return .{ .x = v[0], .y = v[1], .z = v[2] };
        }

        /// Component-wise addition (SIMD).
        pub fn add(self: Self, other: Self) Self {
            return fromVec(self.toVec() + other.toVec());
        }

        /// Component-wise subtraction (SIMD).
        pub fn sub(self: Self, other: Self) Self {
            return fromVec(self.toVec() - other.toVec());
        }

        /// Component-wise multiplication (SIMD).
        pub fn mul(self: Self, other: Self) Self {
            return fromVec(self.toVec() * other.toVec());
        }

        /// Component-wise division (SIMD).
        pub fn div(self: Self, other: Self) Self {
            // padding lane divides by 1, not 0 (0/0 panics for integer types)
            return fromVec(self.toVec() / Vec{ other.x, other.y, other.z, 1 });
        }

        /// Multiply every component by `scalar` (SIMD).
        pub fn scale(self: Self, scalar: T) Self {
            return fromVec(self.toVec() * @as(Vec, @splat(scalar)));
        }

        /// Negate every component (SIMD).
        pub fn negate(self: Self) Self {
            return fromVec(-self.toVec());
        }

        /// Dot product (SIMD multiply + reduce).
        pub fn dot(self: Self, other: Self) T {
            return @reduce(.Add, self.toVec() * other.toVec());
        }

        /// Cross product.
        pub fn cross(self: Self, other: Self) Self {
            return .{
                .x = self.y * other.z - self.z * other.y,
                .y = self.z * other.x - self.x * other.z,
                .z = self.x * other.y - self.y * other.x,
            };
        }

        /// Squared length. Cheaper than `length` when only comparing magnitudes.
        pub fn lengthSquared(self: Self) T {
            return self.dot(self);
        }

        /// Euclidean length (magnitude).
        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        /// Unit-length copy of this vector. Returns `zero` for the zero vector.
        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) return zero;
            return self.scale(1.0 / len);
        }

        /// Euclidean distance to `other`.
        pub fn distance(self: Self, other: Self) T {
            return self.sub(other).length();
        }

        /// Squared Euclidean distance to `other`.
        pub fn distanceSquared(self: Self, other: Self) T {
            return self.sub(other).lengthSquared();
        }

        /// Linear interpolation towards `other` by `t` (0 = self, 1 = other).
        pub fn lerp(self: Self, other: Self, t: T) Self {
            return self.add(other.sub(self).scale(t));
        }

        /// Component-wise minimum (SIMD).
        pub fn min(self: Self, other: Self) Self {
            return fromVec(@min(self.toVec(), other.toVec()));
        }

        /// Component-wise maximum (SIMD).
        pub fn max(self: Self, other: Self) Self {
            return fromVec(@max(self.toVec(), other.toVec()));
        }

        /// Component-wise clamp between `min_val` and `max_val` (SIMD).
        pub fn clamp(self: Self, min_val: Self, max_val: Self) Self {
            return self.max(min_val).min(max_val);
        }

        /// Component-wise absolute value (SIMD).
        pub fn abs(self: Self) Self {
            return fromVec(@abs(self.toVec()));
        }

        /// Component-wise floor (SIMD).
        pub fn floor(self: Self) Self {
            return fromVec(@floor(self.toVec()));
        }

        /// Component-wise ceiling (SIMD).
        pub fn ceil(self: Self) Self {
            return fromVec(@ceil(self.toVec()));
        }

        /// Component-wise round to nearest (SIMD).
        pub fn round(self: Self) Self {
            return fromVec(@round(self.toVec()));
        }

        /// Reflect vector off a surface with given normal
        pub fn reflect(self: Self, normal: Self) Self {
            return self.sub(normal.scale(2.0 * self.dot(normal)));
        }

        /// Project self onto other
        pub fn project(self: Self, onto: Self) Self {
            return onto.scale(self.dot(onto) / onto.dot(onto));
        }

        /// Exact component-wise equality.
        pub fn eql(self: Self, other: Self) bool {
            return self.x == other.x and self.y == other.y and self.z == other.z;
        }

        /// Approximate equality: true when every component differs by at most `epsilon`.
        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.x - other.x) <= epsilon and
                @abs(self.y - other.y) <= epsilon and
                @abs(self.z - other.z) <= epsilon;
        }

        /// The x and y components as a 2D vector.
        pub fn xy(self: Self) Vector2(T) {
            return .{ .x = self.x, .y = self.y };
        }
    };
}

/// A 4D vector with SIMD-backed arithmetic. `extern` layout.
pub fn Vector4(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,
        z: T,
        w: T,

        const Self = @This();
        /// The underlying SIMD `@Vector` type used for arithmetic.
        pub const Vec = @Vector(4, T);

        /// All components zero.
        pub const zero = Self{ .x = 0, .y = 0, .z = 0, .w = 0 };
        /// All components one.
        pub const one = Self{ .x = 1, .y = 1, .z = 1, .w = 1 };

        /// Create a new value from its components.
        pub fn init(x: T, y: T, z: T, w: T) Self {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }

        /// Create a vector with every component set to `value`.
        pub fn splat(value: T) Self {
            return .{ .x = value, .y = value, .z = value, .w = value };
        }

        /// Create from a 3D vector and a `w` component.
        pub fn fromVec3(v: Vector3(T), w: T) Self {
            return .{ .x = v.x, .y = v.y, .z = v.z, .w = w };
        }

        /// Create a vector from an array of components.
        pub fn fromArray(arr: [4]T) Self {
            return .{ .x = arr[0], .y = arr[1], .z = arr[2], .w = arr[3] };
        }

        /// Convert to an array of components.
        pub fn toArray(self: Self) [4]T {
            return .{ self.x, self.y, self.z, self.w };
        }

        fn toVec(self: Self) Vec {
            return .{ self.x, self.y, self.z, self.w };
        }

        fn fromVec(v: Vec) Self {
            return .{ .x = v[0], .y = v[1], .z = v[2], .w = v[3] };
        }

        /// Component-wise addition (SIMD).
        pub fn add(self: Self, other: Self) Self {
            return fromVec(self.toVec() + other.toVec());
        }

        /// Component-wise subtraction (SIMD).
        pub fn sub(self: Self, other: Self) Self {
            return fromVec(self.toVec() - other.toVec());
        }

        /// Component-wise multiplication (SIMD).
        pub fn mul(self: Self, other: Self) Self {
            return fromVec(self.toVec() * other.toVec());
        }

        /// Component-wise division (SIMD).
        pub fn div(self: Self, other: Self) Self {
            return fromVec(self.toVec() / other.toVec());
        }

        /// Multiply every component by `scalar` (SIMD).
        pub fn scale(self: Self, scalar: T) Self {
            return fromVec(self.toVec() * @as(Vec, @splat(scalar)));
        }

        /// Negate every component (SIMD).
        pub fn negate(self: Self) Self {
            return fromVec(-self.toVec());
        }

        /// Dot product (SIMD multiply + reduce).
        pub fn dot(self: Self, other: Self) T {
            return @reduce(.Add, self.toVec() * other.toVec());
        }

        /// Squared length. Cheaper than `length` when only comparing magnitudes.
        pub fn lengthSquared(self: Self) T {
            return self.dot(self);
        }

        /// Euclidean length (magnitude).
        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        /// Unit-length copy of this vector. Returns `zero` for the zero vector.
        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) return zero;
            return self.scale(1.0 / len);
        }

        /// Linear interpolation towards `other` by `t` (0 = self, 1 = other).
        pub fn lerp(self: Self, other: Self, t: T) Self {
            return self.add(other.sub(self).scale(t));
        }

        /// Component-wise minimum (SIMD).
        pub fn min(self: Self, other: Self) Self {
            return fromVec(@min(self.toVec(), other.toVec()));
        }

        /// Component-wise maximum (SIMD).
        pub fn max(self: Self, other: Self) Self {
            return fromVec(@max(self.toVec(), other.toVec()));
        }

        /// Component-wise clamp between `min_val` and `max_val` (SIMD).
        pub fn clamp(self: Self, min_val: Self, max_val: Self) Self {
            return self.max(min_val).min(max_val);
        }

        /// Component-wise absolute value (SIMD).
        pub fn abs(self: Self) Self {
            return fromVec(@abs(self.toVec()));
        }

        /// Exact component-wise equality.
        pub fn eql(self: Self, other: Self) bool {
            return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w;
        }

        /// Approximate equality: true when every component differs by at most `epsilon`.
        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.x - other.x) <= epsilon and
                @abs(self.y - other.y) <= epsilon and
                @abs(self.z - other.z) <= epsilon and
                @abs(self.w - other.w) <= epsilon;
        }

        /// The x, y and z components as a 3D vector.
        pub fn xyz(self: Self) Vector3(T) {
            return .{ .x = self.x, .y = self.y, .z = self.z };
        }

        /// The x and y components as a 2D vector.
        pub fn xy(self: Self) Vector2(T) {
            return .{ .x = self.x, .y = self.y };
        }
    };
}
