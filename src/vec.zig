const std = @import("std");

pub fn Vector2(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,

        const Self = @This();
        pub const Vec = @Vector(2, T);

        pub const zero = Self{ .x = 0, .y = 0 };
        pub const one = Self{ .x = 1, .y = 1 };
        pub const unit_x = Self{ .x = 1, .y = 0 };
        pub const unit_y = Self{ .x = 0, .y = 1 };

        pub fn init(x: T, y: T) Self {
            return .{ .x = x, .y = y };
        }

        pub fn splat(value: T) Self {
            return .{ .x = value, .y = value };
        }

        pub fn fromArray(arr: [2]T) Self {
            return .{ .x = arr[0], .y = arr[1] };
        }

        pub fn toArray(self: Self) [2]T {
            return .{ self.x, self.y };
        }

        fn toVec(self: Self) Vec {
            return .{ self.x, self.y };
        }

        fn fromVec(v: Vec) Self {
            return .{ .x = v[0], .y = v[1] };
        }

        pub fn add(self: Self, other: Self) Self {
            return fromVec(self.toVec() + other.toVec());
        }

        pub fn sub(self: Self, other: Self) Self {
            return fromVec(self.toVec() - other.toVec());
        }

        pub fn mul(self: Self, other: Self) Self {
            return fromVec(self.toVec() * other.toVec());
        }

        pub fn div(self: Self, other: Self) Self {
            return fromVec(self.toVec() / other.toVec());
        }

        pub fn scale(self: Self, scalar: T) Self {
            return fromVec(self.toVec() * @as(Vec, @splat(scalar)));
        }

        pub fn negate(self: Self) Self {
            return fromVec(-self.toVec());
        }

        pub fn dot(self: Self, other: Self) T {
            return @reduce(.Add, self.toVec() * other.toVec());
        }

        pub fn lengthSquared(self: Self) T {
            return self.dot(self);
        }

        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) return zero;
            return self.scale(1.0 / len);
        }

        pub fn distance(self: Self, other: Self) T {
            return self.sub(other).length();
        }

        pub fn distanceSquared(self: Self, other: Self) T {
            return self.sub(other).lengthSquared();
        }

        pub fn lerp(self: Self, other: Self, t: T) Self {
            return self.add(other.sub(self).scale(t));
        }

        pub fn min(self: Self, other: Self) Self {
            return fromVec(@min(self.toVec(), other.toVec()));
        }

        pub fn max(self: Self, other: Self) Self {
            return fromVec(@max(self.toVec(), other.toVec()));
        }

        pub fn clamp(self: Self, min_val: Self, max_val: Self) Self {
            return self.max(min_val).min(max_val);
        }

        pub fn abs(self: Self) Self {
            return fromVec(@abs(self.toVec()));
        }

        pub fn floor(self: Self) Self {
            return fromVec(@floor(self.toVec()));
        }

        pub fn ceil(self: Self) Self {
            return fromVec(@ceil(self.toVec()));
        }

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

        pub fn eql(self: Self, other: Self) bool {
            return self.x == other.x and self.y == other.y;
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.x - other.x) <= epsilon and @abs(self.y - other.y) <= epsilon;
        }
    };
}

pub fn Vector3(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,
        z: T,

        const Self = @This();
        pub const Vec = @Vector(4, T); // Use 4 for alignment, ignore w

        pub const zero = Self{ .x = 0, .y = 0, .z = 0 };
        pub const one = Self{ .x = 1, .y = 1, .z = 1 };
        pub const unit_x = Self{ .x = 1, .y = 0, .z = 0 };
        pub const unit_y = Self{ .x = 0, .y = 1, .z = 0 };
        pub const unit_z = Self{ .x = 0, .y = 0, .z = 1 };
        pub const up = unit_y;
        pub const down = Self{ .x = 0, .y = -1, .z = 0 };
        pub const forward = Self{ .x = 0, .y = 0, .z = -1 };
        pub const back = unit_z;
        pub const left = Self{ .x = -1, .y = 0, .z = 0 };
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

        pub fn add(self: Self, other: Self) Self {
            return fromVec(self.toVec() + other.toVec());
        }

        pub fn sub(self: Self, other: Self) Self {
            return fromVec(self.toVec() - other.toVec());
        }

        pub fn mul(self: Self, other: Self) Self {
            return fromVec(self.toVec() * other.toVec());
        }

        pub fn div(self: Self, other: Self) Self {
            return fromVec(self.toVec() / other.toVec());
        }

        pub fn scale(self: Self, scalar: T) Self {
            return fromVec(self.toVec() * @as(Vec, @splat(scalar)));
        }

        pub fn negate(self: Self) Self {
            return fromVec(-self.toVec());
        }

        pub fn dot(self: Self, other: Self) T {
            const v = self.toVec() * other.toVec();
            return v[0] + v[1] + v[2];
        }

        pub fn cross(self: Self, other: Self) Self {
            return .{
                .x = self.y * other.z - self.z * other.y,
                .y = self.z * other.x - self.x * other.z,
                .z = self.x * other.y - self.y * other.x,
            };
        }

        pub fn lengthSquared(self: Self) T {
            return self.dot(self);
        }

        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) return zero;
            return self.scale(1.0 / len);
        }

        pub fn distance(self: Self, other: Self) T {
            return self.sub(other).length();
        }

        pub fn distanceSquared(self: Self, other: Self) T {
            return self.sub(other).lengthSquared();
        }

        pub fn lerp(self: Self, other: Self, t: T) Self {
            return self.add(other.sub(self).scale(t));
        }

        pub fn min(self: Self, other: Self) Self {
            return fromVec(@min(self.toVec(), other.toVec()));
        }

        pub fn max(self: Self, other: Self) Self {
            return fromVec(@max(self.toVec(), other.toVec()));
        }

        pub fn clamp(self: Self, min_val: Self, max_val: Self) Self {
            return self.max(min_val).min(max_val);
        }

        pub fn abs(self: Self) Self {
            return fromVec(@abs(self.toVec()));
        }

        pub fn floor(self: Self) Self {
            return fromVec(@floor(self.toVec()));
        }

        pub fn ceil(self: Self) Self {
            return fromVec(@ceil(self.toVec()));
        }

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

        pub fn eql(self: Self, other: Self) bool {
            return self.x == other.x and self.y == other.y and self.z == other.z;
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.x - other.x) <= epsilon and
                @abs(self.y - other.y) <= epsilon and
                @abs(self.z - other.z) <= epsilon;
        }

        pub fn xy(self: Self) Vector2(T) {
            return .{ .x = self.x, .y = self.y };
        }
    };
}

pub fn Vector4(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,
        z: T,
        w: T,

        const Self = @This();
        pub const Vec = @Vector(4, T);

        pub const zero = Self{ .x = 0, .y = 0, .z = 0, .w = 0 };
        pub const one = Self{ .x = 1, .y = 1, .z = 1, .w = 1 };

        pub fn init(x: T, y: T, z: T, w: T) Self {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }

        pub fn splat(value: T) Self {
            return .{ .x = value, .y = value, .z = value, .w = value };
        }

        pub fn fromVec3(v: Vector3(T), w: T) Self {
            return .{ .x = v.x, .y = v.y, .z = v.z, .w = w };
        }

        pub fn fromArray(arr: [4]T) Self {
            return .{ .x = arr[0], .y = arr[1], .z = arr[2], .w = arr[3] };
        }

        pub fn toArray(self: Self) [4]T {
            return .{ self.x, self.y, self.z, self.w };
        }

        fn toVec(self: Self) Vec {
            return .{ self.x, self.y, self.z, self.w };
        }

        fn fromVec(v: Vec) Self {
            return .{ .x = v[0], .y = v[1], .z = v[2], .w = v[3] };
        }

        pub fn add(self: Self, other: Self) Self {
            return fromVec(self.toVec() + other.toVec());
        }

        pub fn sub(self: Self, other: Self) Self {
            return fromVec(self.toVec() - other.toVec());
        }

        pub fn mul(self: Self, other: Self) Self {
            return fromVec(self.toVec() * other.toVec());
        }

        pub fn div(self: Self, other: Self) Self {
            return fromVec(self.toVec() / other.toVec());
        }

        pub fn scale(self: Self, scalar: T) Self {
            return fromVec(self.toVec() * @as(Vec, @splat(scalar)));
        }

        pub fn negate(self: Self) Self {
            return fromVec(-self.toVec());
        }

        pub fn dot(self: Self, other: Self) T {
            return @reduce(.Add, self.toVec() * other.toVec());
        }

        pub fn lengthSquared(self: Self) T {
            return self.dot(self);
        }

        pub fn length(self: Self) T {
            return @sqrt(self.lengthSquared());
        }

        pub fn normalize(self: Self) Self {
            const len = self.length();
            if (len == 0) return zero;
            return self.scale(1.0 / len);
        }

        pub fn lerp(self: Self, other: Self, t: T) Self {
            return self.add(other.sub(self).scale(t));
        }

        pub fn min(self: Self, other: Self) Self {
            return fromVec(@min(self.toVec(), other.toVec()));
        }

        pub fn max(self: Self, other: Self) Self {
            return fromVec(@max(self.toVec(), other.toVec()));
        }

        pub fn clamp(self: Self, min_val: Self, max_val: Self) Self {
            return self.max(min_val).min(max_val);
        }

        pub fn abs(self: Self) Self {
            return fromVec(@abs(self.toVec()));
        }

        pub fn eql(self: Self, other: Self) bool {
            return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w;
        }

        pub fn approxEql(self: Self, other: Self, epsilon: T) bool {
            return @abs(self.x - other.x) <= epsilon and
                @abs(self.y - other.y) <= epsilon and
                @abs(self.z - other.z) <= epsilon and
                @abs(self.w - other.w) <= epsilon;
        }

        pub fn xyz(self: Self) Vector3(T) {
            return .{ .x = self.x, .y = self.y, .z = self.z };
        }

        pub fn xy(self: Self) Vector2(T) {
            return .{ .x = self.x, .y = self.y };
        }
    };
}
