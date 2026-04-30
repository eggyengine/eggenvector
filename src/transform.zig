const math = @import("../math.zig");

/// A type that allows for position, rotation and scaling of an object.
pub const Transform = struct {
    position: math.Vec3,
    rotation: math.Quat,
    scale: math.Vec3,

    pub const identity = Transform{
        .position = math.Vec3.splat(0),
        .rotation = math.Quat.identity(),
        .scale = math.Vec3.splat(1),
    };

    /// Compose the TRS (Translation * Rotation * Scale) model matrix, typically used for the camera.
    pub fn toMat4(self: Transform) math.Mat4 {
        const s = math.scaling4x4(f32, self.scale.x, self.scale.y, self.scale.z);
        const r = self.rotation.toMatrix();
        const t = math.translationVec4x4(f32, self.position);
        return math.multiply4x4(f32, t, math.multiply4x4(f32, r, s));
    }

    /// Translate by an offset.
    pub fn translate(self: Transform, offset: math.Vec3) Transform {
        return .{ .position = self.position.add(offset), .rotation = self.rotation, .scale = self.scale };
    }

    /// Rotate by a quaternion (applied after the current rotation).
    pub fn rotate(self: Transform, q: math.Quat) Transform {
        return .{ .position = self.position, .rotation = q.mul(self.rotation), .scale = self.scale };
    }

    /// Rotate by an axis-angle.
    pub fn rotateAxisAngle(self: Transform, axis: math.Vec3, angle: f32) Transform {
        return self.rotate(math.Quat.fromAxisAngle(axis, angle));
    }

    /// Uniform scale.
    pub fn scaleUniform(self: Transform, s: f32) Transform {
        return .{ .position = self.position, .rotation = self.rotation, .scale = self.scale.scale(s) };
    }

    /// Non-uniform scale.
    pub fn scaleBy(self: Transform, s: math.Vec3) Transform {
        return .{ .position = self.position, .rotation = self.rotation, .scale = self.scale.mul(s) };
    }

    /// The local forward direction (−Z rotated by the quaternion).
    pub fn forward(self: Transform) math.Vec3 {
        return self.rotation.rotateVector(math.Vec3.init(0, 0, -1));
    }

    /// The local right direction (+X rotated by the quaternion).
    pub fn right(self: Transform) math.Vec3 {
        return self.rotation.rotateVector(math.Vec3.init(1, 0, 0));
    }

    /// The local up direction (+Y rotated by the quaternion).
    pub fn up(self: Transform) math.Vec3 {
        return self.rotation.rotateVector(math.Vec3.init(0, 1, 0));
    }

    /// Linearly interpolate between two transforms.
    pub fn lerp(self: Transform, other: Transform, t: f32) Transform {
        return .{
            .position = self.position.lerp(other.position, t),
            .rotation = self.rotation.slerp(other.rotation, t),
            .scale = self.scale.lerp(other.scale, t),
        };
    }

    /// Compute the inverse transform.
    pub fn inverse(self: Transform) Transform {
        const inv_rot = self.rotation.inverse();
        const inv_scale = math.Vec3.init(1.0 / self.scale.x, 1.0 / self.scale.y, 1.0 / self.scale.z);
        const inv_pos = inv_rot.rotateVector(self.position.negate().mul(inv_scale));
        return .{ .position = inv_pos, .rotation = inv_rot, .scale = inv_scale };
    }

    /// Transform a point (applies scale, rotation, then translation).
    pub fn transformPoint(self: Transform, point: math.Vec3) math.Vec3 {
        const scaled = point.mul(self.scale);
        const rotated = self.rotation.rotateVector(scaled);
        return rotated.add(self.position);
    }

    /// Transform a direction (applies scale and rotation, no translation).
    pub fn transformDirection(self: Transform, dir: math.Vec3) math.Vec3 {
        return self.rotation.rotateVector(dir.mul(self.scale));
    }
};

/// A superset of Transform which supports local and world transforms.
pub const EntityTransform = struct {
    local: Transform,
    world: Transform,

    /// Compute world transform from a parent's world transform and this entity's local transform, then reset local back to its identity.
    pub fn updateWorld(self: *EntityTransform, parent_world: Transform) void {
        self.world = .{
            .position = parent_world.transformPoint(self.local.position),
            .rotation = parent_world.rotation.mul(self.local.rotation),
            .scale = parent_world.scale.mul(self.local.scale),
        };
        self.local = Transform.identity;
    }

    /// Create an EntityTransform with no parent (local == world).
    pub fn root(local: Transform) EntityTransform {
        return .{ .local = local, .world = local };
    }

    // todo: get access to the world and create a function that computes `complete transform`.
};
