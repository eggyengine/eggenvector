//! eggenvector - eggy math types
//!
//! includes:
//! - vector (2, 3, 4)
//! - matrix (2x2, 3x3, 4x4, axb)
//! - quaternion
//! - transform
//! - angles
//!
//! for primarily `f32` but with support for `f64`, and backed by SIMD `@Vector` types

const std = @import("std");

// --------------- vectors ---------------
const vec = @import("vec.zig");

pub const Vec2 = vec.Vector2(f32);
pub const Vec3 = vec.Vector3(f32);
pub const Vec4 = vec.Vector4(f32);

pub const Vec2d = vec.Vector2(f64);
pub const Vec3d = vec.Vector3(f64);
pub const Vec4d = vec.Vector4(f64);

pub const Vec2i = vec.Vector2(i32);
pub const Vec3i = vec.Vector3(i32);
pub const Vec4i = vec.Vector4(i32);

pub const Vec2u = vec.Vector2(u32);
pub const Vec3u = vec.Vector3(u32);
pub const Vec4u = vec.Vector4(u32);

// --------------- matrices ---------------
const mat = @import("mat.zig");

pub const Mat2 = mat.Mat2;
pub const Mat3 = mat.Mat3;
pub const Mat4 = mat.Mat4;

pub const Mat2d = mat.Mat2d;
pub const Mat3d = mat.Mat3d;
pub const Mat4d = mat.Mat4d;

pub const Mat2i = mat.Mat2i;
pub const Mat3i = mat.Mat3i;
pub const Mat4i = mat.Mat4i;

pub const Matrix = mat.Matrix;

// matrix operations
pub const identity = mat.identity;
pub const zero = mat.zero;
pub const splat = mat.splat;
pub const fromArray = mat.fromArray;
pub const add = mat.add;
pub const sub = mat.sub;
pub const scale = mat.scale;
pub const mul = mat.mul;
pub const transpose = mat.transpose;

pub const determinant2x2 = mat.determinant2x2;
pub const inverse2x2 = mat.inverse2x2;
pub const rotation2x2 = mat.rotation2x2;
pub const scaling2x2 = mat.scaling2x2;

pub const determinant3x3 = mat.determinant3x3;
pub const rotationX3x3 = mat.rotationX3x3;
pub const rotationY3x3 = mat.rotationY3x3;
pub const rotationZ3x3 = mat.rotationZ3x3;
pub const scaling3x3 = mat.scaling3x3;
pub const transformVec3by3x3 = mat.transformVec3by3x3;

pub const translation4x4 = mat.translation4x4;
pub const translationVec4x4 = mat.translationVec4x4;
pub const scaling4x4 = mat.scaling4x4;
pub const uniformScaling4x4 = mat.uniformScaling4x4;
pub const rotationX4x4 = mat.rotationX4x4;
pub const rotationY4x4 = mat.rotationY4x4;
pub const rotationZ4x4 = mat.rotationZ4x4;
pub const rotationAxis4x4 = mat.rotationAxis4x4;
pub const lookAt4x4 = mat.lookAt4x4;
pub const perspective4x4 = mat.perspective4x4;
pub const orthographic4x4 = mat.orthographic4x4;
pub const transformVec4by4x4 = mat.transformVec4by4x4;
pub const transformPoint4x4 = mat.transformPoint4x4;
pub const transformDirection4x4 = mat.transformDirection4x4;
pub const multiply4x4 = mat.multiply4x4;
pub const inverse4x4 = mat.inverse4x4;

// type validation
pub const isPermittedType = mat.isPermittedType;
pub const isFloatType = mat.isFloatType;

// --------------- quaternions ---------------
const quat = @import("quat.zig");

pub const Quat = quat.Quat;
/// A `f64` Quaternion.
pub const Quatd = quat.Quatd;
pub const Quaternion = quat.Quaternion;

// --------------- helpers ---------------

/// Returns an identity quaternion rotated at an angle on an axis
pub fn rotate(angle: Angle, axis: Vec3) Quat {
    return Quat.identity().rotate(angle, axis);
}

/// Create a look-at view matrix using f32.
pub fn lookAt(eye: Vec3, target: Vec3, up: Vec3) Mat4 {
    return lookAt4x4(f32, eye, target, up);
}

/// Create a perspective projection matrix using f32.
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) Mat4 {
    return perspective4x4(f32, fov_y, aspect, near, far);
}

// --------------- padding ---------------

/// Allows you to create a padding of `type` `count` amount of times. Typically used to ensure alignment in a struct.
///
/// Initialise with `Padding(...).default`
pub fn Padding(comptime T: type, comptime count: usize) type {
    return struct {
        data: [count]T = [_]T{0} ** count,

        /// The default initialiser.
        ///
        /// Since this padding likely won't be used at all, its best to set it equal to `.default` and forget about it.
        pub const default: @This() = .{};
    };
}

// --------------- angles ---------------

pub const ang = @import("angle.zig");

pub const Angle = Angle(f32);
pub const Angled = Angle(f64);
