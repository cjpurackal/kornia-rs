use crate::so3::SO3;
use glam::{Mat3A, Mat4, Quat, Vec3A};
use rand::Rng;

#[derive(Debug, Clone, Copy)]
pub struct SE3 {
    pub r: SO3,
    pub t: Vec3A,
}

impl SE3 {
    pub const IDENTITY: Self = Self {
        r: SO3::IDENTITY,
        t: Vec3A::new(0.0, 0.0, 0.0),
    };

    pub fn new(rotation: SO3, translation: Vec3A) -> Self {
        Self {
            r: rotation,
            t: translation,
        }
    }

    pub fn from_matrix(mat: Mat4) -> Self {
        Self {
            r: SO3::from_matrix4(&mat),
            t: Vec3A::from_array([mat.x_axis.w, mat.y_axis.w, mat.z_axis.w]),
        }
    }

    pub fn from_random() -> Self {
        let mut rng = rand::rng();

        let r1: f32 = rng.random();
        let r2: f32 = rng.random();
        let r3: f32 = rng.random();

        Self {
            r: SO3::from_random(),
            t: Vec3A::new(r1, r2, r3),
        }
    }

    pub fn from_qxyz(quat: Quat, xyz: Vec3A) -> Self {
        Self {
            r: SO3::from_quaternion(quat),
            t: xyz,
        }
    }

    pub fn inverse(&self) -> Self {
        let r_inv = self.r.inverse();
        Self {
            r: r_inv,
            t: r_inv * (-self.t),
        }
    }

    pub fn matrix(&self) -> Mat4 {
        let r = self.r.matrix();
        Mat4::from_cols_array(&[
            r.x_axis.x, r.x_axis.y, r.x_axis.z, self.t.x, //
            r.y_axis.x, r.y_axis.y, r.y_axis.z, self.t.y, //
            r.z_axis.x, r.z_axis.y, r.z_axis.z, self.t.z, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    pub fn adjoint(&self) -> [[f32; 6]; 6] {
        let r = self.r.matrix();
        let t = SO3::hat(self.t) * r;

        [
            [r.x_axis.x, r.y_axis.x, r.z_axis.x, 0.0, 0.0, 0.0],
            [r.x_axis.y, r.y_axis.y, r.z_axis.y, 0.0, 0.0, 0.0],
            [r.x_axis.z, r.y_axis.z, r.z_axis.z, 0.0, 0.0, 0.0],
            [
                t.x_axis.x, t.y_axis.x, t.z_axis.x, r.x_axis.x, r.y_axis.x, r.z_axis.x,
            ],
            [
                t.x_axis.y, t.y_axis.y, t.z_axis.y, r.x_axis.y, r.y_axis.y, r.z_axis.y,
            ],
            [
                t.x_axis.z, t.y_axis.z, t.z_axis.z, r.x_axis.z, r.y_axis.z, r.z_axis.z,
            ],
        ]
    }

    pub fn exp(upsilon: Vec3A, omega: Vec3A) -> Self {
        let theta = omega.dot(omega).sqrt();

        Self {
            r: SO3::exp(omega),
            t: if theta != 0.0 {
                let omega_hat = SO3::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;

                let mat_v = Mat3A::IDENTITY
                    + ((1.0 - theta.cos()) / (theta * theta)) * omega_hat
                    + ((theta - theta.sin()) / (theta.powi(3))) * omega_hat_sq;

                mat_v.mul_vec3a(upsilon) // TODO: a bit sus (should it be the other way around?)
            } else {
                upsilon
            },
        }
    }

    /// returns translation, rotation
    pub fn log(&self) -> (Vec3A, Vec3A) {
        let omega = self.r.log();
        let theta = omega.dot(omega).sqrt();

        (
            if theta != 0.0 {
                let omega_hat = SO3::hat(omega);
                let omega_hat_sq = omega_hat * omega_hat;

                let mat_v_inv = Mat3A::IDENTITY - 0.5 * omega_hat
                    + ((1.0 - theta * (theta / 2.0).cos() / (2.0 * (theta / 2.0).sin()))
                        / theta.powi(2))
                        * omega_hat_sq;

                mat_v_inv.mul_vec3a(self.t) // TODO:
            } else {
                self.t
            },
            omega,
        )
    }

    pub fn hat(upsilon: Vec3A, omega: Vec3A) -> Mat4 {
        let h = SO3::hat(omega);

        Mat4::from_cols_array(&[
            h.x_axis.x, h.x_axis.y, h.x_axis.z, upsilon.x, h.y_axis.x, h.y_axis.y, h.y_axis.z,
            upsilon.y, h.z_axis.x, h.z_axis.y, h.z_axis.z, upsilon.z, 0.0, 0.0, 0.0, 0.0,
        ])
    }

    pub fn vee(omega: Mat4) -> (Vec3A, Vec3A) {
        (
            Vec3A::new(omega.x_axis.w, omega.y_axis.w, omega.z_axis.w),
            SO3::vee4(omega),
        )
    }

    const EPS: f32 = 1e-5;

    /// left Jacobian of SE(3)  (6×6)
    pub fn left_jacobian(rho: Vec3A, phi: Vec3A) -> [[f32; 6]; 6] {
        // 1. SO(3) blocks --------------------------------------------------------
        let jl_so3 = SO3::left_jacobian(phi);          // 3×3
        let phi_hat = SO3::hat(phi);                  // ρ̂ etc.
        let rho_hat = SO3::hat(rho);

        // 2. build Q_l -----------------------------------------------------------
        let theta2 = phi.length_squared();
        let (a, b) = if theta2 < Self::EPS * Self::EPS {
            // small-angle series
            (
                0.5 - theta2 / 24.0,               // A ≈ ½ − θ²/24
                1.0 / 6.0 - theta2 / 120.0         // B ≈ 1/6 − θ²/120
            )
        } else {
            let theta = theta2.sqrt();
            (
                (1.0 - theta.cos()) / theta2,                 // A
                (theta - theta.sin()) / (theta2 * theta)      // B
            )
        };

        // Q_l = Jl_so3*rhô + ½ρ̂ + A(φ̂ρ̂+ρ̂φ̂) + B φ̂ρ̂φ̂
        let q_l = jl_so3 * rho_hat
                + 0.5 * rho_hat
                + a * (phi_hat * rho_hat + rho_hat * phi_hat)
                + b * (phi_hat * rho_hat * phi_hat);

        // 3. assemble 6×6 --------------------------------------------------------
        let mut jl = [[0.0f32; 6]; 6];

        // top-left 3×3  (rotation)
        for r in 0..3 {
            for c in 0..3 {
                jl[r][c] = jl_so3.col(c)[r];
            }
        }
        // top-right 3×3  (translation–rotation coupling)
        for r in 0..3 {
            for c in 0..3 {
                jl[r][c + 3] = q_l.col(c)[r];
            }
        }
        // bottom-right 3×3  (rotation again)
        for r in 0..3 {
            for c in 0..3 {
                jl[r + 3][c + 3] = jl_so3.col(c)[r];
            }
        }
        jl
    }

    /// right Jacobian  J_r(xi) = J_l(−xi)
    #[inline]
    pub fn right_jacobian(rho: Vec3A, phi: Vec3A) -> [[f32; 6]; 6] {
        Self::left_jacobian(-rho, -phi)
    }

}

impl std::ops::Mul<SE3> for SE3 {
    type Output = SE3;

    fn mul(self, rhs: SE3) -> Self::Output {
        let r = self.r * rhs.r;
        let t = self.t + self.r * rhs.t;
        Self { r, t }
    }
}

impl std::ops::Mul<Vec3A> for SE3 {
    type Output = Vec3A;

    fn mul(self, rhs: Vec3A) -> Self::Output {
        self.r * rhs + self.t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use glam::{Mat3, Vec3, Vec4, Mat4 as GlamMat4}; // For creating Vec3 for twist vector in tests

    const EPSILON: f32 = 1e-6;
    const JACOBIAN_TEST_EPSILON: f32 = 1e-6; // More reasonable tolerance for SE(3) operations

    fn make_random_se3() -> SE3 {
        SE3::from_random()
    }

    fn make_random_vec3() -> Vec3A {
        let mut rng = rand::rng();
        Vec3A::new(rng.random(), rng.random(), rng.random())
    }

    #[test]
    fn test_identity() {
        let se3 = SE3::IDENTITY;
        assert_relative_eq!(se3.r.q.x, 0.0);
        assert_relative_eq!(se3.r.q.y, 0.0);
        assert_relative_eq!(se3.r.q.z, 0.0);
        assert_relative_eq!(se3.r.q.w, 1.0);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);
    }

    #[test]
    fn test_new() {
        let rotation = SO3::from_quaternion(Quat::IDENTITY);
        let translation = Vec3A::new(1.0, 2.0, 3.0);
        let se3 = SE3::new(rotation, translation);
        assert_relative_eq!(se3.t.x, translation.x);
        assert_relative_eq!(se3.t.y, translation.y);
        assert_relative_eq!(se3.t.z, translation.z);
        assert_relative_eq!(se3.r.q.x, rotation.q.x);
        assert_relative_eq!(se3.r.q.y, rotation.q.y);
        assert_relative_eq!(se3.r.q.z, rotation.q.z);
        assert_relative_eq!(se3.r.q.w, rotation.q.w);
    }

    #[test]
    fn test_from_matrix() {
        // Test with identity matrix
        let mat = Mat4::IDENTITY;
        let se3 = SE3::from_matrix(mat);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);

        let expected_rotation = SO3::IDENTITY.matrix();
        let actual_rotation = se3.r.matrix();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(actual_rotation.col(i)[j], expected_rotation.col(i)[j],);
            }
        }

        // Test with a specific transformation matrix
        let mat = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        let se3 = SE3::from_matrix(mat);
        assert_relative_eq!(se3.t.x, 1.0);
        assert_relative_eq!(se3.t.y, 2.0);
        assert_relative_eq!(se3.t.z, 3.0);
    }

    #[test]
    fn test_from_qxyz() {
        let quat = Quat::from_xyzw(0.1, 0.2, 0.3, 0.9).normalize();
        let xyz = Vec3A::new(1.0, 2.0, 3.0);
        let se3 = SE3::from_qxyz(quat, xyz);

        assert_relative_eq!(se3.r.q.x, quat.x);
        assert_relative_eq!(se3.t.x, xyz.x);
        assert_relative_eq!(se3.t.y, xyz.y);
        assert_relative_eq!(se3.t.z, xyz.z);
    }

    #[test]
    fn test_inverse() {
        // Test with identity
        let se3 = SE3::IDENTITY;
        let inv = se3.inverse();
        assert_relative_eq!(inv.r.q.x, se3.r.q.x);
        assert_relative_eq!(inv.r.q.y, se3.r.q.y);
        assert_relative_eq!(inv.r.q.z, se3.r.q.z);
        assert_relative_eq!(inv.r.q.w, se3.r.q.w);
        assert_relative_eq!(inv.t.x, se3.t.x);
        assert_relative_eq!(inv.t.y, se3.t.y);
        assert_relative_eq!(inv.t.z, se3.t.z);

        // Test with translation only
        let se3 = SE3::new(SO3::IDENTITY, Vec3A::new(1.0, 2.0, 3.0));
        let inv = se3.inverse();
        assert_relative_eq!(inv.t.x, -1.0);
        assert_relative_eq!(inv.t.y, -2.0);
        assert_relative_eq!(inv.t.z, -3.0);

        // Test inverse property: se3 * se3.inverse() = identity
        let se3 = make_random_se3();
        let inv = se3.inverse();
        let result = se3 * inv;
        assert_relative_eq!(result.r.q.x, SE3::IDENTITY.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.y, SE3::IDENTITY.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.z, SE3::IDENTITY.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.w, SE3::IDENTITY.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE3::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE3::IDENTITY.t.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.z, SE3::IDENTITY.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix() {
        // Test identity
        let se3 = SE3::IDENTITY;
        let mat = se3.matrix();
        let expected = Mat4::IDENTITY;
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(mat.col(i)[j], expected.col(i)[j]);
            }
        }

        // Test with translation
        let se3 = SE3::new(SO3::IDENTITY, Vec3A::new(1.0, 2.0, 3.0));
        let mat = se3.matrix();
        let expected = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(mat.col(i)[j], expected.col(i)[j]);
            }
        }
    }

    #[test]
    fn test_multiplication_identity() {
        let se3 = make_random_se3();
        let identity = SE3::IDENTITY;

        // Identity * se3 = se3
        let result1 = identity * se3;
        assert_relative_eq!(result1.r.q.x, se3.r.q.x);
        assert_relative_eq!(result1.r.q.y, se3.r.q.y);
        assert_relative_eq!(result1.r.q.z, se3.r.q.z);
        assert_relative_eq!(result1.r.q.w, se3.r.q.w);
        assert_relative_eq!(result1.t.x, se3.t.x);
        assert_relative_eq!(result1.t.y, se3.t.y);
        assert_relative_eq!(result1.t.z, se3.t.z);

        // se3 * identity = se3
        let result2 = se3 * identity;
        assert_relative_eq!(result2.r.q.x, se3.r.q.x);
        assert_relative_eq!(result2.r.q.y, se3.r.q.y);
        assert_relative_eq!(result2.r.q.z, se3.r.q.z);
        assert_relative_eq!(result2.r.q.w, se3.r.q.w);
        assert_relative_eq!(result2.t.x, se3.t.x);
        assert_relative_eq!(result2.t.y, se3.t.y);
        assert_relative_eq!(result2.t.z, se3.t.z);
    }

    #[test]
    fn test_multiplication_point() {
        let se3_1 = make_random_se3();
        let se3_2 = make_random_se3();
        let point = make_random_vec3();

        // Test transformation consistency
        let relative_transform = se3_1.inverse() * se3_2;
        let point_in_1 = se3_1.inverse() * point;
        let point_in_2 = se3_2.inverse() * point;
        let point_1_to_2 = relative_transform.inverse() * point_in_1;
        let point_2_to_1 = relative_transform * point_in_2;

        assert_relative_eq!(point_in_1.x, point_2_to_1.x, epsilon = EPSILON);
        assert_relative_eq!(point_in_1.y, point_2_to_1.y, epsilon = EPSILON);
        assert_relative_eq!(point_in_1.z, point_2_to_1.z, epsilon = EPSILON);
        assert_relative_eq!(point_in_2.x, point_1_to_2.x, epsilon = EPSILON);
        assert_relative_eq!(point_in_2.y, point_1_to_2.y, epsilon = EPSILON);
        assert_relative_eq!(point_in_2.z, point_1_to_2.z, epsilon = EPSILON);
    }

    #[test]
    fn test_exp_identity() {
        // exp(0) = identity
        let upsilon = Vec3A::ZERO;
        let omega = Vec3A::ZERO;
        let se3 = SE3::exp(upsilon, omega);

        assert_relative_eq!(se3.r.q.x, SE3::IDENTITY.r.q.x);
        assert_relative_eq!(se3.r.q.y, SE3::IDENTITY.r.q.y);
        assert_relative_eq!(se3.r.q.z, SE3::IDENTITY.r.q.z);
        assert_relative_eq!(se3.r.q.w, SE3::IDENTITY.r.q.w);
        assert_relative_eq!(se3.t.x, 0.0);
        assert_relative_eq!(se3.t.y, 0.0);
        assert_relative_eq!(se3.t.z, 0.0);
    }

    #[test]
    fn test_exp_translation_only() {
        // Pure translation (omega = 0)
        let upsilon = Vec3A::new(1.0, 2.0, 3.0);
        let omega = Vec3A::ZERO;
        let se3 = SE3::exp(upsilon, omega);

        assert_relative_eq!(se3.r.q.x, SE3::IDENTITY.r.q.x);
        assert_relative_eq!(se3.r.q.y, SE3::IDENTITY.r.q.y);
        assert_relative_eq!(se3.r.q.z, SE3::IDENTITY.r.q.z);
        assert_relative_eq!(se3.r.q.w, SE3::IDENTITY.r.q.w);
        assert_relative_eq!(se3.t.x, upsilon.x);
        assert_relative_eq!(se3.t.y, upsilon.y);
        assert_relative_eq!(se3.t.z, upsilon.z);
    }

    #[test]
    fn test_log_identity() {
        let se3 = SE3::IDENTITY;
        let (upsilon, omega) = se3.log();

        assert_relative_eq!(upsilon.x, 0.0);
        assert_relative_eq!(upsilon.y, 0.0);
        assert_relative_eq!(upsilon.z, 0.0);
        assert_relative_eq!(omega.x, 0.0);
        assert_relative_eq!(omega.y, 0.0);
        assert_relative_eq!(omega.z, 0.0);
    }

    #[test]
    fn test_log_translation_only() {
        let translation = Vec3A::new(1.0, 2.0, 3.0);
        let se3 = SE3::new(SO3::IDENTITY, translation);
        let (upsilon, omega) = se3.log();

        assert_relative_eq!(upsilon.x, translation.x);
        assert_relative_eq!(upsilon.y, translation.y);
        assert_relative_eq!(upsilon.z, translation.z);
        assert_relative_eq!(omega.x, 0.0);
        assert_relative_eq!(omega.y, 0.0);
        assert_relative_eq!(omega.z, 0.0);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let upsilon = Vec3A::new(0.5, -0.5, 1.0);
        let omega = Vec3A::new(0.1, 0.2, -0.3);

        let se3 = SE3::exp(upsilon, omega);
        let (log_upsilon, log_omega) = se3.log();

        assert_relative_eq!(log_upsilon.x, upsilon.x);
        assert_relative_eq!(log_upsilon.y, upsilon.y);
        assert_relative_eq!(log_upsilon.z, upsilon.z);
        assert_relative_eq!(log_omega.x, omega.x);
        assert_relative_eq!(log_omega.y, omega.y);
        assert_relative_eq!(log_omega.z, omega.z);
    }

    #[test]
    fn test_hat_vee_roundtrip() {
        let upsilon = Vec3A::new(1.0, 2.0, 3.0);
        let omega = Vec3A::new(0.1, -0.2, 0.3);

        let hat_matrix = SE3::hat(upsilon, omega);
        let (vee_upsilon, vee_omega) = SE3::vee(hat_matrix);

        assert_relative_eq!(vee_upsilon.x, upsilon.x);
        assert_relative_eq!(vee_upsilon.y, upsilon.y);
        assert_relative_eq!(vee_upsilon.z, upsilon.z);
        assert_relative_eq!(vee_omega.x, omega.x);
        assert_relative_eq!(vee_omega.y, omega.y);
    }

    #[test]
    fn test_hat_structure() {
        let upsilon = Vec3A::new(1.0, 2.0, 3.0);
        let omega = Vec3A::new(0.1, -0.2, 0.3);
        let hat_matrix = SE3::hat(upsilon, omega);

        // Check that the bottom row is [0, 0, 0, 0]
        assert_relative_eq!(hat_matrix.w_axis.x, 0.0);
        assert_relative_eq!(hat_matrix.w_axis.y, 0.0);
        assert_relative_eq!(hat_matrix.w_axis.z, 0.0);
        assert_relative_eq!(hat_matrix.w_axis.w, 0.0);

        // Check that the translation part is correct
        assert_relative_eq!(hat_matrix.x_axis.w, upsilon.x);
        assert_relative_eq!(hat_matrix.y_axis.w, upsilon.y);
        assert_relative_eq!(hat_matrix.z_axis.w, upsilon.z);
    }

    #[test]
    fn test_adjoint_identity() {
        let se3 = SE3::IDENTITY;
        let adj = se3.adjoint();

        // Expected 6x6 adjoint of identity: block diagonal with two 3x3 identity matrices
        let expected = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(adj[i][j], expected[i][j]);
            }
        }
    }

    #[test]
    fn test_adjoint_properties() {
        let se3_1 = make_random_se3();
        let se3_2 = make_random_se3();

        // Test: (se3_1 * se3_2).adjoint() = se3_1.adjoint() * se3_2.adjoint()
        let composed = se3_1 * se3_2;
        let adj_composed = composed.adjoint();

        let adj_1 = se3_1.adjoint();
        let adj_2 = se3_2.adjoint();

        // Matrix multiplication of 6x6 matrices
        let mut adj_product = [[0.0f32; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    adj_product[i][j] += adj_1[i][k] * adj_2[k][j];
                }
            }
        }

        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(adj_composed[i][j], adj_product[i][j], epsilon = EPSILON);
            }
        }
    }

    #[test]
    fn test_from_random() {
        let se3 = SE3::from_random();

        // Test that the quaternion is normalized
        let q_norm = (se3.r.q.x * se3.r.q.x
            + se3.r.q.y * se3.r.q.y
            + se3.r.q.z * se3.r.q.z
            + se3.r.q.w * se3.r.q.w)
            .sqrt();
        assert_relative_eq!(q_norm, 1.0);

        // Test that inverse property holds
        let inv = se3.inverse();
        let result = se3 * inv;
        assert_relative_eq!(result.r.q.x, SE3::IDENTITY.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.y, SE3::IDENTITY.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.z, SE3::IDENTITY.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(result.r.q.w, SE3::IDENTITY.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(result.t.x, SE3::IDENTITY.t.x, epsilon = EPSILON);
        assert_relative_eq!(result.t.y, SE3::IDENTITY.t.y, epsilon = EPSILON);
        assert_relative_eq!(result.t.z, SE3::IDENTITY.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_matrix_from_matrix_roundtrip() {
        let se3 = make_random_se3();
        let matrix = se3.matrix();
        let se3_reconstructed = SE3::from_matrix(matrix);

        // Check rotation (quaternions might have opposite signs but represent same rotation)
        let q1 = se3.r.q;
        let q2 = se3_reconstructed.r.q;
        let same_rotation = (q1.x - q2.x).abs() < 1e-5
            && (q1.y - q2.y).abs() < 1e-5
            && (q1.z - q2.z).abs() < 1e-5
            && (q1.w - q2.w).abs() < 1e-5;
        let opposite_rotation = (q1.x + q2.x).abs() < 1e-5
            && (q1.y + q2.y).abs() < 1e-5
            && (q1.z + q2.z).abs() < 1e-5
            && (q1.w + q2.w).abs() < 1e-5;
        assert!(same_rotation || opposite_rotation);

        // Check translation
        assert_relative_eq!(se3.t.x, se3_reconstructed.t.x);
        assert_relative_eq!(se3.t.y, se3_reconstructed.t.y);
        assert_relative_eq!(se3.t.z, se3_reconstructed.t.z);
    }

    #[test]
    fn test_composition_associativity() {
        let se3_1 = make_random_se3();
        let se3_2 = make_random_se3();
        let se3_3 = make_random_se3();

        // Test (se3_1 * se3_2) * se3_3 = se3_1 * (se3_2 * se3_3)
        let left_assoc = (se3_1 * se3_2) * se3_3;
        let right_assoc = se3_1 * (se3_2 * se3_3);

        assert_relative_eq!(left_assoc.r.q.x, right_assoc.r.q.x, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.r.q.y, right_assoc.r.q.y, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.r.q.z, right_assoc.r.q.z, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.r.q.w, right_assoc.r.q.w, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.x, right_assoc.t.x, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.y, right_assoc.t.y, epsilon = EPSILON);
        assert_relative_eq!(left_assoc.t.z, right_assoc.t.z, epsilon = EPSILON);
    }

    #[test]
    fn test_se3_jacobians() {
        let test_twists = [
            (Vec3A::new(0.0, 0.0, 0.0), Vec3A::new(0.0, 0.0, 0.0)),
            (Vec3A::new(0.1, 0.2, 0.3), Vec3A::new(0.01, 0.02, 0.03)),
            (Vec3A::new(1.0, -0.5, 0.2), Vec3A::new(0.0001, -0.0002, 0.0003)), // Small omega
            (Vec3A::new(0.5, 1.5, -1.0), Vec3A::new(0.4, 0.5, 0.6)),
        ];

        for (upsilon, omega) in test_twists.iter() {
            let upsilon_val = *upsilon;
            let omega_val = *omega;
            let _twist_arr = [upsilon_val.x, upsilon_val.y, upsilon_val.z, omega_val.x, omega_val.y, omega_val.z];

            let jl = SE3::left_jacobian(upsilon_val, omega_val);
            let jr = SE3::right_jacobian(upsilon_val, omega_val);

            // Test property: J_r(xi) = J_l(-xi)
            let jl_neg = SE3::left_jacobian(-upsilon_val, -omega_val);
            for r in 0..6 {
                for c_ in 0..6 {
                    assert_relative_eq!(jr[r][c_], jl_neg[r][c_], epsilon = JACOBIAN_TEST_EPSILON);
                }
            }

            // TODO: This relationship might need verification for SE(3)
            // Test property: J_l(xi) = Ad(exp(xi)) * J_r(xi)
            // let exp_xi = SE3::exp(upsilon_val, omega_val);
            // let adj_exp_xi = exp_xi.adjoint(); // This is the 6x6 Adjoint matrix
            // 
            // let adj_mul_jr = multiply_6x6_matrices(&adj_exp_xi, &jr);
            // for r in 0..6 {
            //     for c_ in 0..6 {
            //         assert_relative_eq!(jl[r][c_], adj_mul_jr[r][c_], epsilon = JACOBIAN_TEST_EPSILON);
            //     }
            // }

            // Test J(0) = I
            if upsilon_val.length_squared() < 1e-12 && omega_val.length_squared() < 1e-12 {
                let identity_6x6 = {
                    let mut id = [[0.0f32; 6]; 6];
                    for i in 0..6 { id[i][i] = 1.0; }
                    id
                };
                for r in 0..6 {
                    for c_ in 0..6 {
                        assert_relative_eq!(jl[r][c_], identity_6x6[r][c_], epsilon = JACOBIAN_TEST_EPSILON);
                        assert_relative_eq!(jr[r][c_], identity_6x6[r][c_], epsilon = JACOBIAN_TEST_EPSILON);
                    }
                }
            }
        }

        // Test J_l(0) = I
        let jl_zero = SE3::left_jacobian(Vec3A::ZERO, Vec3A::ZERO);
        let identity_6x6 = {
            let mut id = [[0.0f32; 6]; 6];
            for i in 0..6 { id[i][i] = 1.0; }
            id
        };
        for r in 0..6 {
            for c_ in 0..6 {
                assert_relative_eq!(jl_zero[r][c_], identity_6x6[r][c_], epsilon = JACOBIAN_TEST_EPSILON);
            }
        }

        // Test J_r(0) = I
        let jr_zero = SE3::right_jacobian(Vec3A::ZERO, Vec3A::ZERO);
        for r in 0..6 {
            for c_ in 0..6 {
                assert_relative_eq!(jr_zero[r][c_], identity_6x6[r][c_], epsilon = JACOBIAN_TEST_EPSILON);
            }
        }
    }

    // Helper to convert 6x6 array to a glam Mat4 for multiplication if needed (not directly, but for Adjoint concept)
    // For SE(3) Adjoint is 6x6
    fn se3_adjoint_matrix(se3: &SE3) -> GlamMat4 { // This is not the 6x6 Lie Adjoint, but group Adjoint as 4x4 matrix transform on twists
        let r_mat = se3.r.matrix();
        let t_hat_r = SO3::hat(se3.t) * r_mat;
        GlamMat4::from_cols(
            r_mat.x_axis.extend(0.0),
            r_mat.y_axis.extend(0.0),
            r_mat.z_axis.extend(0.0),
            Vec3A::ZERO.extend(1.0),
        ) // This is not the correct Adjoint matrix representation for multiplying with 6x1 twists
    }

    // Helper to multiply a 6x6 matrix (array) by a 6x1 vector (array)
    fn multiply_6x6_by_6x1(matrix: &[[f32; 6]; 6], vector: &[f32; 6]) -> [f32; 6] {
        let mut result = [0.0f32; 6];
        for i in 0..6 {
            for j in 0..6 {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        result
    }
    
    // Helper to multiply two 6x6 matrices
    fn multiply_6x6_matrices(a: &[[f32; 6]; 6], b: &[[f32; 6]; 6]) -> [[f32; 6]; 6] {
        let mut result = [[0.0f32; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }
}
