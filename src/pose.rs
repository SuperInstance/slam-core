//! Pose representation for 2D SLAM.

/// 2D robot pose (position + orientation).
#[derive(Debug, Clone, Copy)]
pub struct Pose2D {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl Pose2D {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }

    /// Identity pose (origin, facing right).
    pub fn identity() -> Self {
        Self { x: 0.0, y: 0.0, theta: 0.0 }
    }

    /// Compose two poses: result = self ∘ other.
    /// Transforms `other` into `self`'s frame.
    pub fn compose(&self, other: &Pose2D) -> Pose2D {
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        Pose2D {
            x: self.x + cos_t * other.x - sin_t * other.y,
            y: self.y + sin_t * other.x + cos_t * other.y,
            theta: self.theta + other.theta,
        }
    }

    /// Inverse of this pose.
    pub fn inverse(&self) -> Pose2D {
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        Pose2D {
            x: -cos_t * self.x - sin_t * self.y,
            y: sin_t * self.x - cos_t * self.y,
            theta: -self.theta,
        }
    }

    /// Distance to another pose (position only).
    pub fn distance_to(&self, other: &Pose2D) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Normalize angle to [-π, π].
    pub fn normalize_angle(angle: f64) -> f64 {
        let mut a = angle;
        while a > std::f64::consts::PI { a -= 2.0 * std::f64::consts::PI; }
        while a < -std::f64::consts::PI { a += 2.0 * std::f64::consts::PI; }
        a
    }

    /// Create from an odometry delta.
    pub fn from_odometry(base: &Pose2D, dx: f64, dy: f64, dtheta: f64) -> Pose2D {
        let cos_t = base.theta.cos();
        let sin_t = base.theta.sin();
        Pose2D {
            x: base.x + cos_t * dx - sin_t * dy,
            y: base.y + sin_t * dx + cos_t * dy,
            theta: Self::normalize_angle(base.theta + dtheta),
        }
    }

    /// Transform a point from local to global frame.
    pub fn transform_point(&self, local_x: f64, local_y: f64) -> (f64, f64) {
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        (
            self.x + cos_t * local_x - sin_t * local_y,
            self.y + sin_t * local_x + cos_t * local_y,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pose_creation() {
        let p = Pose2D::new(1.0, 2.0, 0.5);
        assert!((p.x - 1.0).abs() < 1e-9);
        assert!((p.y - 2.0).abs() < 1e-9);
        assert!((p.theta - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_compose_identity() {
        let a = Pose2D::identity();
        let b = Pose2D::new(1.0, 2.0, 0.0);
        let c = a.compose(&b);
        assert!((c.x - 1.0).abs() < 1e-9);
        assert!((c.y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_inverse() {
        let p = Pose2D::new(1.0, 2.0, 0.0);
        let inv = p.inverse();
        assert!((inv.x + 1.0).abs() < 1e-9);
        assert!((inv.y + 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_compose_inverse() {
        let p = Pose2D::new(3.0, 4.0, 0.5);
        let inv = p.inverse();
        let result = p.compose(&inv);
        assert!(result.x.abs() < 1e-9);
        assert!(result.y.abs() < 1e-9);
    }

    #[test]
    fn test_normalize_angle() {
        assert!((Pose2D::normalize_angle(7.0) - (7.0 - 2.0 * std::f64::consts::PI)).abs() < 1e-9);
        assert!((Pose2D::normalize_angle(-4.0) - (-4.0 + 2.0 * std::f64::consts::PI)).abs() < 1e-9);
        assert!((Pose2D::normalize_angle(1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_distance() {
        let a = Pose2D::new(0.0, 0.0, 0.0);
        let b = Pose2D::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_transform_point() {
        let p = Pose2D::new(5.0, 5.0, 0.0);
        let (gx, gy) = p.transform_point(1.0, 1.0);
        assert!((gx - 6.0).abs() < 1e-9);
        assert!((gy - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_odometry() {
        let base = Pose2D::identity();
        let new_pose = Pose2D::from_odometry(&base, 1.0, 0.0, 0.0);
        assert!((new_pose.x - 1.0).abs() < 1e-9);
    }
}
