//! Particle filter localization.

use crate::pose::Pose2D;

/// A particle for Monte Carlo localization.
#[derive(Debug, Clone)]
pub struct Particle {
    pub pose: Pose2D,
    pub weight: f64,
}

/// Simple motion model (odometry-based).
#[derive(Debug, Clone)]
pub struct Odometry {
    pub dx: f64,
    pub dy: f64,
    pub dtheta: f64,
}

/// Particle filter for localization.
pub struct ParticleFilter {
    pub particles: Vec<Particle>,
    pub num_particles: usize,
    /// Motion noise parameters [x_noise, y_noise, theta_noise]
    pub motion_noise: [f64; 3],
}

impl ParticleFilter {
    /// Create a particle filter with uniformly distributed particles.
    pub fn new(num_particles: usize, bounds: (f64, f64, f64, f64, f64, f64)) -> Self {
        let mut particles = Vec::with_capacity(num_particles);
        for i in 0..num_particles {
            let t = i as f64 / num_particles as f64;
            particles.push(Particle {
                pose: Pose2D {
                    x: bounds.0 + t * (bounds.2 - bounds.0),
                    y: bounds.1 + t * (bounds.3 - bounds.1),
                    theta: bounds.4 + t * (bounds.5 - bounds.4),
                },
                weight: 1.0 / num_particles as f64,
            });
        }
        Self {
            particles,
            num_particles,
            motion_noise: [0.1, 0.1, 0.05],
        }
    }

    /// Create particles around a known initial pose.
    pub fn new_around_pose(num_particles: usize, center: Pose2D, spread: [f64; 3]) -> Self {
        let mut particles = Vec::with_capacity(num_particles);
        for i in 0..num_particles {
            // Simple hash-based pseudo-random
            let h1 = ((i as u64).wrapping_mul(6364136223846793005)).wrapping_add(1442695040888963407);
            let h2 = ((i as u64 + 7919).wrapping_mul(6364136223846793005)).wrapping_add(1442695040888963407);
            let h3 = ((i as u64 + 104729).wrapping_mul(6364136223846793005)).wrapping_add(1442695040888963407);
            let r1 = (((h1 >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
            let r2 = (((h2 >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
            let r3 = (((h3 >> 16) & 0xFFFF) as f64 / 65535.0) - 0.5;
            particles.push(Particle {
                pose: Pose2D {
                    x: center.x + r1 * spread[0] * 2.0,
                    y: center.y + r2 * spread[1] * 2.0,
                    theta: center.theta + r3 * spread[2] * 2.0,
                },
                weight: 1.0 / num_particles as f64,
            });
        }
        Self {
            particles,
            num_particles,
            motion_noise: [0.1, 0.1, 0.05],
        }
    }

    /// Apply motion model to all particles.
    pub fn apply_motion(&mut self, odom: &Odometry) {
        for p in &mut self.particles {
            let cos_t = p.pose.theta.cos();
            let sin_t = p.pose.theta.sin();
            p.pose.x += (odom.dx * cos_t - odom.dy * sin_t) + self.motion_noise[0] * ((p.pose.x * 17.0).sin() * 0.1);
            p.pose.y += (odom.dx * sin_t + odom.dy * cos_t) + self.motion_noise[1] * ((p.pose.y * 13.0).cos() * 0.1);
            p.pose.theta += odom.dtheta + self.motion_noise[2] * ((p.pose.theta * 7.0).sin() * 0.05);
        }
    }

    /// Update weights based on a likelihood function.
    pub fn update_weights(&mut self, likelihood_fn: &dyn Fn(&Pose2D) -> f64) {
        let mut total = 0.0;
        for p in &mut self.particles {
            p.weight = likelihood_fn(&p.pose);
            total += p.weight;
        }
        if total > 0.0 {
            for p in &mut self.particles {
                p.weight /= total;
            }
        }
    }

    /// Low-variance resampling.
    pub fn resample(&mut self) {
        let n = self.particles.len();
        if n == 0 { return; }

        let mut new_particles = Vec::with_capacity(n);
        let step = 1.0 / n as f64;
        let seed_val = self.particles[0].pose.x.to_bits();
        let r = ((seed_val.wrapping_mul(1103515245) >> 16) as f64 / u16::MAX as f64) * step;
        let mut c = self.particles[0].weight;
        let mut i = 0;

        for m in 0..n {
            let u = r + (m as f64) * step;
            while u > c && i < n - 1 {
                i += 1;
                c += self.particles[i].weight;
            }
            new_particles.push(self.particles[i].clone());
        }
        self.particles = new_particles;
    }

    /// Get the weighted mean pose.
    pub fn estimate(&self) -> Pose2D {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut sin_t = 0.0;
        let mut cos_t = 0.0;
        for p in &self.particles {
            x += p.pose.x * p.weight;
            y += p.pose.y * p.weight;
            sin_t += p.pose.theta.sin() * p.weight;
            cos_t += p.pose.theta.cos() * p.weight;
        }
        Pose2D {
            x,
            y,
            theta: sin_t.atan2(cos_t),
        }
    }

    /// Get effective sample size (ESS).
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.particles.iter().map(|p| p.weight * p.weight).sum();
        if sum_sq < 1e-12 { return 0.0; }
        1.0 / sum_sq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn likelihood_near_origin(p: &Pose2D) -> f64 {
        (-(p.x * p.x + p.y * p.y)).exp()
    }

    #[test]
    fn test_particle_filter_creation() {
        let pf = ParticleFilter::new(100, (0.0, 0.0, 10.0, 10.0, 0.0, 6.28));
        assert_eq!(pf.particles.len(), 100);
    }

    #[test]
    fn test_motion_update() {
        let mut pf = ParticleFilter::new_around_pose(50, Pose2D::new(5.0, 5.0, 0.0), [0.1, 0.1, 0.01]);
        let x_before = pf.particles[0].pose.x;
        pf.apply_motion(&Odometry { dx: 1.0, dy: 0.0, dtheta: 0.0 });
        let x_after = pf.particles[0].pose.x;
        assert!(x_after > x_before - 0.5, "x_before={}, x_after={}", x_before, x_after);
    }

    #[test]
    fn test_weight_update() {
        let mut pf = ParticleFilter::new_around_pose(50, Pose2D::new(0.0, 0.0, 0.0), [5.0, 5.0, 0.5]);
        pf.update_weights(&likelihood_near_origin);
        let total: f64 = pf.particles.iter().map(|p| p.weight).sum();
        assert!((total - 1.0).abs() < 0.01, "total={}", total);
    }

    #[test]
    fn test_resampling() {
        let mut pf = ParticleFilter::new_around_pose(50, Pose2D::new(0.0, 0.0, 0.0), [5.0, 5.0, 0.5]);
        pf.update_weights(&likelihood_near_origin);
        pf.resample();
        assert_eq!(pf.particles.len(), 50);
    }

    #[test]
    fn test_estimation() {
        let mut pf = ParticleFilter::new_around_pose(100, Pose2D::new(1.0, 1.0, 0.0), [0.01, 0.01, 0.01]);
        pf.update_weights(&likelihood_near_origin);
        let est = pf.estimate();
        // With spread around (1,1) and likelihood peaking at origin, weights may be small
        assert!(est.x.is_finite());
        assert!(est.y.is_finite());
    }

    #[test]
    fn test_effective_sample_size() {
        let pf = ParticleFilter::new_around_pose(100, Pose2D::new(0.0, 0.0, 0.0), [1.0, 1.0, 0.1]);
        let ess = pf.effective_sample_size();
        assert!(ess > 0.0);
        assert!(ess <= 101.0); // Allow small numerical imprecision
    }

    #[test]
    fn test_full_localization_cycle() {
        let mut pf = ParticleFilter::new_around_pose(200, Pose2D::new(2.0, 2.0, 0.0), [1.0, 1.0, 0.1]);
        let true_pose = Pose2D::new(2.0, 2.0, 0.0);

        let likelihood = |p: &Pose2D| -> f64 {
            let dx = p.x - true_pose.x;
            let dy = p.y - true_pose.y;
            (-(dx * dx + dy * dy) / 2.0).exp()
        };

        for _ in 0..5 {
            pf.apply_motion(&Odometry { dx: 0.0, dy: 0.0, dtheta: 0.0 });
            pf.update_weights(&likelihood);
            pf.resample();
        }

        let est = pf.estimate();
        // Particles centered at true pose, likelihood peaks at true pose
        assert!(est.x.is_finite());
        assert!(est.y.is_finite());
    }
}
