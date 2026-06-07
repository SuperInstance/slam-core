//! Map representation for SLAM.

use crate::landmark::Landmark;
use crate::pose::Pose2D;

/// A 2D map containing landmarks and optionally an occupancy representation.
#[derive(Debug, Clone)]
pub struct Map2D {
    pub landmarks: Vec<Landmark>,
    pub bounds: (f64, f64, f64, f64), // (min_x, min_y, max_x, max_y)
}

impl Map2D {
    pub fn new(bounds: (f64, f64, f64, f64)) -> Self {
        Self {
            landmarks: Vec::new(),
            bounds,
        }
    }

    /// Add a landmark to the map.
    pub fn add_landmark(&mut self, lm: Landmark) {
        self.landmarks.push(lm);
    }

    /// Find the nearest landmark to a given position.
    pub fn nearest_landmark(&self, x: f64, y: f64) -> Option<(usize, f64)> {
        let mut best = None;
        let mut best_dist = f64::INFINITY;
        for (i, lm) in self.landmarks.iter().enumerate() {
            let d = ((lm.x - x).powi(2) + (lm.y - y).powi(2)).sqrt();
            if d < best_dist {
                best_dist = d;
                best = Some((i, d));
            }
        }
        best
    }

    /// Find all landmarks within a radius.
    pub fn landmarks_within(&self, x: f64, y: f64, radius: f64) -> Vec<usize> {
        self.landmarks.iter().enumerate()
            .filter(|(_, lm)| ((lm.x - x).powi(2) + (lm.y - y).powi(2)).sqrt() < radius)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute similarity between two maps (landmark matching).
    pub fn similarity(&self, other: &Map2D, threshold: f64) -> f64 {
        if self.landmarks.is_empty() || other.landmarks.is_empty() {
            return 0.0;
        }
        let mut matches = 0;
        for lm in &self.landmarks {
            if let Some((_, d)) = other.nearest_landmark(lm.x, lm.y) {
                if d < threshold {
                    matches += 1;
                }
            }
        }
        matches as f64 / self.landmarks.len().max(other.landmarks.len()) as f64
    }

    /// Merge another map into this one.
    pub fn merge(&mut self, other: &Map2D, threshold: f64) {
        for lm in &other.landmarks {
            if let Some((_, d)) = self.nearest_landmark(lm.x, lm.y) {
                if d < threshold {
                    continue; // Skip duplicate
                }
            }
            self.add_landmark(lm.clone());
        }
    }

    /// Compute map bounds from landmarks.
    pub fn compute_bounds(&self) -> (f64, f64, f64, f64) {
        if self.landmarks.is_empty() {
            return self.bounds;
        }
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for lm in &self.landmarks {
            min_x = min_x.min(lm.x);
            min_y = min_y.min(lm.y);
            max_x = max_x.max(lm.x);
            max_y = max_y.max(lm.y);
        }
        (min_x, min_y, max_x, max_y)
    }
}

/// Submap for large-scale mapping.
pub struct SubMap {
    pub map: Map2D,
    pub origin: Pose2D,
}

impl SubMap {
    pub fn new(origin: Pose2D, bounds: (f64, f64, f64, f64)) -> Self {
        Self {
            map: Map2D::new(bounds),
            origin,
        }
    }

    /// Transform a global landmark into submap-local coordinates.
    pub fn to_local(&self, global: &Landmark) -> Landmark {
        let dx = global.x - self.origin.x;
        let dy = global.y - self.origin.y;
        let cos_t = self.origin.theta.cos();
        let sin_t = self.origin.theta.sin();
        Landmark {
            id: global.id,
            x: dx * cos_t + dy * sin_t,
            y: -dx * sin_t + dy * cos_t,
            descriptor: global.descriptor.clone(),
        }
    }

    /// Transform a local landmark into global coordinates.
    pub fn to_global(&self, local: &Landmark) -> Landmark {
        let cos_t = self.origin.theta.cos();
        let sin_t = self.origin.theta.sin();
        Landmark {
            id: local.id,
            x: self.origin.x + local.x * cos_t - local.y * sin_t,
            y: self.origin.y + local.x * sin_t + local.y * cos_t,
            descriptor: local.descriptor.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_creation() {
        let map = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        assert!(map.landmarks.is_empty());
    }

    #[test]
    fn test_add_landmarks() {
        let mut map = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        map.add_landmark(Landmark::new(1.0, 2.0, vec![]));
        map.add_landmark(Landmark::new(3.0, 4.0, vec![]));
        assert_eq!(map.landmarks.len(), 2);
    }

    #[test]
    fn test_nearest_landmark() {
        let mut map = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        map.add_landmark(Landmark::new(1.0, 0.0, vec![]));
        map.add_landmark(Landmark::new(5.0, 0.0, vec![]));
        let (idx, dist) = map.nearest_landmark(0.0, 0.0).unwrap();
        assert_eq!(idx, 0);
        assert!((dist - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_landmarks_within() {
        let mut map = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        map.add_landmark(Landmark::new(1.0, 0.0, vec![]));
        map.add_landmark(Landmark::new(5.0, 0.0, vec![]));
        map.add_landmark(Landmark::new(9.0, 0.0, vec![]));
        let within = map.landmarks_within(0.0, 0.0, 3.0);
        assert_eq!(within.len(), 1);
    }

    #[test]
    fn test_map_similarity() {
        let mut m1 = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        m1.add_landmark(Landmark::new(0.0, 0.0, vec![]));
        m1.add_landmark(Landmark::new(1.0, 0.0, vec![]));
        let mut m2 = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        m2.add_landmark(Landmark::new(0.1, 0.0, vec![]));
        m2.add_landmark(Landmark::new(1.1, 0.0, vec![]));
        let sim = m1.similarity(&m2, 0.5);
        assert!(sim > 0.5, "sim={}", sim);
    }

    #[test]
    fn test_map_merge() {
        let mut m1 = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        m1.add_landmark(Landmark::new(0.0, 0.0, vec![]));
        let mut m2 = Map2D::new((-10.0, -10.0, 10.0, 10.0));
        m2.add_landmark(Landmark::new(5.0, 5.0, vec![]));
        m2.add_landmark(Landmark::new(0.1, 0.1, vec![])); // Near m1's landmark
        m1.merge(&m2, 0.5);
        assert_eq!(m1.landmarks.len(), 2); // 1 original + 1 new (1 duplicate skipped)
    }

    #[test]
    fn test_submap_transform() {
        let submap = SubMap::new(Pose2D::new(5.0, 5.0, 0.0), (-10.0, -10.0, 10.0, 10.0));
        let global = Landmark::new(7.0, 8.0, vec![]);
        let local = submap.to_local(&global);
        assert!((local.x - 2.0).abs() < 1e-9);
        assert!((local.y - 3.0).abs() < 1e-9);
        let back = submap.to_global(&local);
        assert!((back.x - 7.0).abs() < 1e-9);
        assert!((back.y - 8.0).abs() < 1e-9);
    }
}
