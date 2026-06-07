//! Landmark detection and data association.

/// A detected landmark with position and descriptor.
#[derive(Debug, Clone)]
pub struct Landmark {
    pub id: Option<usize>,
    pub x: f64,
    pub y: f64,
    /// Descriptor (feature vector) for matching
    pub descriptor: Vec<f64>,
}

impl Landmark {
    pub fn new(x: f64, y: f64, descriptor: Vec<f64>) -> Self {
        Self { id: None, x, y, descriptor }
    }

    /// Euclidean distance to another landmark.
    pub fn distance_to(&self, other: &Landmark) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Descriptor distance (Euclidean).
    pub fn descriptor_distance(&self, other: &Landmark) -> f64 {
        self.descriptor.iter().zip(other.descriptor.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Landmark database for data association.
#[derive(Default)]
pub struct LandmarkDB {
    pub landmarks: Vec<Landmark>,
    pub next_id: usize,
    /// Maximum distance for landmark association
    pub association_threshold: f64,
    /// Maximum descriptor distance for matching
    pub descriptor_threshold: f64,
}

impl LandmarkDB {
    pub fn new() -> Self {
        Self {
            landmarks: Vec::new(),
            next_id: 0,
            association_threshold: 1.0,
            descriptor_threshold: 2.0,
        }
    }

    /// Try to associate a detected landmark with existing ones.
    /// Returns the ID of the matched or newly created landmark.
    pub fn associate_or_create(&mut self, detected: &Landmark) -> usize {
        let mut best_id = None;
        let mut best_dist = f64::INFINITY;

        for lm in &self.landmarks {
            let pos_dist = lm.distance_to(detected);
            let desc_dist = lm.descriptor_distance(detected);
            if pos_dist < self.association_threshold
                && desc_dist < self.descriptor_threshold
                && desc_dist < best_dist
            {
                best_dist = desc_dist;
                best_id = lm.id;
            }
        }

        if let Some(id) = best_id {
            // Update existing landmark (simple average)
            if let Some(lm) = self.landmarks.iter_mut().find(|l| l.id == Some(id)) {
                let n = 2.0; // approximate count
                lm.x = (lm.x + detected.x) / n;
                lm.y = (lm.y + detected.y) / n;
            }
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            let mut new_lm = detected.clone();
            new_lm.id = Some(id);
            self.landmarks.push(new_lm);
            id
        }
    }

    /// Get landmark by ID.
    pub fn get(&self, id: usize) -> Option<&Landmark> {
        self.landmarks.iter().find(|l| l.id == Some(id))
    }

    /// Number of known landmarks.
    pub fn len(&self) -> usize {
        self.landmarks.len()
    }

    /// Check if the database is empty.
    pub fn is_empty(&self) -> bool {
        self.landmarks.is_empty()
    }
}

/// Detect landmarks from range-bearing measurements.
pub fn detect_landmarks_from_scan(
    robot_x: f64,
    robot_y: f64,
    robot_theta: f64,
    scan: &[(f64, f64)], // (range, bearing) pairs
    range_threshold: f64,
) -> Vec<Landmark> {
    let mut landmarks = Vec::new();
    for &(range, bearing) in scan {
        if range < range_threshold {
            let angle = robot_theta + bearing;
            let lx = robot_x + range * angle.cos();
            let ly = robot_y + range * angle.sin();
            // Simple descriptor: [range, bearing]
            landmarks.push(Landmark::new(lx, ly, vec![range, bearing]));
        }
    }
    landmarks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landmark_creation() {
        let lm = Landmark::new(1.0, 2.0, vec![1.0, 2.0, 3.0]);
        assert!((lm.x - 1.0).abs() < 1e-9);
        assert!(lm.id.is_none());
    }

    #[test]
    fn test_landmark_distance() {
        let a = Landmark::new(0.0, 0.0, vec![]);
        let b = Landmark::new(3.0, 4.0, vec![]);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_descriptor_distance() {
        let a = Landmark::new(0.0, 0.0, vec![1.0, 0.0]);
        let b = Landmark::new(0.0, 0.0, vec![0.0, 1.0]);
        assert!((a.descriptor_distance(&b) - std::f64::consts::SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn test_landmark_association() {
        let mut db = LandmarkDB::new();
        let lm1 = Landmark::new(1.0, 1.0, vec![1.0, 0.0]);
        let id1 = db.associate_or_create(&lm1);
        assert_eq!(id1, 0);

        // Same position, same descriptor → should associate
        let lm2 = Landmark::new(1.05, 0.95, vec![0.99, 0.01]);
        let id2 = db.associate_or_create(&lm2);
        assert_eq!(id1, id2);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_new_landmark_creation() {
        let mut db = LandmarkDB::new();
        let lm1 = Landmark::new(1.0, 1.0, vec![1.0, 0.0]);
        db.associate_or_create(&lm1);
        let lm2 = Landmark::new(10.0, 10.0, vec![0.0, 1.0]);
        let id2 = db.associate_or_create(&lm2);
        assert_eq!(id2, 1);
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn test_landmark_detection_from_scan() {
        let landmarks = detect_landmarks_from_scan(0.0, 0.0, 0.0, &[
            (5.0, 0.0), (3.0, 1.0), (10.0, 2.0),
        ], 8.0);
        assert_eq!(landmarks.len(), 2); // Only range < 8.0
    }

    #[test]
    fn test_landmark_position_accuracy() {
        let landmarks = detect_landmarks_from_scan(0.0, 0.0, 0.0, &[
            (5.0, 0.0),
        ], 10.0);
        assert!((landmarks[0].x - 5.0).abs() < 1e-9);
        assert!((landmarks[0].y - 0.0).abs() < 1e-9);
    }
}
