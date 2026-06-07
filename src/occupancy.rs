//! Occupancy grid mapping.

/// Log-odds occupancy grid.
pub struct OccupancyGrid {
    pub width: usize,
    pub height: usize,
    pub resolution: f64,
    /// Log-odds values for each cell
    log_odds: Vec<f64>,
    /// Clamp limits
    pub log_min: f64,
    pub log_max: f64,
}

impl OccupancyGrid {
    /// Create a new occupancy grid (all unknown = 0.5 probability).
    pub fn new(width: usize, height: usize, resolution: f64) -> Self {
        Self {
            width,
            height,
            resolution,
            log_odds: vec![0.0; width * height],
            log_min: -5.0,
            log_max: 5.0,
        }
    }

    /// Get probability at cell.
    pub fn probability(&self, x: usize, y: usize) -> f64 {
        let lo = self.log_odds[y * self.width + x];
        1.0 - 1.0 / (1.0 + lo.exp())
    }

    /// Update cell with log-odds increment.
    pub fn update(&mut self, x: usize, y: usize, log_odds_update: f64) {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            self.log_odds[idx] = (self.log_odds[idx] + log_odds_update).clamp(self.log_min, self.log_max);
        }
    }

    /// Mark cell as occupied (positive log-odds update).
    pub fn mark_occupied(&mut self, x: usize, y: usize) {
        self.update(x, y, 0.7); // log-odds for occupied observation
    }

    /// Mark cell as free (negative log-odds update).
    pub fn mark_free(&mut self, x: usize, y: usize) {
        self.update(x, y, -0.7);
    }

    /// Check if cell is considered occupied.
    pub fn is_occupied(&self, x: usize, y: usize) -> bool {
        self.probability(x, y) > 0.5
    }

    /// Check if cell is considered free.
    pub fn is_free(&self, x: usize, y: usize) -> bool {
        self.probability(x, y) < 0.5
    }

    /// Convert world coordinates to grid cell.
    pub fn world_to_cell(&self, wx: f64, wy: f64) -> (usize, usize) {
        let cx = (wx / self.resolution) as usize;
        let cy = (wy / self.resolution) as usize;
        (cx.min(self.width - 1), cy.min(self.height - 1))
    }

    /// Convert grid cell to world coordinates (center of cell).
    pub fn cell_to_world(&self, cx: usize, cy: usize) -> (f64, f64) {
        ((cx as f64 + 0.5) * self.resolution, (cy as f64 + 0.5) * self.resolution)
    }

    /// Ray-cast from robot position to mark free cells along a beam, and the endpoint as occupied.
    pub fn update_ray(&mut self, rx: usize, ry: usize, ex: usize, ey: usize) {
        // Bresenham's line algorithm
        let dx = (ex as i64 - rx as i64).abs();
        let dy = (ey as i64 - ry as i64).abs();
        let sx: i64 = if ex > rx { 1 } else { -1 };
        let sy: i64 = if ey > ry { 1 } else { -1 };
        let mut err = dx - dy;
        let mut x = rx as i64;
        let mut y = ry as i64;

        loop {
            let ux = x as usize;
            let uy = y as usize;
            if ux >= self.width || uy >= self.height { break; }

            if ux == ex && uy == ey {
                self.mark_occupied(ux, uy);
                break;
            } else {
                self.mark_free(ux, uy);
            }

            let e2 = 2 * err;
            if e2 > -dy { err -= dy; x += sx; }
            if e2 < dx { err += dx; y += sy; }

            if x < 0 || y < 0 { break; }
        }
    }

    /// Reset the grid.
    pub fn reset(&mut self) {
        self.log_odds.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_creation() {
        let grid = OccupancyGrid::new(100, 100, 0.1);
        // All cells should be unknown (p=0.5)
        assert!((grid.probability(50, 50) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mark_occupied() {
        let mut grid = OccupancyGrid::new(100, 100, 0.1);
        grid.mark_occupied(50, 50);
        grid.mark_occupied(50, 50);
        grid.mark_occupied(50, 50);
        assert!(grid.is_occupied(50, 50));
    }

    #[test]
    fn test_mark_free() {
        let mut grid = OccupancyGrid::new(100, 100, 0.1);
        grid.mark_free(50, 50);
        grid.mark_free(50, 50);
        grid.mark_free(50, 50);
        assert!(grid.is_free(50, 50));
    }

    #[test]
    fn test_ray_update() {
        let mut grid = OccupancyGrid::new(100, 100, 0.1);
        grid.update_ray(10, 10, 30, 10);
        // Endpoint should be occupied
        assert!(grid.is_occupied(30, 10));
        // Cells along the way should be free
        assert!(grid.is_free(20, 10));
    }

    #[test]
    fn test_world_to_cell_conversion() {
        let grid = OccupancyGrid::new(100, 100, 0.5);
        let (cx, cy) = grid.world_to_cell(5.0, 7.0);
        assert_eq!(cx, 10);
        assert_eq!(cy, 14);
    }

    #[test]
    fn test_log_odds_clamping() {
        let mut grid = OccupancyGrid::new(10, 10, 0.1);
        for _ in 0..1000 {
            grid.mark_occupied(5, 5);
        }
        // Should be clamped
        assert!(grid.log_odds[5 * 10 + 5] <= grid.log_max);
    }

    #[test]
    fn test_reset() {
        let mut grid = OccupancyGrid::new(10, 10, 0.1);
        grid.mark_occupied(5, 5);
        grid.reset();
        assert!((grid.probability(5, 5) - 0.5).abs() < 1e-9);
    }
}
