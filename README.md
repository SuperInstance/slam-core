# slam-core

Simultaneous Localization and Mapping (SLAM) primitives for robotics: occupancy grid mapping, particle filter localization, and landmark detection. Pure Rust, no external dependencies.

## Features

- **Occupancy Grid**: Log-odds mapping with ray casting (Bresenham)
- **Particle Filter**: Monte Carlo localization with low-variance resampling
- **Landmarks**: Detection from range-bearing scans, data association, landmark DB
- **Maps**: 2D map representation, merging, submap coordinate transforms
- **Poses**: SE(2) pose composition, inversion, odometry, point transforms

## License

MIT
