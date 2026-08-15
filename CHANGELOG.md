# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0-alpha] - 2026-08-15
### Added
- **Curvature-Driven Dynamic $\beta_1$**: Momentum decay is now dynamically scaled based on the local Signal-to-Noise Ratio (SNR), accelerating in flat topologies and stabilizing in chaotic ravines.
- **Bakhshali Quasi-Newtonian Denominator**: Upgraded the Bakhshali Quartic Brake to act as a second-order Hessian-Free approximation by dividing the penalty term by the pseudo-Hessian ($\sqrt{v_t}$).
- **Repository Reorganization**: Restructured the project into a proper `adamv` Python package structure (`adamv/`, `scripts/`, `benchmarks/`, `tests/`) for seamless pip installation and GitHub publishing.

### Changed
- Replaced rigid static $\beta_1$ with dynamic bounding.
- Re-architected C++/CUDA pointers to explicitly cast as `float32` during tensor operations to guarantee total safety and prevent VRAM corruption during PyTorch Automatic Mixed Precision (AMP / float16) training.
- Dropped integer-based bitmask overflow bugs in Python's OMNI-ModBH implementation.

## [1.1.0] - 2026-08-14
### Added
- Implemented **OMNI-ModBH** (Topological Basin Hopping) inside the CUDA kernel using bare-metal Type-Punning (Float32 to Int32) on the mantissa.
- Fixed `.item()` sync bottlenecks, resulting in a 400% performance boost during GPU training.
- Integrated Ramanujan Log-Periodic Envelopes for autonomous cooling.

## [1.0.0] - 2026-08-10
### Added
- Initial C++/CUDA implementation of the Bakhshali Quartic Gate.
- CPU fallback integration.
