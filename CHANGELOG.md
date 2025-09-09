# Changelog

All notable changes to RAPIDS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-01-09

### Added
- Interactive 3D structure visualization with `visualize_structures.py`
- Energy display in visualization (shows interaction energies for each structure)
- Version tracking system with `version.py`
- Color-coded energy values (green for favorable, red for unfavorable)
- Multi-structure toggle in single HTML viewer
- Master index page for all visualizations
- Version display in reports and visualizations

### Changed
- Enhanced smart_report.txt to include version information
- Improved visualization with spin control, labels, and background toggle

### Fixed
- Spin toggle now properly starts and stops rotation
- Surface rendering no longer accumulates when switching styles
- Improved state management in visualizer

## [1.0.0] - 2024-09-05

### Initial Release
- Core RAPIDS functionality
- Smart FAIRChem workflow with auto box sizing
- Three-component system calculations
- Batch comparison and screening
- Automatic molecule download from PubChem
- Support for 2D materials and MOFs
- Smart optimization with continuation
- Comprehensive documentation and examples