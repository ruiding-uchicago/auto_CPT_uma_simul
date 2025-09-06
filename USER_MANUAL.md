# FAIRChem Auto-UMA Simulation Toolkit - User Manual

## Overview

This is an intelligent molecular simulation toolkit based on FAIRChem UMA models, designed for calculating intermolecular interactions and adsorption energies. The system uses machine learning-accelerated quantum chemistry calculations to rapidly evaluate molecular interaction strengths in vacuum or on 2D material surfaces.

## Core Features

### 1. Automatic Molecule Download
- **Local rare_molecules collection**: Checks pre-optimized rare/complex molecules first
- **Direct structure retrieval from chemical names**: Input common names like "glucose", "caffeine"
- **Automatic 3D structure generation**: Converts 2D to 3D using RDKit if PubChem only has 2D
- **Supported formats**: SDF, XYZ, VASP, and other molecular file formats

### 2. Intelligent Structure Building
- **Automatic overlap avoidance**: Smart algorithms ensure reasonable intermolecular spacing
- **Supported substrates**:
  - Graphene
  - MoS2 (Molybdenum disulfide)
  - BP (Black phosphorus)
  - Si (Silicon)
  - ZnO (Zinc oxide)
  - vacuum (no substrate)

### 3. FAIRChem Energy Calculations
- **Uses UMA-S-1p1 model**: Universal Materials Atomistic model developed by Meta AI
- **Calculation types**:
  - Molecular adsorption energy
  - Intermolecular interaction energy
  - Structure optimization

### 4. Smart Optimization Features
- **Automatic box sizing**: Calculates simulation box dimensions based on molecular size
- **Smart continuation**: Auto-continues when near convergence, stops when forces too large
- **Structure validation**: Detects abnormal bond lengths and large structural changes

## Quick Start

### Environment Setup

```bash
# Activate FAIRChem environment
conda activate uma

# Ensure required packages are installed
pip install ase pubchempy rdkit scipy
```

### Minimal Configuration Run

Use an example config:

```bash
python smart_fairchem_flow.py example_configs/tutorials/01_simplest.json
```

Or create your own minimal config:

```json
{
  "probe": "glucose",
  "substrate": "Graphene"
}
```

Run simulation:

```bash
python smart_fairchem_flow.py config.json
```

Only 2 parameters needed - everything else is automatic!

## Example Configurations

The `example_configs/` folder contains ready-to-use configurations organized by complexity:

- **tutorials/** - Step-by-step learning examples (start here!)
  - Includes `05_rare_molecules.json` showing beta-CD from local collection
- **screening/** - Batch comparison for multiple molecules
- **advanced/** - Custom positioning and orientation
- **applications/** - Real research scenarios (PFAS, MOFs, CNT, etc.)
  - `beta_cd_pfas_encapsulation.json` - Beta-cyclodextrin with PFAS
  - `cnt_molecule_screening.json` - Carbon nanotube interactions

See `example_configs/README.md` for detailed descriptions.

## Configuration Parameters

### Required Parameters
- `probe`: Probe molecule name (primary study object)
- `substrate`: Substrate material or "vacuum"

### Optional Parameters

#### Basic Parameters
- `target`: Target molecule (for probe-target interaction studies)
- `task_name`: FAIRChem task type, default "omat"
  - "omat": Open Catalyst 2020 (default, recommended)
  - "oc20": Open Catalyst 2020
- `box_size`: Simulation box dimensions, default "auto" for automatic calculation
- `probe_height`: Probe molecule height from substrate, default 2.5 Å (used with auto positioning)
- `target_height`: Target molecule height, default probe_height + 3.5 Å (used with auto positioning)
- `probe_target_distance`: Probe-target spacing, default 4.0 Å
- `fmax`: Optimization convergence criterion, default 0.05 eV/Å
- `max_steps`: Maximum optimization steps per run, default 100
- `max_continuations`: Maximum continuation attempts, default 3

#### Advanced Placement Parameters (NEW)
- `probe_position`: Custom probe placement, default "auto"
  - `"auto"`: Automatic centering above substrate
  - `[x, y, z]`: Absolute coordinates in Å
  - `{"x": 0.5, "y": 0.5, "z": 12.0}`: Fractional x,y (0-1), absolute z
  - `{"frac": [0.5, 0.5, 0.3]}`: All fractional coordinates
  - `{"cylindrical": {"r": 5.0, "theta": 45, "z": 12.0}}`: Cylindrical coordinates (useful for pores)
- `probe_orientation`: Custom probe orientation, default "auto"
  - `"auto"`: No rotation
  - `[rx, ry, rz]`: Euler angles in degrees
  - `{"axis": [x, y, z], "angle": degrees}`: Axis-angle rotation
  - `{"align": "flat"/"standing"/"tilted"}`: Preset orientations
- `target_position`: Custom target placement (same formats as probe_position)
- `target_orientation`: Custom target orientation (same formats as probe_orientation)

### Complete Configuration Example

```json
{
  "probe": "glucose",
  "target": "caffeine",
  "substrate": "Graphene",
  "task_name": "omat",
  "box_size": "auto",
  "probe_height": 2.5,
  "target_height": 6.0,
  "probe_target_distance": 4.0,
  "fmax": 0.05,
  "max_steps": 100,
  "max_continuations": 3,
  "output_dir": "simulations",
  "device": "cpu"
}
```

## Output Files

Generated in `simulations/[run_name]/` directory:

### Structure Files
- `probe_vacuum.vasp`: Probe molecule in vacuum
- `target_vacuum.vasp`: Target molecule in vacuum (if applicable)
- `substrate_only.vasp`: Pure substrate structure
- `probe_substrate.vasp`: Probe molecule on substrate
- `probe_target_vacuum.vasp`: Probe-target complex in vacuum
- `*_optimized.vasp`: Optimized structures (from smart_fairchem_flow.py)

### Optimization Logs
- `*_opt_0.log`: Optimization trajectories for each structure

### Analysis Results
- `interactions.json`: Calculated interaction energies
- `smart_report.txt`: Complete calculation report

## Results Interpretation

### Adsorption Energy
```
E_ads = E(probe_substrate) - E(probe_vacuum) - E(substrate_only)
```
- **Negative values**: Stable adsorption, more negative = more stable
- **Positive values**: Unstable, possibly local minimum

### Interaction Energy
```
E_int = E(probe_target) - E(probe) - E(target)
```
- **Negative values**: Attractive interaction
- **Positive values**: Repulsive interaction

## Smart Features

### 1. Automatic Box Sizing
- Analyzes molecular dimensions
- Adds 10 Å buffer zone
- Ensures minimum size 25×25×35 Å

### 2. Smart Continuation Logic
- Force < 2×target: Continue optimization
- Force > 0.5 eV/Å: Stop to avoid structural distortion
- Automatically determines if continuation worthwhile

### 3. Structure Validation
- RMSD > 3 Å: Warns of large structural changes
- Shortest bond < 0.8 Å: Warns of abnormally short bonds

## FAQ

### Q: Why is adsorption energy positive?
A: Possible reasons:
1. Local minimum rather than global minimum
2. Poor initial positioning
3. Weak substrate-molecule interaction

### Q: What if optimization doesn't converge?
A: System automatically continues up to 3 times. If still not converged:
1. Check if initial structure is reasonable
2. Adjust `fmax` convergence criterion
3. Increase `max_continuations`

### Q: How to use GPU acceleration?
A: Set in configuration file:
```json
{
  "device": "cuda"
}
```

## Command Line Tools

### Single Simulation
```bash
python smart_fairchem_flow.py config.json

# Example
python smart_fairchem_flow.py example_configs/tutorials/01_simplest.json
```

### Batch Comparison (Screening)
```bash
python batch_comparison.py batch_config.json

# Example - single substrate
python batch_comparison.py example_configs/screening/sugar_screening.json

# Example - multiple substrates (NEW!)
python batch_comparison.py example_configs/applications/pfas_mof_comprehensive.json
```

**Multiple Substrates Support (NEW):**
- Use `"substrates": ["Sub1", "Sub2", "Sub3"]` for multiple substrates
- Use `"substrate": "Sub1"` for single substrate (backward compatible)
- Automatically tests all probe-substrate combinations
- Generates rankings for each substrate separately

### Batch Geometry Optimization
```bash
# Optimize only atomic positions (default)
python batch_opt.py /path/to/folder

# Optimize atoms and cell parameters (full relaxation)
python batch_opt.py /path/to/folder --relax-cell

# Optimize atoms and cell for 2D materials (maintain x/y ratio)
python batch_opt.py /path/to/folder --relax-cell --iso-2d
```

**Features:**
- Searches for all `.vasp` files in target folder
- Skips already optimized files (containing "_opted" in filename)
- Shows real-time optimization progress
- Saves optimized structures with descriptive suffixes:
  - `*_opted.vasp`: Atoms-only optimization
  - `*_opted_cell.vasp`: Full cell relaxation
  - `*_opted_cell_iso2d.vasp`: Isotropic 2D cell relaxation
- Generates `optimization_summary.txt` with energies and forces
- Supports cell optimization for periodic systems
- Special isotropic mode for 2D materials maintains x/y aspect ratio

**Cell Relaxation Modes:**
- **Default**: Only atomic positions optimized, cell fixed
- **--relax-cell**: Full cell optimization with all degrees of freedom
- **--relax-cell --iso-2d**: For 2D materials, maintains x/y ratio while allowing isotropic scaling

### Download Molecules
```bash
python molecule_downloader.py
# Enter molecule name to download
```

### Build Structures
```bash
python simulation_builder.py config.json
# Only builds structures, doesn't run optimization
```

## Advanced Placement Examples

### MOF Pore Center Placement
```json
{
  "probe": "PFOS",
  "substrate": "Cu_HHTP",
  "probe_position": {"frac": [0.5, 0.5, 0.4]},
  "probe_orientation": {"align": "flat"}
}
```

### Multiple Position Screening
```json
{
  "probe": "PFHxS",
  "substrate": "Co_HHTP",
  "probe_position": {"cylindrical": {"r": 0, "theta": 0, "z": 15.0}},
  "probe_orientation": [0, 0, 45]
}
```

### Precise Orientation Control
```json
{
  "probe": "PFDoDA",
  "substrate": "Ni_HHTP",
  "probe_position": [15.0, 15.0, 12.0],
  "probe_orientation": {"axis": [1, 0, 0], "angle": 90}
}
```

## Rare Molecules Collection

The `rare_molecules/` folder contains pre-optimized 3D structures for complex molecules that are difficult to obtain from PubChem:

- **beta_cd.sdf** - Beta-cyclodextrin (optimized 3D structure)
- **CNT.sdf** - H-capped carbon nanotube

### How It Works

1. When you specify a molecule name, the system first checks `rare_molecules/`
2. If found, it uses the pre-optimized structure
3. If not found, it downloads from PubChem as usual

### Adding Your Own Rare Molecules

Simply place your optimized `.sdf` files in the `rare_molecules/` folder with descriptive names:
- Use lowercase with underscores: `molecule_name.sdf`
- The system will automatically find them when referenced

### Example Usage

```json
{
  "probe": "beta-CD",
  "target": "PFOS",
  "substrate": "Graphene"
}
```

In this example:
- `beta-CD` will be loaded from `rare_molecules/beta_cd.sdf`
- `PFOS` will be downloaded from PubChem
- Both work seamlessly together

## Advanced Usage

### Batch Calculations
Create multiple configuration files and run with script:

```bash
# Run all tutorials
for config in example_configs/tutorials/*.json; do
    python smart_fairchem_flow.py "$config"
done

# Or use batch_comparison.py for screening
python batch_comparison.py example_configs/screening/sugar_screening.json
```

### Custom Substrates
Add new materials in the `create_substrate()` method in `simulation_builder.py`.

### Model Selection
Modify `model_name` to use other FAIRChem models:
- uma-s-1p1 (default, most stable)
- equiformer_v2 (high accuracy)
- schnet (fast)

## Technical Details

### Dependencies
- **ASE**: Atomic Simulation Environment
- **FAIRChem**: ML-accelerated quantum chemistry
- **PubChemPy**: Molecular database interface
- **RDKit**: Molecular processing and 3D generation
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (for advanced rotations)

### Periodic Boundary Conditions
All structures use full PBC [True, True, True] for VASP/FAIRChem compatibility.

### Energy Reference States
Vacuum and substrate calculations use identical box sizes to ensure consistent energy reference states.

## Troubleshooting

### Error: Inconsistent PBC settings
Ensure all structures have PBC set to [True, True, True]

### Error: Molecule not found
1. Check chemical name spelling
2. Try using IUPAC name
3. Manually download SDF file

### Warning: Large structural changes
1. Check initial geometry
2. Reduce optimization step size
3. Use stricter convergence criteria

## Contact & Support

If encountering issues, check:
1. Configuration file format is correct
2. Environment is properly activated
3. Review warnings in smart_report.txt

## License

This tool is based on the FAIRChem open-source project and follows its license agreement.