# Example Configurations

This folder contains example configuration files organized by complexity and use case.

## Running Examples

```bash
# Single simulation
python smart_fairchem_flow.py example_configs/tutorials/01_simplest.json

# Batch comparison (screening)
python batch_comparison.py example_configs/screening/sugar_screening.json
```

## Folder Structure

### 📚 tutorials/ (Use `smart_fairchem_flow.py`)
Step-by-step examples for beginners:
- `01_simplest.json` - Minimal 2-parameter config
- `02_with_height.json` - Controlling molecule height
- `03_two_molecules.json` - Probe + target molecules
- `04_vacuum_study.json` - No substrate (vacuum box)
- `05_rare_molecules.json` - Using beta-CD from rare_molecules
- `glucose_caffeine_minimal.json` - Classic biomolecule example
- `ethanol_acetone_vacuum.json` - Solvent interactions

### 🔬 screening/ (Use `batch_comparison.py`)
Batch comparison configs for screening multiple molecules:
- `sugar_screening.json` - Compare 5 sugars binding to caffeine
- `amino_acid_adsorption.json` - Test amino acids on MoS2
- `drug_screening_vacuum.json` - Drug encapsulation study

### 🎯 advanced/ (Use `smart_fairchem_flow.py`)
Examples with custom positioning and orientation:
- `custom_position.json` - Precise molecular placement
- `mof_pore_study.json` - MOF pore center positioning
- `test_*.json` - Various advanced placement tests

### 🏭 applications/ (Mixed usage)
Real-world research applications:
- `pfas_screening_basic.json` - PFAS environmental study (Use `batch_comparison.py`)
- `pfas_mof_comprehensive.json` - PFAS screening on ALL 3 MOFs (Use `batch_comparison.py`)
- `pfos_pore_positions.json` - PFOS in MOF pores (Use `smart_fairchem_flow.py`)
- `beta_cd_pfas_encapsulation.json` - Beta-cyclodextrin PFAS capture (Use `smart_fairchem_flow.py`)
- `cnt_molecule_screening.json` - Alkane interactions with CNT (Use `batch_comparison.py`)

## Quick Start Guide

### Beginner (2 lines)
```json
{
  "probe": "water",
  "substrate": "Graphene"
}
```

### Intermediate (add control)
```json
{
  "probe": "methane",
  "substrate": "MoS2",
  "probe_height": 4.0,
  "fmax": 0.03
}
```

### Advanced (precise placement)
```json
{
  "probe": "PFOS",
  "substrate": "Cu_HHTP",
  "probe_position": {"frac": [0.5, 0.5, 0.35]},
  "probe_orientation": {"align": "flat"}
}
```

## Parameter Reference

See USER_MANUAL.md for complete parameter documentation.