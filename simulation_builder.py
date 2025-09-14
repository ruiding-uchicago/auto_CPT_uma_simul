#!/usr/bin/env python3
"""
Intelligent Atomic Simulation Environment Builder
Automatically creates simulation environments from JSON configuration
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ASE imports
from ase import Atoms
from ase.io import read, write
from ase.build import molecule as ase_molecule
from ase.geometry import get_distances
from ase.constraints import FixAtoms
from ase.visualize import view
from scipy.spatial.transform import Rotation

# Import our molecule downloader
from molecule_downloader import MoleculeDownloader


class SimulationBuilder:
    """Build atomic simulation environments from configuration"""
    
    def __init__(self, config_file: str = None, config_dict: dict = None):
        """
        Initialize builder with configuration
        
        Args:
            config_file: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to file)
        """
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_file or config_dict")
            
        self.substrate_dir = Path("substrate")
        self.molecules_dir = Path("molecules")
        self.output_dir = Path(self.config.get("output_dir", "simulations"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize molecule downloader
        self.downloader = MoleculeDownloader(str(self.molecules_dir))
        
        # Store loaded structures
        self.substrate = None
        self.probe = None
        self.target = None
        
    def load_substrate(self) -> Optional[Atoms]:
        """Load substrate from CONTCAR file"""
        substrate_name = self.config.get("substrate", None)
        
        if not substrate_name or substrate_name.lower() == "vacuum":
            print("No substrate specified, using vacuum box")
            return None
            
        substrate_path = self.substrate_dir / substrate_name / "CONTCAR"
        
        if not substrate_path.exists():
            print(f"Warning: Substrate file not found: {substrate_path}")
            print("Available substrates:", [d.name for d in self.substrate_dir.iterdir() if d.is_dir()])
            return None
            
        try:
            substrate = read(substrate_path)
            print(f"Loaded substrate: {substrate_name} ({len(substrate)} atoms)")
            
            # Ensure proper cell and PBC
            if substrate.cell.rank < 3:
                print("Warning: Substrate cell not properly defined")
                
            # Fix bottom layers (optional)
            if self.config.get("fix_substrate_layers", 1) > 0:
                z_positions = substrate.positions[:, 2]
                z_min = z_positions.min()
                z_threshold = z_min + self.config.get("fix_substrate_layers", 1) * 2.5  # Approximate layer spacing
                mask = z_positions < z_threshold
                substrate.set_constraint(FixAtoms(mask=mask))
                print(f"Fixed {mask.sum()} substrate atoms")
                
            return substrate
            
        except Exception as e:
            print(f"Error loading substrate: {e}")
            return None
            
    def download_and_load_molecule(self, name: str) -> Optional[Atoms]:
        """Download molecule if needed and load it"""
        # Sanitize filename
        filename = name.replace(' ', '_').replace('/', '_')
        sdf_path = self.molecules_dir / f"{filename}.sdf"
        
        # Download if not exists
        if not sdf_path.exists():
            print(f"Downloading molecule: {name}")
            result = self.downloader.download_molecule(name, prefer_3d=True)
            if not result:
                print(f"Failed to download {name}")
                return None
                
        # Load molecule
        try:
            mol = read(str(sdf_path))
            print(f"Loaded molecule: {name} ({len(mol)} atoms)")
            return mol
        except Exception as e:
            print(f"Error loading molecule {name}: {e}")
            return None
            
    def calculate_overlap(self, atoms1: Atoms, atoms2: Atoms, 
                         cutoff: float = 1.5) -> bool:
        """
        Check if two structures overlap
        
        Args:
            atoms1, atoms2: Atomic structures
            cutoff: Minimum allowed distance (Angstrom)
            
        Returns:
            True if structures overlap
        """
        # Use cell and pbc from atoms2 (the existing structure)
        if atoms2.cell is not None and atoms2.cell.rank > 0:
            cell = atoms2.cell
            pbc = atoms2.pbc
        else:
            cell = None
            pbc = False
            
        dist_array = get_distances(atoms1.positions, atoms2.positions, 
                                  cell=cell, pbc=pbc)[1]
        
        min_dist = np.min(dist_array)
        return min_dist < cutoff
        
    def find_safe_position(self, mol_to_place: Atoms, existing: Atoms,
                          initial_pos: np.ndarray, 
                          min_dist: float = 1.5,  # Reduced from 2.0 to allow closer approach
                          max_attempts: int = 100) -> np.ndarray:
        """
        Find a safe position for molecule avoiding overlaps
        
        Args:
            mol_to_place: Molecule to position
            existing: Existing structure to avoid
            initial_pos: Initial desired position
            min_dist: Minimum separation distance
            max_attempts: Maximum positioning attempts
            
        Returns:
            Safe position vector
        """
        # Try initial position
        mol_test = mol_to_place.copy()
        mol_test.translate(initial_pos - mol_test.get_center_of_mass())
        
        if not self.calculate_overlap(mol_test, existing, min_dist):
            return initial_pos
            
        print(f"Initial position has overlap, searching for safe position...")
        
        # Try systematic displacement
        for attempt in range(max_attempts):
            # Try different strategies
            if attempt < 20:
                # Small random lateral displacement
                dx, dy = np.random.randn(2) * 2.0
                dz = 0
            elif attempt < 40:
                # Larger lateral displacement
                angle = 2 * np.pi * attempt / 20
                radius = 3.0 + (attempt - 20) * 0.2
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                dz = 0
            elif attempt < 60:
                # Vertical displacement
                dx, dy = 0, 0
                dz = (attempt - 40) * 0.5
            else:
                # Random 3D displacement
                dx, dy, dz = np.random.randn(3) * 3.0
                
            test_pos = initial_pos + np.array([dx, dy, dz])
            mol_test = mol_to_place.copy()
            mol_test.translate(test_pos - mol_test.get_center_of_mass())
            
            if not self.calculate_overlap(mol_test, existing, min_dist):
                print(f"Found safe position after {attempt + 1} attempts")
                return test_pos
                
        print("Warning: Could not find overlap-free position, using best effort")
        return initial_pos + np.array([0, 0, 3.0])  # Move up as last resort
    
    def parse_position(self, position_config, cell, substrate_top=0.0):
        """
        Parse position configuration to absolute coordinates
        
        Args:
            position_config: Can be:
                - "auto": Use automatic positioning
                - [x, y, z]: Absolute coordinates in Å
                - {"x": 0.5, "y": 0.5, "z": 12.0}: Fractional x,y (0-1), absolute z
                - {"frac": [0.5, 0.5, 0.3]}: All fractional coordinates
                - {"cylindrical": {"r": 5.0, "theta": 45, "z": 12.0}}: Cylindrical coords
            cell: Unit cell for fractional coordinates
            substrate_top: Z-position of substrate top (for relative positioning)
            
        Returns:
            np.array of absolute [x, y, z] coordinates or "auto"
        """
        if position_config is None or position_config == "auto":
            return "auto"
        
        # List format: absolute coordinates
        if isinstance(position_config, (list, tuple)) and len(position_config) == 3:
            return np.array(position_config)
        
        # Dictionary format: various coordinate systems
        if isinstance(position_config, dict):
            # Fractional x,y with absolute z
            if "x" in position_config and "y" in position_config and "z" in position_config:
                x = position_config["x"] * cell[0, 0] if position_config["x"] <= 1.0 else position_config["x"]
                y = position_config["y"] * cell[1, 1] if position_config["y"] <= 1.0 else position_config["y"]
                z = position_config["z"]
                # Handle relative z positioning
                if isinstance(z, dict) and "above_substrate" in z:
                    z = substrate_top + z["above_substrate"]
                return np.array([x, y, z])
            
            # Full fractional coordinates
            if "frac" in position_config:
                frac = np.array(position_config["frac"])
                return np.dot(frac, cell)
            
            # Cylindrical coordinates (useful for pores)
            if "cylindrical" in position_config:
                cyl = position_config["cylindrical"]
                r = cyl["r"]
                theta = np.radians(cyl["theta"])
                z = cyl["z"]
                center = cell.diagonal()[:2] / 2
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                return np.array([x, y, z])
        
        print(f"Warning: Unrecognized position format {position_config}, using auto")
        return "auto"
    
    def parse_orientation(self, orientation_config):
        """
        Parse orientation configuration to rotation matrix
        
        Args:
            orientation_config: Can be:
                - "auto" or None: No rotation
                - [rx, ry, rz]: Euler angles in degrees
                - {"axis": [x, y, z], "angle": degrees}: Axis-angle rotation
                - {"euler": [rx, ry, rz], "convention": "xyz"}: Explicit Euler angles
                - {"quaternion": [w, x, y, z]}: Quaternion rotation
                - {"align": "flat"/"standing"/"tilted"}: Preset orientations
                
        Returns:
            3x3 rotation matrix or "auto"
        """
        if orientation_config is None or orientation_config == "auto":
            return "auto"
        
        # List format: Euler angles in degrees
        if isinstance(orientation_config, (list, tuple)) and len(orientation_config) == 3:
            rotation = Rotation.from_euler('xyz', orientation_config, degrees=True)
            return rotation.as_matrix()
        
        # Dictionary format: various rotation specifications
        if isinstance(orientation_config, dict):
            # Axis-angle rotation
            if "axis" in orientation_config and "angle" in orientation_config:
                axis = np.array(orientation_config["axis"])
                axis = axis / np.linalg.norm(axis)  # Normalize
                angle = np.radians(orientation_config["angle"])
                rotation = Rotation.from_rotvec(axis * angle)
                return rotation.as_matrix()
            
            # Explicit Euler angles with convention
            if "euler" in orientation_config:
                angles = orientation_config["euler"]
                convention = orientation_config.get("convention", "xyz")
                rotation = Rotation.from_euler(convention, angles, degrees=True)
                return rotation.as_matrix()
            
            # Quaternion
            if "quaternion" in orientation_config:
                quat = orientation_config["quaternion"]
                rotation = Rotation.from_quat(quat)
                return rotation.as_matrix()
            
            # Preset orientations
            if "align" in orientation_config:
                align = orientation_config["align"]
                if align == "flat":
                    # Rotate to lie flat (90 degrees around x-axis)
                    rotation = Rotation.from_euler('x', 90, degrees=True)
                elif align == "standing":
                    # No rotation, molecule stands as is
                    rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
                elif align == "tilted":
                    # 45 degree tilt
                    rotation = Rotation.from_euler('x', 45, degrees=True)
                else:
                    print(f"Unknown alignment preset: {align}")
                    return "auto"
                return rotation.as_matrix()
        
        print(f"Warning: Unrecognized orientation format {orientation_config}, using auto")
        return "auto"
    
    def apply_transformation(self, molecule, position, orientation):
        """
        Apply position and orientation transformations to molecule
        
        Args:
            molecule: ASE Atoms object to transform
            position: Target position (np.array) or "auto"
            orientation: Rotation matrix or "auto"
            
        Returns:
            Transformed molecule (copy)
        """
        mol_copy = molecule.copy()
        
        # Apply orientation first (rotate around center of mass)
        if not isinstance(orientation, str) and orientation is not None:
            com = mol_copy.get_center_of_mass()
            # Translate to origin
            mol_copy.positions -= com
            # Apply rotation
            mol_copy.positions = mol_copy.positions @ orientation.T
            # Translate back
            mol_copy.positions += com
        
        # Apply position
        if not isinstance(position, str) and position is not None:
            current_com = mol_copy.get_center_of_mass()
            mol_copy.translate(position - current_com)
        
        return mol_copy
        
    def build_simulation(self) -> Dict[str, Atoms]:
        """
        Build all simulation structures according to configuration
        
        Returns:
            Dictionary of simulation structures
        """
        structures = {}
        
        # Load components
        print("\n" + "="*60)
        print("Loading components...")
        print("="*60)
        
        # Load substrate
        self.substrate = self.load_substrate()
        
        # Load molecules
        probe_name = self.config.get("probe")
        target_name = self.config.get("target")
        
        if probe_name:
            self.probe = self.download_and_load_molecule(probe_name)
        if target_name:
            self.target = self.download_and_load_molecule(target_name)
            
        # Get positioning parameters (backward compatible)
        probe_height = self.config.get("probe_height", 2.5)  # Height above substrate
        target_height = self.config.get("target_height", 6.0)  # Height above substrate
        probe_target_distance = self.config.get("probe_target_distance", 4.0)  # Vertical distance between probe and target
        
        # Get custom position/orientation if specified
        probe_position_config = self.config.get("probe_position", "auto")
        probe_orientation_config = self.config.get("probe_orientation", "auto")
        target_position_config = self.config.get("target_position", "auto")
        target_orientation_config = self.config.get("target_orientation", "auto")
        
        # Determine box size
        if self.substrate:
            cell = self.substrate.cell.copy()
            pbc = [True, True, True]  # Full PBC for VASP/FAIRChem compatibility
            # Add vacuum above
            cell[2, 2] = max(cell[2, 2], 30.0)  # Ensure enough vacuum
        else:
            # Create vacuum box
            box_size = self.config.get("box_size", [30.0, 30.0, 40.0])
            cell = np.diag(box_size)
            pbc = [True, True, True]  # Full PBC even for vacuum
            
        # Get substrate top
        if self.substrate:
            substrate_top = self.substrate.positions[:, 2].max()
        else:
            substrate_top = 0.0
            
        print(f"\nSubstrate top at z = {substrate_top:.2f} Å")
        
        # Build structures
        print("\n" + "="*60)
        print("Building structures...")
        print("="*60)
        
        # 1. Substrate only (if exists)
        if self.substrate:
            structures["substrate_only"] = self.substrate.copy()
            print("Created: substrate_only")
            
        # 2. Probe + Substrate
        if self.probe and self.substrate:
            system_probe_sub = self.substrate.copy()
            
            # Parse position and orientation
            probe_position = self.parse_position(probe_position_config, cell, substrate_top)
            probe_orientation = self.parse_orientation(probe_orientation_config)
            
            # Use automatic positioning if not specified
            if isinstance(probe_position, str) and probe_position == "auto":
                center_xy = cell.diagonal()[:2] / 2
                probe_pos = np.array([center_xy[0], center_xy[1], substrate_top + probe_height])
            else:
                probe_pos = probe_position
            
            # Apply transformations
            probe_mol = self.apply_transformation(self.probe, probe_pos, probe_orientation)
            
            # Check and adjust for overlaps (only if using auto position)
            if isinstance(probe_position, str) and probe_position == "auto":
                safe_pos = self.find_safe_position(probe_mol, system_probe_sub, probe_pos)
                probe_mol.translate(safe_pos - probe_pos)
            
            # Combine
            system_probe_sub.extend(probe_mol)
            system_probe_sub.cell = cell
            system_probe_sub.pbc = pbc
            structures["probe_substrate"] = system_probe_sub
            
            # Report what was used
            if not (isinstance(probe_position, str) and probe_position == "auto") or not (isinstance(probe_orientation, str) and probe_orientation == "auto"):
                print(f"Created: probe_substrate (custom placement)")
            else:
                print("Created: probe_substrate")
            
        # 3. Probe in vacuum (use same cell as substrate system for consistency)
        if self.probe:
            probe_vac = self.probe.copy()
            # Use the same cell dimensions as the substrate system
            probe_vac.cell = cell.copy()
            probe_vac.pbc = pbc
            # Center in the cell
            probe_vac.center()
            structures["probe_vacuum"] = probe_vac
            print("Created: probe_vacuum")
            
        # 4. Target in vacuum (use same cell as substrate system for consistency)
        if self.target:
            target_vac = self.target.copy()
            # Use the same cell dimensions as the substrate system
            target_vac.cell = cell.copy()
            target_vac.pbc = pbc
            # Center in the cell
            target_vac.center()
            structures["target_vacuum"] = target_vac
            print("Created: target_vacuum")
            
        # 5. Both probe and target in vacuum (for reference) - target above probe
        if self.probe and self.target:
            # Create vacuum box
            vacuum_box = Atoms(cell=cell, pbc=[True, True, True])  # Full PBC
            center_xy = cell.diagonal()[:2] / 2
            
            # Parse positions and orientations for vacuum setup
            probe_position_vac = self.parse_position(probe_position_config, cell, 0)
            probe_orientation_vac = self.parse_orientation(probe_orientation_config)
            target_position_vac = self.parse_position(target_position_config, cell, 0)
            target_orientation_vac = self.parse_orientation(target_orientation_config)
            
            # Position probe
            if isinstance(probe_position_vac, str) and probe_position_vac == "auto":
                probe_pos = np.array([center_xy[0], center_xy[1], cell[2,2]/2 - probe_target_distance/2])
            else:
                probe_pos = probe_position_vac
            
            probe_mol = self.apply_transformation(self.probe, probe_pos, probe_orientation_vac)
            vacuum_box.extend(probe_mol)
            
            # Position target
            if isinstance(target_position_vac, str) and target_position_vac == "auto":
                target_pos = np.array([center_xy[0], center_xy[1], cell[2,2]/2 + probe_target_distance/2])
            else:
                target_pos = target_position_vac
            
            target_mol = self.apply_transformation(self.target, target_pos, target_orientation_vac)
            
            # Check and fix overlap (only if using auto positions)
            if isinstance(probe_position_vac, str) and probe_position_vac == "auto" and isinstance(target_position_vac, str) and target_position_vac == "auto":
                safe_target_pos = self.find_safe_position(target_mol, vacuum_box, target_pos, min_dist=2.5)
                target_mol.translate(safe_target_pos - target_pos)
            
            vacuum_box.extend(target_mol)
            structures["probe_target_vacuum"] = vacuum_box
            
            # Report what was used
            custom_used = (not (isinstance(probe_position_vac, str) and probe_position_vac == "auto") or not (isinstance(probe_orientation_vac, str) and probe_orientation_vac == "auto") or 
                          not (isinstance(target_position_vac, str) and target_position_vac == "auto") or not (isinstance(target_orientation_vac, str) and target_orientation_vac == "auto"))
            if custom_used:
                print("Created: probe_target_vacuum (custom placement)")
            else:
                print("Created: probe_target_vacuum (probe + target in vacuum, target above probe)")
            
        # 6. Three-component system: Probe + Target + Substrate
        # This will be created after probe_substrate is optimized
        if self.probe and self.target and self.substrate:
            # Create the three-component system
            system_three = self.substrate.copy()
            
            # Parse probe position and orientation (same as used above)
            probe_position = self.parse_position(probe_position_config, cell, substrate_top)
            probe_orientation = self.parse_orientation(probe_orientation_config)
            
            # Add probe at its position
            if isinstance(probe_position, str) and probe_position == "auto":
                center_xy = cell.diagonal()[:2] / 2
                probe_pos = np.array([center_xy[0], center_xy[1], substrate_top + probe_height])
            else:
                probe_pos = probe_position
            
            probe_mol = self.apply_transformation(self.probe, probe_pos, probe_orientation)
            
            # Check and adjust for overlaps (only if using auto position)
            if isinstance(probe_position, str) and probe_position == "auto":
                safe_pos = self.find_safe_position(probe_mol, system_three, probe_pos)
                probe_mol.translate(safe_pos - probe_pos)
            
            system_three.extend(probe_mol)
            
            # Parse target orientation
            target_orientation = self.parse_orientation(target_orientation_config)
            
            # Now add target above the probe
            if target_position_config == "auto" or target_position_config is None:
                # Place target above probe
                target_pos = np.array([center_xy[0], center_xy[1], substrate_top + target_height])
            else:
                target_pos = self.parse_position(target_position_config, cell, substrate_top)
            
            target_mol = self.apply_transformation(self.target, target_pos, target_orientation)
            
            # Check for overlaps with the existing system
            if target_position_config == "auto" or target_position_config is None:
                safe_target_pos = self.find_safe_position(target_mol, system_three, target_pos, min_dist=2.5)
                target_mol.translate(safe_target_pos - target_pos)
            
            system_three.extend(target_mol)
            system_three.cell = cell
            system_three.pbc = pbc
            structures["probe_target_substrate"] = system_three
            print("Created: probe_target_substrate (three-component system)")
            
        return structures
        
    def save_structures(self, structures: Dict[str, Atoms]):
        """Save all structures to files"""
        print("\n" + "="*60)
        print("Saving structures...")
        print("="*60)
        
        # Create run directory
        run_name = self.config.get("run_name", "simulation")
        run_dir = self.output_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = run_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved configuration: {config_path}")
        
        # Save structures
        for name, atoms in structures.items():
            # Save in multiple formats
            vasp_path = run_dir / f"{name}.vasp"
            xyz_path = run_dir / f"{name}.xyz"
            
            write(vasp_path, atoms, format='vasp')
            write(xyz_path, atoms, format='xyz')
            
            print(f"Saved: {name} ({len(atoms)} atoms)")
            print(f"  - VASP format: {vasp_path}")
            print(f"  - XYZ format: {xyz_path}")
            
        # Generate summary
        summary_path = run_dir / "summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Simulation Builder Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Run name: {run_name}\n")
            f.write(f"Substrate: {self.config.get('substrate', 'vacuum')}\n")
            f.write(f"Probe: {self.config.get('probe', 'None')}\n")
            f.write(f"Target: {self.config.get('target', 'None')}\n")
            f.write(f"Probe height: {self.config.get('probe_height', 2.5)} Å\n")
            f.write(f"Target height: {self.config.get('target_height', 6.0)} Å\n")
            f.write(f"Probe-Target distance: {self.config.get('probe_target_distance', 4.0)} Å\n")
            f.write("\nGenerated structures:\n")
            for name, atoms in structures.items():
                f.write(f"  - {name}: {len(atoms)} atoms\n")
                
        print(f"\nSummary saved: {summary_path}")
        print(f"All files saved to: {run_dir}")
        
    def run(self):
        """Execute the complete workflow"""
        print("\n" + "="*60)
        print("SIMULATION BUILDER")
        print("="*60)
        
        # Build structures
        structures = self.build_simulation()
        
        if not structures:
            print("Error: No structures generated")
            return
            
        # Save structures
        self.save_structures(structures)
        
        print("\n" + "="*60)
        print("COMPLETED SUCCESSFULLY")
        print("="*60)
        

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build atomic simulation environments from JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON configuration:
{
  "run_name": "glucose_caffeine_graphene",
  "substrate": "Graphene",
  "molecule_a": "glucose",
  "molecule_b": "caffeine", 
  "height_a": 3.0,
  "height_b": 6.0,
  "distance_ab": 5.0,
  "fix_substrate_layers": 1,
  "output_dir": "simulations"
}

Available substrates: Graphene, MoS2, BP, Si, ZnO (or "vacuum" for no substrate)
        """
    )
    
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("--visualize", action="store_true", 
                       help="Visualize structures after creation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
        
    # Run builder
    builder = SimulationBuilder(config_file=args.config)
    builder.run()
    
    # Optional visualization
    if args.visualize:
        try:
            from ase.visualize import view
            run_name = builder.config.get("run_name", "simulation")
            full_system_path = builder.output_dir / run_name / "full_system.vasp"
            if full_system_path.exists():
                atoms = read(full_system_path)
                view(atoms)
        except:
            print("Visualization not available (requires ase-gui)")
            

if __name__ == "__main__":
    main()