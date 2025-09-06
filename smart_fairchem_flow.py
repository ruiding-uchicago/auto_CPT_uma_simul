#!/usr/bin/env python3
"""
Smart FAIRChem Flow with intelligent features:
- Auto box sizing
- Smart continuation
- Minimal input mode
- Structure validation
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import FixAtoms

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    FAIRCHEM_AVAILABLE = True
except ImportError:
    FAIRCHEM_AVAILABLE = False

from simulation_builder import SimulationBuilder
from molecule_downloader import MoleculeDownloader


class SmartFAIRChemFlow:
    """Enhanced FAIRChem workflow with intelligent features"""
    
    def __init__(self, config_file: str = None, config_dict: dict = None):
        """Initialize with smart defaults"""
        
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_file or config_dict")
        
        # Apply smart defaults
        self.apply_smart_defaults()
        
        # Setup
        self.run_name = self.config.get("run_name", self.generate_run_name())
        self.output_dir = Path(self.config.get("output_dir", "simulations")) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model settings
        self.model_name = self.config.get("model_name", "uma-s-1p1")
        self.device = self.config.get("device", "cpu")
        
        # Task selection - default to omat, allow user override
        self.task_name = self.config.get("task_name", "omat")
        
        # Optimization settings
        self.fmax = self.config.get("fmax", 0.05)
        self.max_steps = self.config.get("max_steps", 100)
        self.max_continuations = self.config.get("max_continuations", 3)
        
        # Initialize calculator
        self.calculator = None
        if FAIRCHEM_AVAILABLE:
            self.init_calculator()
            
        self.energies = {}
        self.warnings = []
        
    def apply_smart_defaults(self):
        """Apply intelligent defaults based on system"""
        
        # Essential parameters only
        if "probe" not in self.config:
            raise ValueError("Must specify 'probe' molecule")
            
        # Smart defaults for optional parameters
        if "probe_height" not in self.config:
            self.config["probe_height"] = 2.5  # Reasonable default
            
        if "target" in self.config and self.config["target"]:
            if "target_height" not in self.config:
                self.config["target_height"] = self.config["probe_height"] + 3.5
            if "probe_target_distance" not in self.config:
                self.config["probe_target_distance"] = 4.0
                
        # Auto box size will be calculated dynamically
        if "box_size" not in self.config:
            self.config["box_size"] = "auto"
            
    def generate_run_name(self):
        """Generate descriptive run name from config"""
        probe = self.config.get("probe", "mol")
        target = self.config.get("target", "")
        substrate = self.config.get("substrate", "vac")
        
        if target:
            return f"{probe}_{target}_{substrate}"
        return f"{probe}_{substrate}"
        
            
    def calculate_optimal_box_size(self, molecules: list) -> list:
        """Calculate optimal box size based on molecular extent"""
        
        if not molecules:
            return [30.0, 30.0, 40.0]  # Default
            
        # Get combined extent of all molecules
        all_positions = []
        for mol in molecules:
            if isinstance(mol, Atoms):
                all_positions.extend(mol.positions)
                
        if not all_positions:
            return [30.0, 30.0, 40.0]
            
        positions = np.array(all_positions)
        mol_min = positions.min(axis=0)
        mol_max = positions.max(axis=0)
        extent = mol_max - mol_min
        
        # Add padding (at least 10 Å on each side)
        padding = 10.0
        box_size = extent + 2 * padding
        
        # Ensure minimum size
        min_size = np.array([25.0, 25.0, 35.0])
        box_size = np.maximum(box_size, min_size)
        
        print(f"Auto-calculated box size: [{box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f}] Å")
        return box_size.tolist()
        
    def init_calculator(self):
        """Initialize FAIRChem calculator"""
        try:
            print(f"Initializing {self.model_name} on {self.device}...")
            predictor = pretrained_mlip.get_predict_unit(self.model_name, device=self.device)
            self.calculator = FAIRChemCalculator(predictor, task_name=self.task_name)
            print(f"✓ Calculator initialized (task: {self.task_name})")
        except Exception as e:
            print(f"Error initializing calculator: {e}")
            self.calculator = None
            
    def smart_optimize(self, atoms: Atoms, name: str, 
                      fix_substrate: bool = False) -> Tuple[Atoms, bool]:
        """
        Smart optimization with automatic continuation
        
        Returns:
            (optimized_atoms, converged_flag)
        """
        if not self.calculator:
            print(f"Calculator not available, skipping {name}")
            return atoms, False
            
        print(f"\n🔧 Optimizing {name}...")
        
        opt_atoms = atoms.copy()
        opt_atoms.calc = self.calculator
        
        # Get initial energy
        try:
            e_initial = opt_atoms.get_potential_energy()
            print(f"  Initial energy: {e_initial:.4f} eV")
        except Exception as e:
            print(f"  Error getting initial energy: {e}")
            return atoms, False
            
        converged = False
        total_steps = 0
        continuation = 0
        
        while continuation <= self.max_continuations and not converged:
            
            # Setup optimizer
            logfile = str(self.output_dir / f"{name}_opt_{continuation}.log")
            opt = LBFGS(opt_atoms, logfile=logfile)
            
            # Run optimization
            try:
                print(f"  Attempt {continuation+1}: Running up to {self.max_steps} steps...")
                opt.run(fmax=self.fmax, steps=self.max_steps)
                
                # Check convergence
                forces = opt_atoms.get_forces()
                max_force = np.max(np.abs(forces))
                total_steps += opt.nsteps
                
                if max_force <= self.fmax:
                    converged = True
                    print(f"  ✓ Converged! Max force: {max_force:.4f} eV/Å")
                else:
                    print(f"  Max force: {max_force:.4f} eV/Å (target: {self.fmax})")
                    
                    # Smart decision: continue or stop?
                    if max_force <= self.fmax * 2:  # Close to convergence
                        print(f"  📍 Close to convergence, continuing...")
                        continuation += 1
                    elif max_force > 0.5:  # Too far from convergence
                        print(f"  ⚠️ Forces too large ({max_force:.2f} eV/Å), structure may be unreasonable")
                        print(f"  Stopping optimization to avoid distorted structure")
                        self.warnings.append(f"{name}: Stopped due to large forces")
                        break
                    else:
                        print(f"  Continuing optimization...")
                        continuation += 1
                        
            except Exception as e:
                print(f"  Optimization error: {e}")
                break
                
        # Final energy
        try:
            e_final = opt_atoms.get_potential_energy()
            print(f"  Final energy: {e_final:.4f} eV")
            print(f"  Energy change: {e_final - e_initial:.4f} eV")
            print(f"  Total steps: {total_steps}")
            
            self.energies[name] = e_final
            
            # Save structure
            opt_path = self.output_dir / f"{name}_optimized.vasp"
            write(opt_path, opt_atoms, format='vasp')
            
            # Check for structural issues
            self.validate_structure(atoms, opt_atoms, name)
            
        except Exception as e:
            print(f"  Error in final processing: {e}")
            return atoms, False
            
        return opt_atoms, converged
        
    def validate_structure(self, initial: Atoms, final: Atoms, name: str):
        """Validate structural integrity"""
        
        # Check for large geometric changes
        if len(initial) == len(final):
            # Calculate RMSD for same atoms
            initial_pos = initial.positions
            final_pos = final.positions
            
            # Simple RMSD (could be improved with alignment)
            rmsd = np.sqrt(np.mean((final_pos - initial_pos)**2))
            
            if rmsd > 3.0:
                warning = f"⚠️ {name}: Large structural change (RMSD: {rmsd:.2f} Å)"
                print(f"  {warning}")
                self.warnings.append(warning)
                
        # Check for unreasonable bond lengths (simple check)
        distances = final.get_all_distances()
        min_dist = np.min(distances[distances > 0])
        
        if min_dist < 0.8:  # Å
            warning = f"⚠️ {name}: Unusually short distance detected ({min_dist:.2f} Å)"
            print(f"  {warning}")
            self.warnings.append(warning)
            
    def build_structures_with_auto_box(self) -> Dict[str, Atoms]:
        """Build structures with automatic box sizing"""
        
        print("\n" + "="*60)
        print("BUILDING STRUCTURES (with smart box sizing)")
        print("="*60)
        
        # If box_size is "auto", calculate it
        if self.config.get("box_size") == "auto":
            # First load molecules to determine size
            # Create a temporary builder just to access download_and_load_molecule
            temp_builder = SimulationBuilder(config_dict=self.config)
            
            molecules = []
            if self.config.get("probe"):
                mol = temp_builder.download_and_load_molecule(self.config["probe"])
                if mol:
                    molecules.append(mol)
            if self.config.get("target"):
                mol = temp_builder.download_and_load_molecule(self.config["target"])
                if mol:
                    molecules.append(mol)
                    
            # Calculate optimal box
            self.config["box_size"] = self.calculate_optimal_box_size(molecules)
            
        # Now build with optimized settings
        builder = SimulationBuilder(config_dict=self.config)
        structures = builder.build_simulation()
        builder.save_structures(structures)
        
        return structures
        
    def run_workflow(self):
        """Execute smart workflow"""
        
        print("\n" + "="*60)
        print("SMART FAIRCHEM WORKFLOW")
        print("="*60)
        print(f"Configuration: {self.run_name}")
        print(f"Model: {self.model_name}")
        print(f"Task: {self.task_name}")
        print(f"Smart continuation: Enabled (max {self.max_continuations} attempts)")
        
        # Build structures
        structures = self.build_structures_with_auto_box()
        
        if not self.calculator:
            print("\n⚠️ FAIRChem not available")
            return
            
        # Optimize with smart continuation
        print("\n" + "="*60)
        print("SMART OPTIMIZATION")
        print("="*60)
        
        optimized = {}
        convergence_status = {}
        
        # Optimize each structure
        for struct_name in ["substrate_only", "probe_vacuum", "target_vacuum", 
                           "probe_substrate", "probe_target_vacuum"]:
            if struct_name in structures:
                opt_struct, converged = self.smart_optimize(
                    structures[struct_name],
                    struct_name,
                    fix_substrate=(struct_name in ["substrate_only", "probe_substrate"])
                )
                optimized[struct_name] = opt_struct
                convergence_status[struct_name] = converged
        
        # Special handling for three-component system
        # Use optimized probe_substrate as starting point
        if (self.config.get("probe") and self.config.get("target") and self.config.get("substrate") 
            and "probe_substrate" in optimized and "target_vacuum" in structures and "substrate_only" in structures):
            print("\n" + "="*60)
            print("Building three-component system from optimized probe_substrate")
            print("="*60)
            
            # Start with optimized probe_substrate
            system_three = optimized["probe_substrate"].copy()
            
            # Get the current positions
            substrate_atoms = len(structures["substrate_only"])
            substrate_top = system_three.positions[:substrate_atoms, 2].max()
            
            # Add target above the optimized probe position
            probe_positions = system_three.positions[substrate_atoms:]
            probe_center = probe_positions.mean(axis=0)
            
            # Load target and position it above probe
            target_height = self.config.get("target_height", probe_center[2] + 4.0)
            target_position = np.array([probe_center[0], probe_center[1], target_height])
            
            # Add target molecule
            target_mol = structures["target_vacuum"].copy()
            target_mol.positions = target_mol.positions - target_mol.positions.mean(axis=0) + target_position
            
            # Check for overlaps and adjust if needed
            min_dist = 2.5
            for i in range(len(system_three)):
                for j in range(len(target_mol)):
                    dist = np.linalg.norm(system_three.positions[i] - target_mol.positions[j])
                    if dist < min_dist:
                        # Move target up slightly
                        target_mol.positions[:, 2] += (min_dist - dist) + 0.5
                        break
            
            system_three.extend(target_mol)
            
            # Now optimize the three-component system
            # Fix both substrate and probe (only optimize target position)
            substrate_and_probe_atoms = len(system_three) - len(structures["target_vacuum"])
            fix_indices = list(range(substrate_and_probe_atoms))
            system_three.set_constraint(FixAtoms(indices=fix_indices))
            
            opt_three, converged_three = self.smart_optimize(
                system_three,
                "probe_target_substrate",
                fix_substrate=False  # Already set constraints above
            )
            optimized["probe_target_substrate"] = opt_three
            convergence_status["probe_target_substrate"] = converged_three
                
        # Calculate interaction energies
        self.calculate_and_report_interactions()
        
        # Final report
        self.generate_smart_report(convergence_status)
        
    def calculate_and_report_interactions(self):
        """Calculate and report interaction energies"""
        
        print("\n" + "="*60)
        print("INTERACTION ANALYSIS")
        print("="*60)
        
        results = {}
        
        # Probe-target in vacuum
        if all(k in self.energies for k in ["probe_target_vacuum", "probe_vacuum", "target_vacuum"]):
            e_int = self.energies["probe_target_vacuum"] - self.energies["probe_vacuum"] - self.energies["target_vacuum"]
            results["probe_target_vacuum"] = e_int
            print(f"Probe-Target interaction (vacuum): {e_int:.4f} eV")
            
        # Probe on substrate
        if all(k in self.energies for k in ["probe_substrate", "probe_vacuum", "substrate_only"]):
            e_ads = self.energies["probe_substrate"] - self.energies["probe_vacuum"] - self.energies["substrate_only"]
            results["probe_adsorption"] = e_ads
            print(f"Probe adsorption energy: {e_ads:.4f} eV")
            
            if e_ads > 0.1:
                print("  📍 Note: Positive adsorption energy may indicate:")
                print("     - Local minimum (not global)")
                print("     - Constrained geometry")
        
        # Three-component system interactions
        if all(k in self.energies for k in ["probe_target_substrate", "probe_substrate", "target_vacuum"]):
            # Target binding to probe on substrate
            e_target_binding = self.energies["probe_target_substrate"] - self.energies["probe_substrate"] - self.energies["target_vacuum"]
            results["target_binding_to_adsorbed_probe"] = e_target_binding
            print(f"Target binding to adsorbed probe: {e_target_binding:.4f} eV")
            
            # Total interaction energy (all three components)
            if all(k in self.energies for k in ["substrate_only", "probe_vacuum"]):
                e_total = self.energies["probe_target_substrate"] - self.energies["substrate_only"] - self.energies["probe_vacuum"] - self.energies["target_vacuum"]
                results["total_three_component_interaction"] = e_total
                print(f"Total three-component interaction: {e_total:.4f} eV")
                
            # Compare with vacuum interaction
            if "probe_target_vacuum" in results:
                e_substrate_effect = e_target_binding - results["probe_target_vacuum"]
                results["substrate_effect_on_binding"] = e_substrate_effect
                print(f"Substrate effect on probe-target binding: {e_substrate_effect:.4f} eV")
                if e_substrate_effect < 0:
                    print("  → Substrate enhances probe-target binding")
                elif e_substrate_effect > 0:
                    print("  → Substrate weakens probe-target binding")
                
        # Save results
        with open(self.output_dir / "interactions.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
    def generate_smart_report(self, convergence_status: dict):
        """Generate comprehensive report"""
        
        report_path = self.output_dir / "smart_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("SMART WORKFLOW REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"System: {self.config.get('probe')} + {self.config.get('target', 'none')}\n")
            f.write(f"Substrate: {self.config.get('substrate', 'vacuum')}\n")
            f.write(f"Task: {self.task_name}\n")
            f.write(f"Box size: {self.config.get('box_size')}\n\n")
            
            f.write("CONVERGENCE STATUS:\n")
            f.write("-"*40 + "\n")
            for name, converged in convergence_status.items():
                status = "✓ Converged" if converged else "⚠️ Not fully converged"
                f.write(f"{name}: {status}\n")
                
            if self.warnings:
                f.write("\nWARNINGS:\n")
                f.write("-"*40 + "\n")
                for warning in self.warnings:
                    f.write(f"{warning}\n")
                    
            f.write("\nENERGIES (eV):\n")
            f.write("-"*40 + "\n")
            for name, energy in self.energies.items():
                f.write(f"{name}: {energy:.4f}\n")
                
        print(f"\n📊 Report saved: {report_path}")
        

def main():
    """Enhanced command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart FAIRChem workflow with intelligent features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Minimal configuration example:
{
  "probe": "glucose",
  "target": "caffeine",
  "substrate": "Graphene"
}

Everything else is automatically determined!
        """
    )
    
    parser.add_argument("config", help="JSON configuration file")
    args = parser.parse_args()
    
    # Run smart workflow
    flow = SmartFAIRChemFlow(config_file=args.config)
    flow.run_workflow()
    

if __name__ == "__main__":
    main()