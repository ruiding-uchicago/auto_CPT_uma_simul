#!/usr/bin/env python3
"""
Batch Comparison Tool for Multiple Probe Screening
Compare binding energies of multiple probe molecules with the same target
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from smart_fairchem_flow import SmartFAIRChemFlow
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class BatchComparison:
    """Compare multiple probe molecules against the same target"""
    
    def __init__(self, config_file: str = None, config_dict: dict = None):
        """Initialize batch comparison"""
        
        # Load configuration
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_file or config_dict")
            
        # Validate configuration
        if "probes" not in self.config:
            # Backward compatibility: convert single probe to list
            if "probe" in self.config:
                self.config["probes"] = [self.config["probe"]]
                print("Note: Converting single probe to batch format")
            else:
                raise ValueError("Must specify 'probes' (list) or 'probe' (single)")
                
        self.probes = self.config["probes"]
        self.target = self.config.get("target", None)
        
        # Support both single substrate and multiple substrates
        if "substrates" in self.config:
            self.substrates = self.config["substrates"]
        elif "substrate" in self.config:
            self.substrates = [self.config["substrate"]]
        else:
            self.substrates = ["vacuum"]
        
        # Output directory for comparison results
        self.output_dir = Path(self.config.get("output_dir", "simulations"))
        self.comparison_name = self.config.get("comparison_name", self.generate_comparison_name())
        self.comparison_dir = self.output_dir / f"comparison_{self.comparison_name}"
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
        self.energies = {}
        self.rankings = {}
        
    def generate_comparison_name(self) -> str:
        """Generate descriptive name for comparison run"""
        target_str = self.target if self.target else "nosol"
        if len(self.substrates) == 1:
            substrate_str = self.substrates[0].lower().replace(" ", "")
        else:
            substrate_str = f"{len(self.substrates)}substrates"
        n_probes = len(self.probes)
        return f"{target_str}_{substrate_str}_{n_probes}probes"
        
    def run_single_probe(self, probe: str, substrate: str = None) -> Dict:
        """Run simulation for a single probe on a specific substrate"""
        
        if substrate is None:
            substrate = self.substrates[0] if self.substrates else "vacuum"
        
        print("\n" + "="*60)
        print(f"PROCESSING: {probe} on {substrate}")
        print("="*60)
        
        # Create configuration for this probe-substrate combination
        single_config = self.config.copy()
        single_config["probe"] = probe
        single_config["substrate"] = substrate
        single_config.pop("probes", None)  # Remove probes list
        single_config.pop("substrates", None)  # Remove substrates list
        single_config["run_name"] = f"{probe}_{self.target}_{substrate}"
        
        # Run FAIRChem workflow
        try:
            flow = SmartFAIRChemFlow(config_dict=single_config)
            flow.run_workflow()
            
            # Extract results
            result = {
                "probe": probe,
                "energies": flow.energies.copy(),
                "warnings": flow.warnings.copy(),
                "converged": True
            }
            
            # Calculate interaction energies
            if self.target:
                # Probe-Target interaction in vacuum
                if all(k in flow.energies for k in ["probe_target_vacuum", "probe_vacuum", "target_vacuum"]):
                    result["probe_target_interaction"] = (
                        flow.energies["probe_target_vacuum"] - 
                        flow.energies["probe_vacuum"] - 
                        flow.energies["target_vacuum"]
                    )
                    
            # Probe adsorption on substrate
            if substrate != "vacuum":
                if all(k in flow.energies for k in ["probe_substrate", "probe_vacuum", "substrate_only"]):
                    result["probe_adsorption"] = (
                        flow.energies["probe_substrate"] - 
                        flow.energies["probe_vacuum"] - 
                        flow.energies["substrate_only"]
                    )
                    
            return result
            
        except Exception as e:
            print(f"Error processing {probe}: {e}")
            return {
                "probe": probe,
                "energies": {},
                "warnings": [f"Failed: {str(e)}"],
                "converged": False
            }
            
    def run_comparison(self):
        """Run comparison for all probes on all substrates"""
        
        print("\n" + "="*60)
        print("BATCH COMPARISON WORKFLOW")
        print("="*60)
        print(f"Target: {self.target if self.target else 'None (self-interaction)'}")
        print(f"Substrates: {', '.join(self.substrates)}")
        print(f"Number of probes: {len(self.probes)}")
        print(f"Probes: {', '.join(self.probes)}")
        print(f"Total simulations: {len(self.probes) * len(self.substrates)}")
        
        # Process each probe-substrate combination
        sim_count = 0
        total_sims = len(self.probes) * len(self.substrates)
        
        for substrate in self.substrates:
            print(f"\n{'='*60}")
            print(f"SUBSTRATE: {substrate}")
            print(f"{'='*60}")
            
            for probe in self.probes:
                sim_count += 1
                print(f"\n[{sim_count}/{total_sims}] Processing {probe} on {substrate}...")
                result = self.run_single_probe(probe, substrate)
                
                # Store results with substrate as key
                if substrate not in self.results:
                    self.results[substrate] = {}
                self.results[substrate][probe] = result
            
        # Analyze and rank results
        self.analyze_results()
        
        # Generate reports
        self.generate_comparison_report()
        self.generate_ranking_plot()
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.comparison_dir}")
        
    def analyze_results(self):
        """Analyze and rank probe molecules for each substrate"""
        
        # Handle both old (flat) and new (nested) result structures
        if len(self.substrates) == 1 and self.substrates[0] in self.results:
            # New nested structure with single substrate
            flat_results = self.results[self.substrates[0]]
        elif any(isinstance(v, dict) and "energies" in v for v in self.results.values()):
            # Old flat structure (backward compatibility)
            flat_results = self.results
        else:
            # New nested structure with multiple substrates
            # Analyze per substrate
            self.rankings = {}
            for substrate in self.substrates:
                if substrate not in self.results:
                    continue
                    
                interaction_energies = {}
                adsorption_energies = {}
                
                for probe, result in self.results[substrate].items():
                    if "probe_target_interaction" in result:
                        interaction_energies[probe] = result["probe_target_interaction"]
                    if "probe_adsorption" in result:
                        adsorption_energies[probe] = result["probe_adsorption"]
                
                # Store rankings per substrate
                if substrate not in self.rankings:
                    self.rankings[substrate] = {}
                    
                if interaction_energies:
                    sorted_interactions = sorted(interaction_energies.items(), key=lambda x: x[1])
                    self.rankings[substrate]["interaction"] = sorted_interactions
                    
                if adsorption_energies:
                    sorted_adsorptions = sorted(adsorption_energies.items(), key=lambda x: x[1])
                    self.rankings[substrate]["adsorption"] = sorted_adsorptions
            return
            
        # Single substrate case (backward compatible)
        interaction_energies = {}
        adsorption_energies = {}
        
        for probe, result in flat_results.items():
            if "probe_target_interaction" in result:
                interaction_energies[probe] = result["probe_target_interaction"]
            if "probe_adsorption" in result:
                adsorption_energies[probe] = result["probe_adsorption"]
                
        # Rank by interaction energy (more negative = better)
        if interaction_energies:
            sorted_interactions = sorted(interaction_energies.items(), key=lambda x: x[1])
            self.rankings["interaction"] = sorted_interactions
            
        # Rank by adsorption energy (more negative = better)
        if adsorption_energies:
            sorted_adsorptions = sorted(adsorption_energies.items(), key=lambda x: x[1])
            self.rankings["adsorption"] = sorted_adsorptions
    
    def _write_result_details(self, f, result):
        """Helper method to write result details to file"""
        if result["converged"]:
            f.write("Status: ✓ Converged\n")
        else:
            f.write("Status: ⚠️ Not converged\n")
            
        if "probe_target_interaction" in result:
            f.write(f"Probe-Target interaction: {result['probe_target_interaction']:.4f} eV\n")
            
        if "probe_adsorption" in result:
            f.write(f"Adsorption energy: {result['probe_adsorption']:.4f} eV\n")
            
        if result["warnings"]:
            f.write("Warnings:\n")
            for warning in result["warnings"]:
                f.write(f"  - {warning}\n")
            
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        
        report_path = self.comparison_dir / "comparison_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BATCH COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Target molecule: {self.target if self.target else 'None'}\n")
            f.write(f"Substrates: {', '.join(self.substrates)}\n")
            f.write(f"Number of probes tested: {len(self.probes)}\n")
            f.write(f"Probes: {', '.join(self.probes)}\n")
            f.write(f"Total simulations: {len(self.probes) * len(self.substrates)}\n\n")
            
            # Interaction energy ranking
            if "interaction" in self.rankings and self.rankings["interaction"]:
                f.write("="*70 + "\n")
                f.write("PROBE-TARGET INTERACTION ENERGIES (Ranked)\n")
                f.write("-"*70 + "\n")
                f.write(f"{'Rank':<6} {'Probe':<20} {'Energy (eV)':<15} {'Relative (eV)':<15}\n")
                f.write("-"*70 + "\n")
                
                best_energy = self.rankings["interaction"][0][1]
                for i, (probe, energy) in enumerate(self.rankings["interaction"], 1):
                    relative = energy - best_energy
                    status = "✓ BEST" if i == 1 else ""
                    f.write(f"{i:<6} {probe:<20} {energy:>12.4f}    {relative:>12.4f}  {status}\n")
                    
                f.write("\n")
                
            # Adsorption energy ranking
            if "adsorption" in self.rankings and self.rankings["adsorption"]:
                f.write("="*70 + "\n")
                f.write("ADSORPTION ENERGIES ON SUBSTRATE (Ranked)\n")
                f.write("-"*70 + "\n")
                f.write(f"{'Rank':<6} {'Probe':<20} {'Energy (eV)':<15} {'Relative (eV)':<15}\n")
                f.write("-"*70 + "\n")
                
                best_energy = self.rankings["adsorption"][0][1]
                for i, (probe, energy) in enumerate(self.rankings["adsorption"], 1):
                    relative = energy - best_energy
                    status = "✓ BEST" if i == 1 else ""
                    f.write(f"{i:<6} {probe:<20} {energy:>12.4f}    {relative:>12.4f}  {status}\n")
                    
                f.write("\n")
                
            # Handle multiple substrates case
            if len(self.substrates) > 1:
                for substrate in self.substrates:
                    if substrate not in self.rankings:
                        continue
                    f.write(f"\n{'='*70}\n")
                    f.write(f"SUBSTRATE: {substrate}\n")
                    f.write(f"{'='*70}\n\n")
                    
                    # Rankings for this substrate
                    if "interaction" in self.rankings[substrate]:
                        f.write("Probe-Target Interaction Ranking:\n")
                        f.write("-"*40 + "\n")
                        for i, (probe, energy) in enumerate(self.rankings[substrate]["interaction"], 1):
                            f.write(f"{i}. {probe}: {energy:.4f} eV\n")
                        f.write("\n")
                    
                    if "adsorption" in self.rankings[substrate]:
                        f.write("Adsorption Energy Ranking:\n")
                        f.write("-"*40 + "\n")
                        for i, (probe, energy) in enumerate(self.rankings[substrate]["adsorption"], 1):
                            f.write(f"{i}. {probe}: {energy:.4f} eV\n")
                        f.write("\n")
            
            # Detailed results for each probe
            f.write("="*70 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*70 + "\n\n")
            
            # Handle different result structures
            if len(self.substrates) == 1 and self.substrates[0] in self.results:
                # Single substrate with nested structure
                substrate = self.substrates[0]
                for probe in self.probes:
                    if probe in self.results[substrate]:
                        result = self.results[substrate][probe]
                        f.write(f"\n>>> {probe} on {substrate}\n")
                        f.write("-"*40 + "\n")
                        self._write_result_details(f, result)
            elif len(self.substrates) > 1:
                # Multiple substrates
                for substrate in self.substrates:
                    if substrate not in self.results:
                        continue
                    for probe in self.probes:
                        if probe in self.results[substrate]:
                            result = self.results[substrate][probe]
                            f.write(f"\n>>> {probe} on {substrate}\n")
                            f.write("-"*40 + "\n")
                            self._write_result_details(f, result)
            else:
                # Old flat structure (backward compatibility)
                for probe in self.probes:
                    if probe in self.results:
                        result = self.results[probe]
                        f.write(f"\n>>> {probe}\n")
                        f.write("-"*40 + "\n")
                        self._write_result_details(f, result)
                        
            # Summary statistics
            f.write("\n" + "="*70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*70 + "\n")
            
            if "interaction" in self.rankings and len(self.rankings["interaction"]) > 1:
                energies = [e for _, e in self.rankings["interaction"]]
                f.write(f"Interaction energy range: {min(energies):.4f} to {max(energies):.4f} eV\n")
                f.write(f"Mean interaction energy: {np.mean(energies):.4f} eV\n")
                f.write(f"Std deviation: {np.std(energies):.4f} eV\n")
                
            if "adsorption" in self.rankings and len(self.rankings["adsorption"]) > 1:
                energies = [e for _, e in self.rankings["adsorption"]]
                f.write(f"Adsorption energy range: {min(energies):.4f} to {max(energies):.4f} eV\n")
                f.write(f"Mean adsorption energy: {np.mean(energies):.4f} eV\n")
                f.write(f"Std deviation: {np.std(energies):.4f} eV\n")
                
        print(f"Report saved: {report_path}")
        
        # Also save as JSON for programmatic access
        json_path = self.comparison_dir / "comparison_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "config": self.config,
                "results": self.results,
                "rankings": {
                    "interaction": self.rankings.get("interaction", []),
                    "adsorption": self.rankings.get("adsorption", [])
                }
            }, f, indent=2)
            
        print(f"JSON results saved: {json_path}")
        
    def generate_ranking_plot(self):
        """Generate visualization of energy rankings"""
        
        # Skip if no rankings
        if not self.rankings:
            return
            
        # Determine number of subplots needed
        n_plots = len([r for r in self.rankings.values() if r])
        if n_plots == 0:
            return
            
        fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
        if n_plots == 1:
            axes = [axes]
            
        plot_idx = 0
        
        # Plot interaction energies
        if "interaction" in self.rankings and self.rankings["interaction"]:
            ax = axes[plot_idx]
            probes = [p for p, _ in self.rankings["interaction"]]
            energies = [e for _, e in self.rankings["interaction"]]
            
            colors = ['green' if e < 0 else 'red' for e in energies]
            bars = ax.bar(range(len(probes)), energies, color=colors)
            
            ax.set_xticks(range(len(probes)))
            ax.set_xticklabels(probes, rotation=45, ha='right')
            ax.set_ylabel('Interaction Energy (eV)')
            ax.set_title(f'Probe-{self.target} Interaction Energies')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, energy in zip(bars, energies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{energy:.3f}', ha='center', 
                       va='bottom' if height >= 0 else 'top')
                       
            plot_idx += 1
            
        # Plot adsorption energies
        if "adsorption" in self.rankings and self.rankings["adsorption"] and plot_idx < len(axes):
            ax = axes[plot_idx]
            probes = [p for p, _ in self.rankings["adsorption"]]
            energies = [e for _, e in self.rankings["adsorption"]]
            
            colors = ['blue' if e < 0 else 'orange' for e in energies]
            bars = ax.bar(range(len(probes)), energies, color=colors)
            
            ax.set_xticks(range(len(probes)))
            ax.set_xticklabels(probes, rotation=45, ha='right')
            ax.set_ylabel('Adsorption Energy (eV)')
            substrate_name = self.substrates[0] if len(self.substrates) == 1 else "Substrate"
            ax.set_title(f'Adsorption on {substrate_name}')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, energy in zip(bars, energies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{energy:.3f}', ha='center',
                       va='bottom' if height >= 0 else 'top')
                       
        plt.tight_layout()
        plot_path = self.comparison_dir / "energy_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_path}")


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch comparison of multiple probe molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example configuration for batch comparison:
{
  "probes": ["glucose", "fructose", "sucrose", "maltose"],
  "target": "protein_binding_site",
  "substrate": "Graphene",
  "comparison_name": "sugar_screening"
}

This will test each probe molecule and rank them by binding affinity.
        """
    )
    
    parser.add_argument("config", help="JSON configuration file")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate comparison plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
        
    # Run batch comparison
    batch = BatchComparison(config_file=args.config)
    batch.run_comparison()


if __name__ == "__main__":
    main()