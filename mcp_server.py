#!/usr/bin/env python3
"""
RAPIDS MCP Server (v1.8.1)
==========================
Model Context Protocol server for RAPIDS molecular simulation toolkit.

Exposes comprehensive tools for:
- Setting workspace directory for project isolation
- Listing available substrates, molecules, and rare molecules
- Building molecular simulation structures with full positioning control
- Running geometry optimization
- Calculating potential energies
- Batch screening multiple molecules

Usage:
    python mcp_server.py

Or configure in Claude Desktop config.json:
{
    "mcpServers": {
        "rapids": {
            "command": "python",
            "args": ["/path/to/mcp_server.py"]
        }
    }
}

IMPORTANT: Agents must call set_workspace() before running simulations.
This ensures project isolation - each agent's results are saved to their
own project directory, not the MCP server's directory.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, List, Dict

# MCP imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
)

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# RAPIDS imports (lazy loaded to speed up server start)
_simulation_builder = None
_smart_flow = None
_batch_comparison = None

def get_simulation_builder():
    """Lazy load SimulationBuilder"""
    global _simulation_builder
    if _simulation_builder is None:
        from simulation_builder import SimulationBuilder
        _simulation_builder = SimulationBuilder
    return _simulation_builder

def get_smart_flow():
    """Lazy load SmartFAIRChemFlow"""
    global _smart_flow
    if _smart_flow is None:
        from smart_fairchem_flow import SmartFAIRChemFlow
        _smart_flow = SmartFAIRChemFlow
    return _smart_flow

def get_batch_comparison():
    """Lazy load BatchComparison"""
    global _batch_comparison
    if _batch_comparison is None:
        from batch_comparison import BatchComparison
        _batch_comparison = BatchComparison
    return _batch_comparison

# Constants - MCP server's own directories (read-only resources)
MCP_SERVER_DIR = Path(__file__).parent
SUBSTRATE_DIR = MCP_SERVER_DIR / "substrate"
RARE_MOLECULES_DIR = MCP_SERVER_DIR / "rare_molecules"

# Workspace state - set by agent via set_workspace tool
_current_workspace: Optional[Path] = None


def get_workspace() -> Optional[Path]:
    """Get current workspace directory"""
    return _current_workspace


def get_simulations_dir() -> Optional[Path]:
    """Get simulations directory in current workspace"""
    if _current_workspace is None:
        return None
    return _current_workspace / "simulations"


def get_molecules_dir() -> Optional[Path]:
    """Get molecules directory in current workspace"""
    if _current_workspace is None:
        return None
    return _current_workspace / "molecules"


def require_workspace() -> tuple[bool, str]:
    """Check if workspace is set, return (ok, error_message)"""
    if _current_workspace is None:
        return False, (
            "Error: No workspace set.\n\n"
            "You must call set_workspace(path) first to specify where simulation "
            "files should be saved.\n\n"
            "Example: set_workspace('/path/to/your/project')\n\n"
            "This ensures your results are saved to your project directory, "
            "not the MCP server's directory."
        )
    return True, ""

# Available substrates (pre-built crystal structures)
AVAILABLE_SUBSTRATES = [
    "BP",        # Black Phosphorus
    "Co_HHTP",   # Cobalt-HHTP MOF
    "Cu_HHTP",   # Copper-HHTP MOF
    "Graphene",  # Graphene sheet
    "MoS2",      # Molybdenum disulfide
    "Ni_HHTP",   # Nickel-HHTP MOF
    "Si",        # Silicon
    "ZnO",       # Zinc Oxide
]

# Create server instance
server = Server(
    name="rapids",
    version="1.8.1",
    instructions="""RAPIDS (Rapid Adsorption Probe Interaction Discovery System) is a molecular simulation toolkit.

=== IMPORTANT: WORKSPACE SETUP ===

Before running ANY simulation, you MUST call set_workspace() with the user's project directory:

    set_workspace("/path/to/user/project")

This ensures:
- Your simulation results are saved to YOUR project directory
- The MCP server's directory remains clean (it only contains shared resources)
- Different projects don't pollute each other

The MCP server directory contains READ-ONLY shared resources:
- substrate/: Pre-built crystal structures (Graphene, MoS2, etc.)
- rare_molecules/: Complex molecules not available on PubChem

Your workspace will contain:
- simulations/: Your simulation results
- molecules/: Downloaded molecules for your project

=== TYPICAL WORKFLOW ===

1. set_workspace("/path/to/project")  ← MUST do this first!
2. build_simulation(probe="benzene", substrate="Graphene", ...)
3. optimize_structure(run_name="...")
4. calculate_adsorption_energy(run_name="...")

=== FEATURES ===

- Automatic molecule download from PubChem (just use the molecule name)
- 9 pre-built substrates: Graphene, MoS2, BP, Si, ZnO, Co/Cu/Ni_HHTP MOFs
- xTB implicit solvation (automatic with optimize_structure)
- Van der Waals contact distance mode
- Relative positioning between molecules

=== SOLVATION - IMPORTANT ===

TWO solvation methods are available:

1. xTB IMPLICIT SOLVATION (RECOMMENDED) - runs AUTOMATICALLY with optimize_structure()
   - Uses GFN2-xTB + ALPB water model
   - Fast (~seconds per molecule)
   - Gives "Solution binding" energy in solvation.json
   - NO configuration needed - just run optimize_structure()

2. EXPLICIT SOLVATION (NOT RECOMMENDED) - explicit_solvation parameter in build_simulation
   - Adds REAL water molecules to the simulation box
   - MUCH slower (many more atoms to optimize)
   - Only use for: MD simulations, specific solvent structure studies
   - DO NOT use for normal screening tasks!

For binding energy screening: Just use optimize_structure() → get Solution binding from results

=== TASK SELECTION GUIDE ===

task_name options and their accuracy:

1. 'omol' (default) - Best for VACUUM molecular interactions
   - Trained on ωB97M-V/def2-TZVPD with VV10 dispersion
   - ACCURATE: molecule-molecule interactions in vacuum (dimers, complexes)
   - INACCURATE: substrate adsorption (gives unrealistic -20 to -30 eV values)
   - Use for: hydrogen bonding, π-π stacking, drug-receptor interactions

2. 'oc20' - Better for SUBSTRATE systems (qualitative only)
   - Trained on RPBE for catalysis surfaces
   - Gives reasonable magnitude (~0.1 eV) but may have wrong sign
   - Use for: qualitative ranking of molecules on surfaces
   - NOT accurate for absolute adsorption energies

3. 'omat' - For inorganic materials
   - Trained on PBE/PBE+U for bulk materials
   - Use for: materials properties, not recommended for molecular adsorption

IMPORTANT: Use the SAME task_name across ALL tools in a workflow!

=== ACCURACY LIMITATIONS ===

VACUUM MODE (substrate='vacuum'):
✓ RELIABLE for molecule-molecule interactions
✓ Quantitative results comparable to literature
✓ Examples: water dimer (-5.0 kcal/mol), benzene dimer (-2.0 kcal/mol)

SUBSTRATE MODE (Graphene, MoS2, etc.):
✗ NOT RELIABLE for absolute adsorption energies
✗ omol: gives -20 to -30 eV (should be ~-0.5 eV) - 50x overestimate
✗ oc20: gives +0.1 eV (should be ~-0.5 eV) - wrong sign
- Root cause: UMA training data lacks 2D material adsorption
- USE ONLY FOR: qualitative ranking/comparison between molecules
- DO NOT TRUST: absolute energy values

=== RECOMMENDATIONS ===

For QUANTITATIVE analysis:
→ Use vacuum mode (probe-target without substrate)
→ Use omol task for best dispersion description

For QUALITATIVE substrate screening:
→ Use oc20 task (more reasonable magnitudes)
→ Trust relative rankings, not absolute values
→ Validate important results with DFT

=== WHEN TO USE scan_orientations ===

Initial molecular geometry can significantly affect results. Use scan_orientations when:

1. POSITIVE binding energy (> 0 eV): Likely bad initial geometry
   - Example: +6 eV → after scan → -0.3 eV (true binding)

2. VERY WEAK binding (-0.1 to 0 eV, or -2.3 kcal/mol): May be local minimum
   - Scan can find better configuration with stronger binding

3. UNEXPECTEDLY large values: If result seems chemically unreasonable
   - Trust your chemical intuition - if it looks wrong, scan it

4. IONIC compounds in vacuum: May have separation artifacts
   - Example: SDS (Na+ + sulfate anion) gives non-physical results in vacuum
   - Consider: use neutral analogs or solvated simulations instead

5. KEY candidates: For important molecules in your screening
   - Worth the extra computation to confirm the result

Rule of thumb: Quick screening first → scan suspicious results → confirm top candidates

PARALLEL EXECUTION SUPPORTED:
- Multiple optimize_structure calls can run in parallel
- For screening many molecules, you can call optimize_structure in parallel for efficiency

Typical workflow: build_simulation → optimize_structure → calculate_adsorption_energy
Optional: → scan_orientations (if result is suspicious)
""",
)


# ============================================================
# Tool Definitions
# ============================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools"""
    return [
        # ==================== Workspace Tools ====================
        Tool(
            name="set_workspace",
            description="Set the working directory for this session. MUST be called before running any simulations. "
                       "All simulation results and downloaded molecules will be saved to this directory. "
                       "This ensures project isolation - different agents/projects don't pollute each other.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to your project directory (e.g., '/Users/name/my_project')"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="get_workspace",
            description="Get the current workspace directory. Returns None if not set.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # ==================== Listing Tools ====================
        Tool(
            name="list_substrates",
            description="List all available substrate materials for molecular simulations. "
                       "Substrates are pre-built crystal structures that cannot be downloaded from PubChem.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_local_molecules",
            description="List molecules already downloaded in the local library. "
                       "Note: Any molecule can be used by name - it will be auto-downloaded from PubChem if not local.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_rare_molecules",
            description="List pre-optimized rare/complex molecules that are difficult to obtain from PubChem. "
                       "These include MOF linkers, complex ligands, and other specialized molecules.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_simulations",
            description="List all completed simulation runs with their status.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # ==================== Build Tool ====================
        Tool(
            name="build_simulation",
            description="Build molecular simulation structures with full control over positioning, orientation, and solvation. "
                       "Molecules are automatically downloaded from PubChem if not found locally. "
                       "Returns paths to generated structure files (VASP, XYZ formats).",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name for this simulation run (used for output folder)"
                    },
                    "probe": {
                        "type": "string",
                        "description": "Probe molecule name (e.g., 'benzene', 'glucose', 'ibuprofen'). Auto-downloaded from PubChem if not local."
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target molecule name for probe-target interaction studies."
                    },
                    "substrate": {
                        "type": "string",
                        "description": "Substrate material. Use 'vacuum' for gas-phase or one of: BP, Co_HHTP, Cu_HHTP, Graphene, MoS2, Ni_HHTP, Si, ZnO"
                    },
                    # Height parameters
                    "probe_height": {
                        "type": "number",
                        "description": "Height of probe above substrate in Angstroms (default: 2.5)"
                    },
                    "target_height": {
                        "type": "number",
                        "description": "Height of target above substrate in Angstroms (default: 6.0)"
                    },
                    "probe_target_distance": {
                        "type": ["number", "string"],
                        "description": "Distance between probe and target. Use number for fixed distance (Å), or 'contact' for van der Waals contact distance."
                    },
                    # Position parameters
                    "probe_position": {
                        "type": "object",
                        "description": "Custom probe position. Options: "
                                      "1) Cartesian: {\"x\": 10.0, \"y\": 10.0, \"z\": 5.0} "
                                      "2) Fractional: {\"frac\": [0.5, 0.5, 0.3]} "
                                      "3) Cylindrical: {\"cylindrical\": {\"r\": 5.0, \"theta\": 45, \"z\": 3.0}}"
                    },
                    "target_position": {
                        "type": "object",
                        "description": "Custom target position. Options: "
                                      "1) Cartesian: {\"x\": 10.0, \"y\": 10.0, \"z\": 8.0} "
                                      "2) Fractional: {\"frac\": [0.5, 0.5, 0.5]} "
                                      "3) Cylindrical: {\"cylindrical\": {\"r\": 5.0, \"theta\": 45, \"z\": 6.0}} "
                                      "4) Relative: {\"relative_to\": \"probe\", \"lateral_offset\": 4.0, \"vertical_offset\": 1.0, \"direction\": \"x\"}"
                    },
                    # Orientation parameters
                    "probe_orientation": {
                        "type": "object",
                        "description": "Probe molecule orientation. Options: "
                                      "1) Euler angles: {\"euler\": [0, 0, 45]} (degrees, ZYZ convention) "
                                      "2) Axis-angle: {\"axis\": [0, 0, 1], \"angle\": 45} "
                                      "3) Quaternion: {\"quaternion\": [1, 0, 0, 0]} "
                                      "4) Align vector: {\"align\": {\"from\": [0, 0, 1], \"to\": [1, 0, 0]}}"
                    },
                    "target_orientation": {
                        "type": "object",
                        "description": "Target molecule orientation. Same options as probe_orientation."
                    },
                    # Box and substrate parameters
                    "box_size": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Custom simulation box size [x, y, z] in Angstroms. Default: [30, 30, 40] for vacuum."
                    },
                    "fix_substrate_layers": {
                        "type": "integer",
                        "description": "Number of substrate layers to fix during optimization (default: 1). Set to 0 to allow all atoms to move."
                    },
                    # Explicit solvation parameters (NOT RECOMMENDED for most tasks)
                    "explicit_solvation": {
                        "type": "object",
                        "description": "⚠️ EXPLICIT SOLVATION - NOT RECOMMENDED for most tasks! "
                                      "This adds REAL water molecules to the simulation box, making calculations MUCH slower. "
                                      "For binding energy calculations, use optimize_structure() instead - it automatically "
                                      "runs xTB implicit solvation (ALPB) which is fast and gives Solution binding energies. "
                                      "Only use explicit solvation for: (1) MD simulations, (2) studying specific solvent effects. "
                                      "Examples if you really need it: "
                                      "1) Auto mode: {\"enabled\": true, \"mode\": \"auto\", \"shell_thickness\": 5.0} "
                                      "2) Manual mode: {\"enabled\": true, \"mode\": \"manual\", \"count\": 50}",
                        "properties": {
                            "enabled": {"type": "boolean", "description": "Enable explicit solvation (NOT RECOMMENDED)"},
                            "mode": {"type": "string", "enum": ["auto", "manual"], "description": "auto: calculate count from cluster size; manual: specify count"},
                            "solvent": {"type": "string", "description": "Solvent molecule name (default: water)"},
                            "count": {"type": "integer", "description": "Number of solvent molecules (manual mode)"},
                            "shell_thickness": {"type": "number", "description": "Solvation shell thickness in Å (default: 5.0)"}
                        }
                    }
                },
                "required": ["run_name", "probe"]
            }
        ),

        # ==================== Computation Tools ====================
        Tool(
            name="optimize_structure",
            description="Run geometry optimization on a simulation using FAIRChem UMA ML potential. "
                       "This may take several minutes depending on system size. "
                       "Returns optimized energies and structure paths.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name of existing simulation run to optimize"
                    },
                    "fmax": {
                        "type": "number",
                        "description": "Force convergence criterion in eV/Å (default: 0.05)"
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum optimization steps (default: 200)"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default, has VV10 dispersion), 'oc20' (catalysis), 'omat' (materials/surfaces). Must be consistent across all tools in workflow.",
                        "enum": ["omol", "oc20", "omat"]
                    }
                },
                "required": ["run_name"]
            }
        ),
        Tool(
            name="calculate_energy",
            description="Calculate potential energy of a structure using FAIRChem UMA ML potential. "
                       "Fast single-point energy calculation without optimization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "structure_path": {
                        "type": "string",
                        "description": "Path to structure file (VASP or XYZ format). Can be relative to RAPIDS directory."
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default, has VV10 dispersion), 'oc20' (catalysis), 'omat' (materials/surfaces)",
                        "enum": ["omol", "oc20", "omat"],
                        "default": "omol"
                    }
                },
                "required": ["structure_path"]
            }
        ),
        Tool(
            name="calculate_adsorption_energy",
            description="Calculate adsorption/interaction energies from optimized structures. "
                       "Computes: E_ads = E_complex - E_probe - E_target - E_substrate. "
                       "Negative values indicate favorable adsorption.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name of optimized simulation run"
                    }
                },
                "required": ["run_name"]
            }
        ),

        # ==================== Batch Screening Tool ====================
        Tool(
            name="batch_screening",
            description="Screen multiple probe molecules against a target and/or substrate. "
                       "Useful for high-throughput virtual screening of molecular interactions. "
                       "Returns comparative adsorption energies for all probes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name for this batch screening run"
                    },
                    "probes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of probe molecule names to screen (e.g., ['benzene', 'toluene', 'naphthalene'])"
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target molecule for probe-target interaction screening"
                    },
                    "substrate": {
                        "type": "string",
                        "description": "Substrate material (default: 'vacuum')"
                    },
                    "optimize": {
                        "type": "boolean",
                        "description": "Whether to run geometry optimization (default: true). Set to false for quick energy comparison."
                    },
                    "fmax": {
                        "type": "number",
                        "description": "Force convergence criterion in eV/Å (default: 0.05)"
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default, has VV10 dispersion), 'oc20' (catalysis), 'omat' (materials/surfaces). Must be consistent across all tools in workflow.",
                        "enum": ["omol", "oc20", "omat"]
                    }
                },
                "required": ["run_name", "probes"]
            }
        ),

        # ==================== Orientation Scanning Tool ====================
        Tool(
            name="scan_orientations",
            description="Scan multiple molecular orientations to find the most stable configuration. "
                       "Useful when initial geometry may not be optimal. Rotates probe (or target) "
                       "around specified axis and optimizes each configuration, returning the best one. "
                       "Helps avoid local minima in geometry optimization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Base name for this orientation scan"
                    },
                    "probe": {
                        "type": "string",
                        "description": "Probe molecule name"
                    },
                    "target": {
                        "type": "string",
                        "description": "Target molecule name (required for probe-target interaction)"
                    },
                    "substrate": {
                        "type": "string",
                        "description": "Substrate material (default: 'vacuum')"
                    },
                    "num_orientations": {
                        "type": "integer",
                        "description": "Number of orientations to sample (default: 6, evenly spaced)"
                    },
                    "rotate_molecule": {
                        "type": "string",
                        "description": "Which molecule to rotate: 'probe' (default) or 'target'",
                        "enum": ["probe", "target"]
                    },
                    "rotation_axis": {
                        "type": "string",
                        "description": "Axis to rotate around: 'x' (tilt, changes relative angle - RECOMMENDED for dimers), 'y' (tilt), 'z' (in-plane rotation), or 'all' (samples all axes). For molecular dimers like benzene-benzene, use 'x' to explore parallel vs T-shaped configurations.",
                        "enum": ["x", "y", "z", "all"]
                    },
                    "task_name": {
                        "type": "string",
                        "description": "FAIRChem task type: 'omol' (default), 'oc20', 'omat'",
                        "enum": ["omol", "oc20", "omat"]
                    },
                    "fmax": {
                        "type": "number",
                        "description": "Force convergence criterion in eV/Å (default: 0.05)"
                    }
                },
                "required": ["run_name", "probe", "target"]
            }
        ),

        # ==================== Results Tool ====================
        Tool(
            name="get_simulation_results",
            description="Get detailed results and summary from a completed simulation run, including energies, structures, and files.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_name": {
                        "type": "string",
                        "description": "Name of the simulation run"
                    }
                },
                "required": ["run_name"]
            }
        ),

        # ==================== Structure Analysis Tool ====================
        Tool(
            name="analyze_structure",
            description="Analyze a molecular structure - get atom count, formula, dimensions, and basic properties.",
            inputSchema={
                "type": "object",
                "properties": {
                    "structure_path": {
                        "type": "string",
                        "description": "Path to structure file (VASP, XYZ, or SDF format)"
                    }
                },
                "required": ["structure_path"]
            }
        ),
    ]


# ============================================================
# Tool Implementations
# ============================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls"""

    if name == "set_workspace":
        return await handle_set_workspace(arguments)

    elif name == "get_workspace":
        return await handle_get_workspace()

    elif name == "list_substrates":
        return await handle_list_substrates()

    elif name == "list_local_molecules":
        return await handle_list_local_molecules()

    elif name == "list_rare_molecules":
        return await handle_list_rare_molecules()

    elif name == "list_simulations":
        return await handle_list_simulations()

    elif name == "build_simulation":
        return await handle_build_simulation(arguments)

    elif name == "optimize_structure":
        return await handle_optimize_structure(arguments)

    elif name == "calculate_energy":
        return await handle_calculate_energy(arguments)

    elif name == "get_simulation_results":
        return await handle_get_simulation_results(arguments)

    elif name == "calculate_adsorption_energy":
        return await handle_calculate_adsorption_energy(arguments)

    elif name == "batch_screening":
        return await handle_batch_screening(arguments)

    elif name == "scan_orientations":
        return await handle_scan_orientations(arguments)

    elif name == "analyze_structure":
        return await handle_analyze_structure(arguments)

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ==================== Workspace Handlers ====================

async def handle_set_workspace(args: dict) -> list[TextContent]:
    """Set the workspace directory for this session"""
    global _current_workspace

    path = args.get("path", "").strip()
    if not path:
        return [TextContent(type="text", text="Error: path is required")]

    workspace_path = Path(path).resolve()

    # Create workspace directory if it doesn't exist
    try:
        workspace_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return [TextContent(type="text", text=f"Error creating workspace directory: {e}")]

    # Create subdirectories
    simulations_dir = workspace_path / "simulations"
    molecules_dir = workspace_path / "molecules"

    try:
        simulations_dir.mkdir(exist_ok=True)
        molecules_dir.mkdir(exist_ok=True)
    except Exception as e:
        return [TextContent(type="text", text=f"Error creating subdirectories: {e}")]

    _current_workspace = workspace_path

    result = f"Workspace set successfully!\n\n"
    result += f"Workspace: {workspace_path}\n"
    result += f"  └── simulations/  (simulation results will be saved here)\n"
    result += f"  └── molecules/    (downloaded molecules will be saved here)\n\n"
    result += "You can now run simulations. All results will be saved to this workspace."

    return [TextContent(type="text", text=result)]


async def handle_get_workspace() -> list[TextContent]:
    """Get the current workspace directory"""
    if _current_workspace is None:
        return [TextContent(type="text", text="No workspace set. Call set_workspace(path) first.")]

    result = f"Current workspace: {_current_workspace}\n"
    result += f"  └── simulations/: {get_simulations_dir()}\n"
    result += f"  └── molecules/:   {get_molecules_dir()}"

    return [TextContent(type="text", text=result)]


# ==================== Listing Handlers ====================

async def handle_list_substrates() -> list[TextContent]:
    """List available substrates with details"""
    substrate_info = []
    for sub in AVAILABLE_SUBSTRATES:
        sub_path = SUBSTRATE_DIR / sub
        if sub_path.exists():
            vasp_files = list(sub_path.glob("*.vasp"))
            info = f"- {sub}"
            if vasp_files:
                try:
                    from ase.io import read
                    atoms = read(vasp_files[0])
                    formula = atoms.get_chemical_formula()
                    info += f" ({len(atoms)} atoms, {formula})"
                except:
                    pass
            substrate_info.append(info)

    result = "Available Substrates (9 total):\n" + "\n".join(substrate_info)
    result += "\n\nUse 'vacuum' for gas-phase calculations without substrate."
    result += "\n\nSubstrate descriptions:"
    result += "\n- BP: Black Phosphorus (2D material)"
    result += "\n- Graphene: Graphene sheet (2D carbon)"
    result += "\n- MoS2: Molybdenum disulfide (2D semiconductor)"
    result += "\n- Co/Cu/Ni_HHTP: Metal-HHTP MOF structures"
    result += "\n- Si: Silicon surface"
    result += "\n- ZnO: Zinc Oxide surface"
    return [TextContent(type="text", text=result)]


async def handle_list_local_molecules() -> list[TextContent]:
    """List locally available molecules in current workspace and shared directories"""
    molecules_dir = get_molecules_dir()

    # Collect molecules from all sources
    workspace_molecules = set()
    shared_molecules = set()

    # 1. Workspace molecules
    if molecules_dir and molecules_dir.exists():
        for f in molecules_dir.glob("*.sdf"):
            if f.stem and not f.name.startswith('.'):
                workspace_molecules.add(f.stem)

    # 2. Global shared molecules (MCP_SERVER_DIR/molecules)
    global_molecules_dir = MCP_SERVER_DIR / "molecules"
    if global_molecules_dir.exists():
        for f in global_molecules_dir.glob("*.sdf"):
            if f.stem and not f.name.startswith('.'):
                shared_molecules.add(f.stem)

    # 3. Rare molecules (MCP_SERVER_DIR/rare_molecules)
    if RARE_MOLECULES_DIR.exists():
        for f in RARE_MOLECULES_DIR.glob("*.sdf"):
            if f.stem and not f.name.startswith('.'):
                shared_molecules.add(f.stem)

    # Combine and sort
    all_molecules = sorted(workspace_molecules | shared_molecules)

    result = f"Workspace: {_current_workspace or '(not set)'}\n\n"
    result += f"Local Molecules ({len(all_molecules)} available):\n"
    if all_molecules:
        result += ", ".join(all_molecules)
    else:
        result += "(none yet - molecules will be downloaded when you run simulations)"

    # Show breakdown if both sources have molecules
    if workspace_molecules and shared_molecules:
        result += f"\n\n  Workspace: {len(workspace_molecules)} | Shared: {len(shared_molecules)}"

    result += "\n\nNote: Any molecule name can be used - if not found locally, it will be auto-downloaded from PubChem."
    result += "\nExamples of downloadable molecules: ibuprofen, aspirin, caffeine, nicotine, morphine, penicillin, etc."
    return [TextContent(type="text", text=result)]


async def handle_list_rare_molecules() -> list[TextContent]:
    """List rare/complex molecules"""
    if not RARE_MOLECULES_DIR.exists():
        return [TextContent(type="text", text="No rare_molecules directory found. All molecules will be fetched from PubChem.")]

    molecules = []
    for f in RARE_MOLECULES_DIR.glob("*.sdf"):
        if f.stem and not f.name.startswith('.'):
            molecules.append(f.stem)
    for f in RARE_MOLECULES_DIR.glob("*.xyz"):
        if f.stem and not f.name.startswith('.'):
            molecules.append(f.stem)

    molecules = sorted(set(molecules))

    if not molecules:
        return [TextContent(type="text", text="No rare molecules found in rare_molecules directory.")]

    result = f"Rare/Complex Molecules ({len(molecules)} available):\n"
    result += ", ".join(molecules)
    result += "\n\nThese are pre-optimized 3D structures for molecules that may be difficult to obtain from PubChem."
    result += "\nThey are automatically used when you specify the molecule name."
    return [TextContent(type="text", text=result)]


async def handle_list_simulations() -> list[TextContent]:
    """List all simulation runs in current workspace"""
    simulations_dir = get_simulations_dir()

    if simulations_dir is None:
        return [TextContent(type="text", text="No workspace set. Call set_workspace(path) first.")]

    if not simulations_dir.exists():
        return [TextContent(type="text", text=f"Workspace: {_current_workspace}\n\nNo simulations found yet.")]

    simulations = []
    for sim_dir in sorted(simulations_dir.iterdir()):
        if sim_dir.is_dir() and not sim_dir.name.startswith('.'):
            status = "built"
            if (sim_dir / "results.json").exists():
                status = "optimized"
            elif (sim_dir / "summary.txt").exists():
                status = "built"
            else:
                status = "incomplete"

            # Get atom count if possible
            info = f"- {sim_dir.name} [{status}]"
            config_path = sim_dir / "config.json"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    probe = config.get("probe", "?")
                    target = config.get("target", "")
                    substrate = config.get("substrate", "vacuum")
                    info += f" (probe={probe}"
                    if target:
                        info += f", target={target}"
                    info += f", substrate={substrate})"
                except:
                    pass
            simulations.append(info)

    if not simulations:
        return [TextContent(type="text", text=f"Workspace: {_current_workspace}\n\nNo simulations found yet.")]

    result = f"Workspace: {_current_workspace}\n\n"
    result += f"Simulations ({len(simulations)} total):\n" + "\n".join(simulations)
    return [TextContent(type="text", text=result)]


# ==================== Build Handler ====================

async def handle_build_simulation(args: dict) -> list[TextContent]:
    """Build simulation structures with full configuration options"""
    try:
        # Check workspace
        ok, error_msg = require_workspace()
        if not ok:
            return [TextContent(type="text", text=error_msg)]

        simulations_dir = get_simulations_dir()

        # Prepare config with all possible parameters
        config = {
            "run_name": args["run_name"],
            "probe": args["probe"],
        }

        # Basic parameters
        if "target" in args and args["target"]:
            config["target"] = args["target"]

        config["substrate"] = args.get("substrate", "vacuum")

        # Height parameters
        if "probe_height" in args:
            config["probe_height"] = args["probe_height"]
        if "target_height" in args:
            config["target_height"] = args["target_height"]
        if "probe_target_distance" in args:
            config["probe_target_distance"] = args["probe_target_distance"]

        # Position parameters
        if "probe_position" in args:
            config["probe_position"] = args["probe_position"]
        if "target_position" in args:
            config["target_position"] = args["target_position"]

        # Orientation parameters
        if "probe_orientation" in args:
            config["probe_orientation"] = args["probe_orientation"]
        if "target_orientation" in args:
            config["target_orientation"] = args["target_orientation"]

        # Box and substrate parameters
        if "box_size" in args:
            config["box_size"] = args["box_size"]
        if "fix_substrate_layers" in args:
            config["fix_substrate_layers"] = args["fix_substrate_layers"]

        # Explicit solvation parameters (support both old and new parameter names)
        if "explicit_solvation" in args:
            config["solvation"] = args["explicit_solvation"]  # Map to internal 'solvation' key
        elif "solvation" in args:
            config["solvation"] = args["solvation"]  # Backward compatibility

        # Save config to temp file in workspace
        config_path = simulations_dir / f"{args['run_name']}_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Build simulation (redirect stdout to avoid MCP pollution)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            SimulationBuilder = get_simulation_builder()
            # Pass workspace to builder so it knows where to save molecules
            builder = SimulationBuilder(str(config_path), workspace=str(_current_workspace))
            structures = builder.build_simulation()
            builder.save_structures(structures)
        finally:
            sys.stdout = old_stdout

        # Clean up temp config
        config_path.unlink(missing_ok=True)

        # Prepare result
        output_dir = simulations_dir / args["run_name"]
        result = f"Simulation built successfully!\n\n"
        result += f"Output directory: {output_dir}\n\n"
        result += "Configuration:\n"
        result += f"  Probe: {args['probe']}\n"
        if "target" in args and args["target"]:
            result += f"  Target: {args['target']}\n"
        result += f"  Substrate: {config['substrate']}\n"
        # Check for explicit solvation (either parameter name)
        solvation_args = args.get("explicit_solvation") or args.get("solvation")
        if solvation_args and solvation_args.get("enabled"):
            result += f"  ⚠️ Explicit solvation: enabled ({solvation_args.get('mode', 'auto')} mode)\n"
            result += f"     Note: This adds real water molecules. For most tasks, xTB implicit solvation is sufficient.\n"

        result += "\nGenerated structures:\n"
        for name, atoms in structures.items():
            result += f"  - {name}: {len(atoms)} atoms ({atoms.get_chemical_formula()})\n"

        result += f"\nFiles saved in VASP and XYZ formats."
        result += f"\nUse optimize_structure(run_name='{args['run_name']}') to run geometry optimization."

        return [TextContent(type="text", text=result)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error building simulation: {str(e)}\n\n{traceback.format_exc()}")]


# ==================== Computation Handlers ====================

async def handle_optimize_structure(args: dict) -> list[TextContent]:
    """Run geometry optimization"""
    try:
        # Check workspace
        ok, error_msg = require_workspace()
        if not ok:
            return [TextContent(type="text", text=error_msg)]

        simulations_dir = get_simulations_dir()

        run_name = args["run_name"]
        fmax = args.get("fmax", 0.05)
        max_steps = args.get("max_steps", 200)
        task_name = args.get("task_name", "omol")

        sim_dir = simulations_dir / run_name
        config_path = sim_dir / "config.json"

        if not config_path.exists():
            return [TextContent(type="text", text=f"Simulation '{run_name}' not found. Run build_simulation first.")]

        with open(config_path) as f:
            config = json.load(f)

        config["fmax"] = fmax
        config["max_steps"] = max_steps
        config["task_name"] = task_name

        # Save updated config to temp file (SmartFlow expects file path)
        temp_config_path = sim_dir / "config_opt.json"
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        SmartFlow = get_smart_flow()
        flow = SmartFlow(str(temp_config_path), workspace=str(_current_workspace))

        result_text = f"Starting optimization for '{run_name}'...\n"
        result_text += f"Parameters: fmax={fmax} eV/Å, max_steps={max_steps}, task={task_name}\n\n"

        # Redirect stdout to stderr to avoid interfering with MCP stdio protocol
        # MCP uses stdout for JSON-RPC communication
        import io
        old_stdout = sys.stdout
        sys.stdout = sys.stderr  # Redirect prints to stderr
        try:
            await asyncio.to_thread(flow.run_workflow)
        finally:
            sys.stdout = old_stdout  # Restore stdout

        # Clean up
        del flow
        import gc
        gc.collect()

        result_text += "Optimization completed!\n\n"

        # Read results from saved files (run_workflow doesn't return results)
        interactions_path = sim_dir / "interactions.json"
        if interactions_path.exists():
            with open(interactions_path) as f:
                interactions = json.load(f)
            result_text += "Adsorption/Interaction Energies:\n"
            for name, energy in interactions.items():
                kcal = energy * 23.061
                result_text += f"  {name}: {energy:.4f} eV ({kcal:.2f} kcal/mol)\n"
                if energy < 0:
                    result_text += f"    → Favorable (exothermic)\n"
                else:
                    result_text += f"    → Unfavorable (endothermic)\n"

        # Read solvation results if available (vacuum mode only)
        solvation_path = sim_dir / "solvation.json"
        if solvation_path.exists():
            with open(solvation_path) as f:
                solvation = json.load(f)
            result_text += "\nSolvation Analysis (xTB GFN2-xTB + ALPB water):\n"
            for name in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]:
                if name in solvation:
                    G_solv = solvation[name]
                    result_text += f"  G_solv({name}): {G_solv:.4f} eV ({G_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solvation" in solvation:
                dG_solv = solvation["delta_G_solvation"]
                result_text += f"\n  ΔG_solvation: {dG_solv:.4f} eV ({dG_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solution" in solvation:
                dG_sol = solvation["delta_G_solution"]
                result_text += f"  Solution binding: {dG_sol:.4f} eV ({dG_sol * 23.061:.2f} kcal/mol)\n"

        # Also show the report if available
        report_path = sim_dir / "smart_report.txt"
        if report_path.exists():
            result_text += f"\nFull report saved to: {report_path}\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error during optimization: {str(e)}\n\n{traceback.format_exc()}")]


async def handle_calculate_energy(args: dict) -> list[TextContent]:
    """Calculate single-point energy"""
    try:
        from ase.io import read
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        structure_path = args["structure_path"]

        if not os.path.isabs(structure_path):
            structure_path = Path(__file__).parent / structure_path

        if not Path(structure_path).exists():
            return [TextContent(type="text", text=f"Structure file not found: {structure_path}")]

        # Redirect stdout to avoid MCP pollution from fairchem
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            atoms = read(structure_path)

            # Load model with specified task
            task_name = args.get("task_name", "omol")
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p1")
            calc = FAIRChemCalculator(predictor, task_name=task_name)
            atoms.calc = calc

            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            max_force = (forces**2).sum(axis=1).max()**0.5
        finally:
            sys.stdout = old_stdout

        result = f"Energy Calculation Results:\n"
        result += f"  Structure: {Path(structure_path).name}\n"
        result += f"  Task: {task_name}\n"
        result += f"  Atoms: {len(atoms)}\n"
        result += f"  Formula: {atoms.get_chemical_formula()}\n"
        result += f"  Potential Energy: {energy:.4f} eV\n"
        result += f"  Max Force: {max_force:.4f} eV/Å\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating energy: {str(e)}")]


async def handle_calculate_adsorption_energy(args: dict) -> list[TextContent]:
    """Calculate adsorption/interaction energies from optimized structures"""
    try:
        # Check workspace
        ok, error_msg = require_workspace()
        if not ok:
            return [TextContent(type="text", text=error_msg)]

        simulations_dir = get_simulations_dir()
        run_name = args["run_name"]
        sim_dir = simulations_dir / run_name

        if not sim_dir.exists():
            return [TextContent(type="text", text=f"Simulation '{run_name}' not found.")]

        # Check for interactions.json (output from SmartFAIRChemFlow)
        interactions_path = sim_dir / "interactions.json"
        smart_report_path = sim_dir / "smart_report.txt"

        result = f"Adsorption/Interaction Energy Analysis: {run_name}\n\n"

        if interactions_path.exists():
            with open(interactions_path) as f:
                interactions = json.load(f)

            result += "Adsorption/Interaction Energies:\n"
            for name, energy in interactions.items():
                kcal = energy * 23.061
                result += f"  {name}:\n"
                result += f"    {energy:.4f} eV\n"
                result += f"    {kcal:.2f} kcal/mol\n"
                if energy < 0:
                    result += f"    → Favorable (exothermic)\n"
                else:
                    result += f"    → Unfavorable/weak (endothermic)\n"
                result += "\n"
        elif smart_report_path.exists():
            # Fall back to parsing smart_report.txt
            with open(smart_report_path) as f:
                result += "From smart_report.txt:\n"
                result += f.read()
        else:
            return [TextContent(type="text", text=f"No results found for '{run_name}'. Run optimize_structure first.")]

        # Add solvation data if available
        solvation_path = sim_dir / "solvation.json"
        if solvation_path.exists():
            with open(solvation_path) as f:
                solvation = json.load(f)
            result += "\nSolvation Analysis (xTB GFN2-xTB + ALPB water):\n"
            for name in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]:
                if name in solvation:
                    G_solv = solvation[name]
                    result += f"  G_solv({name}): {G_solv:.4f} eV ({G_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solvation" in solvation:
                dG_solv = solvation["delta_G_solvation"]
                result += f"\n  ΔG_solvation: {dG_solv:.4f} eV ({dG_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solution" in solvation:
                dG_sol = solvation["delta_G_solution"]
                result += f"  Solution binding: {dG_sol:.4f} eV ({dG_sol * 23.061:.2f} kcal/mol)\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error calculating adsorption energy: {str(e)}")]


# ==================== Batch Screening Handler ====================

async def handle_batch_screening(args: dict) -> list[TextContent]:
    """Run batch screening of multiple probes"""
    try:
        # Check workspace
        ok, error_msg = require_workspace()
        if not ok:
            return [TextContent(type="text", text=error_msg)]

        simulations_dir = get_simulations_dir()

        run_name = args["run_name"]
        probes = args["probes"]
        target = args.get("target")
        substrate = args.get("substrate", "vacuum")
        optimize = args.get("optimize", True)
        fmax = args.get("fmax", 0.05)
        task_name = args.get("task_name", "omol")

        result_text = f"Batch Screening: {run_name}\n"
        result_text += f"Workspace: {_current_workspace}\n"
        result_text += f"Probes: {', '.join(probes)}\n"
        if target:
            result_text += f"Target: {target}\n"
        result_text += f"Substrate: {substrate}\n"
        result_text += f"Task: {task_name}\n"
        result_text += f"Optimize: {optimize}\n\n"

        all_results = {}

        for i, probe in enumerate(probes):
            result_text += f"[{i+1}/{len(probes)}] Processing {probe}...\n"

            sub_run_name = f"{run_name}_{probe}"
            config = {
                "run_name": sub_run_name,
                "probe": probe,
                "substrate": substrate,
                "task_name": task_name,
                "fmax": fmax,
            }
            if target:
                config["target"] = target

            # Build
            config_path = simulations_dir / f"{sub_run_name}_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Redirect stdout for entire build+optimize to avoid MCP pollution
            old_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                SimulationBuilder = get_simulation_builder()
                builder = SimulationBuilder(str(config_path), workspace=str(_current_workspace))
                structures = builder.build_simulation()
                builder.save_structures(structures)
                config_path.unlink(missing_ok=True)

                if optimize:
                    config["fmax"] = fmax
                    # Save config to temp file (SmartFlow expects file path)
                    opt_config_path = simulations_dir / f"{sub_run_name}" / "config_opt.json"
                    with open(opt_config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    SmartFlow = get_smart_flow()
                    flow = SmartFlow(str(opt_config_path), workspace=str(_current_workspace))
                    await asyncio.to_thread(flow.run_workflow)
                    # Clean up
                    del flow
                    import gc
                    gc.collect()
                    # Read results from saved file
                    interactions_path = simulations_dir / sub_run_name / "interactions.json"
                    if interactions_path.exists():
                        with open(interactions_path) as f:
                            all_results[probe] = json.load(f)
                    else:
                        all_results[probe] = {"status": "optimized but no interactions found"}
                    # Also read solvation data if available
                    solvation_path = simulations_dir / sub_run_name / "solvation.json"
                    if solvation_path.exists():
                        with open(solvation_path) as f:
                            all_results[probe]["solvation"] = json.load(f)
                else:
                    all_results[probe] = {"status": "built (not optimized)"}

            except Exception as e:
                all_results[probe] = {"error": str(e)}
            finally:
                sys.stdout = old_stdout  # Always restore stdout

        # Summary
        result_text += "\n" + "="*50 + "\n"
        result_text += "SCREENING RESULTS SUMMARY\n"
        result_text += "="*50 + "\n\n"

        # Sort by adsorption energy if available
        sorted_probes = []
        for probe, data in all_results.items():
            if isinstance(data, dict) and "error" not in data:
                # Get the most relevant adsorption energy
                if "probe_target" in data:
                    sorted_probes.append((probe, data["probe_target"], "probe_target"))
                elif "probe_substrate" in data:
                    sorted_probes.append((probe, data["probe_substrate"], "probe_substrate"))
                elif "probe_target_vacuum" in data:
                    sorted_probes.append((probe, data["probe_target_vacuum"], "energy"))
                else:
                    sorted_probes.append((probe, 0, "unknown"))
            else:
                sorted_probes.append((probe, float('inf'), "error"))

        sorted_probes.sort(key=lambda x: x[1])

        result_text += "Ranked by adsorption energy (most favorable/negative first):\n\n"
        for rank, (probe, energy, energy_type) in enumerate(sorted_probes, 1):
            if energy == float('inf'):
                result_text += f"{rank}. {probe}: ERROR - {all_results[probe].get('error', 'unknown')}\n"
            elif energy_type == "unknown":
                result_text += f"{rank}. {probe}: Built successfully\n"
            else:
                kcal = energy * 23.061
                line = f"{rank}. {probe}: {energy:.4f} eV ({kcal:.2f} kcal/mol)"
                # Add solution binding if available
                if "solvation" in all_results[probe] and "delta_G_solution" in all_results[probe]["solvation"]:
                    sol_energy = all_results[probe]["solvation"]["delta_G_solution"]
                    sol_kcal = sol_energy * 23.061
                    line += f" | Solution: {sol_energy:.4f} eV ({sol_kcal:.2f} kcal/mol)"
                result_text += line + "\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error in batch screening: {str(e)}\n\n{traceback.format_exc()}")]


# ==================== Orientation Scanning Handler ====================

async def handle_scan_orientations(args: dict) -> list[TextContent]:
    """Scan multiple orientations to find the most stable configuration"""
    try:
        # Check workspace
        ok, error_msg = require_workspace()
        if not ok:
            return [TextContent(type="text", text=error_msg)]

        simulations_dir = get_simulations_dir()

        import numpy as np

        run_name = args["run_name"]
        probe = args["probe"]
        target = args["target"]
        substrate = args.get("substrate", "vacuum")
        num_orientations = args.get("num_orientations", 6)
        rotate_molecule = args.get("rotate_molecule", "probe")
        rotation_axis = args.get("rotation_axis", "x")  # Default to x (tilt) for dimers
        task_name = args.get("task_name", "omol")
        fmax = args.get("fmax", 0.05)

        result_text = f"Orientation Scan: {run_name}\n"
        result_text += f"Workspace: {_current_workspace}\n"
        result_text += f"Probe: {probe}, Target: {target}\n"
        result_text += f"Rotating: {rotate_molecule} around {rotation_axis}-axis\n"
        result_text += f"Sampling {num_orientations} orientations\n"
        result_text += f"Task: {task_name}\n\n"

        # Generate rotation angles
        if rotation_axis == "all":
            # Sample multiple axes
            angles_list = []
            for axis in ["x", "y", "z"]:
                for angle in np.linspace(0, 300, num_orientations // 3 + 1)[:-1]:
                    angles_list.append((axis, float(angle)))
            if len(angles_list) < num_orientations:
                # Add more z rotations
                for angle in np.linspace(0, 300, num_orientations - len(angles_list) + 1)[1:]:
                    angles_list.append(("z", float(angle)))
        else:
            angles_list = [(rotation_axis, float(a)) for a in np.linspace(0, 360 - 360/num_orientations, num_orientations)]

        all_results = []

        # Redirect stdout for all operations
        old_stdout = sys.stdout
        sys.stdout = sys.stderr

        try:
            for i, (axis, angle) in enumerate(angles_list):
                sub_run_name = f"{run_name}_orient_{i}"
                result_text += f"[{i+1}/{len(angles_list)}] {axis}-axis, {angle:.1f}°... "

                # Build euler angles based on rotation axis
                if axis == "x":
                    euler = [angle, 0, 0]
                elif axis == "y":
                    euler = [0, angle, 0]
                else:  # z
                    euler = [0, 0, angle]

                # Prepare config
                config = {
                    "run_name": sub_run_name,
                    "probe": probe,
                    "target": target,
                    "substrate": substrate,
                    "task_name": task_name,
                    "fmax": fmax,
                }

                # Set orientation based on which molecule to rotate
                if rotate_molecule == "probe":
                    config["probe_orientation"] = {"euler": euler}
                else:
                    config["target_orientation"] = {"euler": euler}

                # Build simulation
                config_path = simulations_dir / f"{sub_run_name}_config.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

                try:
                    SimulationBuilder = get_simulation_builder()
                    builder = SimulationBuilder(str(config_path), workspace=str(_current_workspace))
                    structures = builder.build_simulation()
                    builder.save_structures(structures)
                    config_path.unlink(missing_ok=True)

                    # Optimize
                    opt_config_path = simulations_dir / sub_run_name / "config_opt.json"
                    with open(opt_config_path, 'w') as f:
                        json.dump(config, f, indent=2)

                    SmartFlow = get_smart_flow()
                    flow = SmartFlow(str(opt_config_path), workspace=str(_current_workspace))
                    await asyncio.to_thread(flow.run_workflow)

                    # Clean up
                    del flow
                    import gc
                    gc.collect()

                    # Read interaction energy
                    interactions_path = simulations_dir / sub_run_name / "interactions.json"
                    if interactions_path.exists():
                        with open(interactions_path) as f:
                            interactions = json.load(f)
                        # Get probe-target interaction energy
                        energy = interactions.get("probe_target_vacuum", interactions.get("probe_target", None))
                        if energy is not None:
                            kcal = energy * 23.061
                            result_entry = {
                                "run_name": sub_run_name,
                                "axis": axis,
                                "angle": angle,
                                "energy_eV": energy,
                                "energy_kcal": kcal
                            }
                            # Read solvation data if available
                            solvation_path = simulations_dir / sub_run_name / "solvation.json"
                            if solvation_path.exists():
                                with open(solvation_path) as f:
                                    solvation = json.load(f)
                                if "delta_G_solution" in solvation:
                                    result_entry["solution_eV"] = solvation["delta_G_solution"]
                                    result_entry["solution_kcal"] = solvation["delta_G_solution"] * 23.061
                            all_results.append(result_entry)
                            result_text += f"{energy:.4f} eV ({kcal:.2f} kcal/mol)\n"
                        else:
                            result_text += "No interaction energy found\n"
                    else:
                        result_text += "Optimization failed\n"

                except Exception as e:
                    result_text += f"Error: {str(e)}\n"

        finally:
            sys.stdout = old_stdout

        # Find best orientation
        if all_results:
            result_text += "\n" + "="*50 + "\n"
            result_text += "ORIENTATION SCAN RESULTS\n"
            result_text += "="*50 + "\n\n"

            # Sort by energy (most negative first)
            sorted_results = sorted(all_results, key=lambda x: x["energy_eV"])

            result_text += "Ranked by interaction energy (most favorable first):\n\n"
            for rank, r in enumerate(sorted_results, 1):
                marker = " ← BEST" if rank == 1 else ""
                line = f"{rank}. {r['axis']}-axis {r['angle']:.1f}°: {r['energy_eV']:.4f} eV ({r['energy_kcal']:.2f} kcal/mol)"
                if "solution_eV" in r:
                    line += f" | Sol: {r['solution_eV']:.4f} eV"
                line += marker
                result_text += line + "\n"

            best = sorted_results[0]
            worst = sorted_results[-1]
            spread = worst["energy_eV"] - best["energy_eV"]

            result_text += f"\nEnergy spread: {spread:.4f} eV ({spread * 23.061:.2f} kcal/mol)\n"
            result_text += f"Best configuration: {best['run_name']}\n"
            result_text += f"  Location: simulations/{best['run_name']}/\n"
        else:
            result_text += "\nNo successful optimizations completed.\n"

        return [TextContent(type="text", text=result_text)]

    except Exception as e:
        import traceback
        return [TextContent(type="text", text=f"Error in orientation scan: {str(e)}\n\n{traceback.format_exc()}")]


# ==================== Results Handler ====================

async def handle_get_simulation_results(args: dict) -> list[TextContent]:
    """Get simulation results"""
    try:
        # Check workspace
        ok, error_msg = require_workspace()
        if not ok:
            return [TextContent(type="text", text=error_msg)]

        simulations_dir = get_simulations_dir()
        run_name = args["run_name"]
        sim_dir = simulations_dir / run_name

        if not sim_dir.exists():
            return [TextContent(type="text", text=f"Simulation '{run_name}' not found.")]

        result = f"Workspace: {_current_workspace}\n"
        result += f"Simulation: {run_name}\n"
        result += f"Directory: {sim_dir}\n\n"

        # Read config
        config_path = sim_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            result += "Configuration:\n"
            result += f"  Probe: {config.get('probe', 'N/A')}\n"
            if config.get('target'):
                result += f"  Target: {config.get('target')}\n"
            result += f"  Substrate: {config.get('substrate', 'vacuum')}\n\n"

        # Read summary
        summary_path = sim_dir / "summary.txt"
        if summary_path.exists():
            result += "Summary:\n"
            result += summary_path.read_text()
            result += "\n"

        # Read results.json
        results_path = sim_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)

            if "energies" in results:
                result += "Energies (eV):\n"
                for name, energy in results["energies"].items():
                    result += f"  {name}: {energy:.4f}\n"

            if "binding_energies" in results:
                result += "\nAdsorption/Interaction Energies:\n"
                for name, energy in results["binding_energies"].items():
                    kcal = energy * 23.061
                    result += f"  {name}: {energy:.4f} eV ({kcal:.2f} kcal/mol)\n"

        # Read solvation data if available
        solvation_path = sim_dir / "solvation.json"
        if solvation_path.exists():
            with open(solvation_path) as f:
                solvation = json.load(f)
            result += "\nSolvation Analysis (xTB GFN2-xTB + ALPB water):\n"
            for name in ["probe_vacuum", "target_vacuum", "probe_target_vacuum"]:
                if name in solvation:
                    G_solv = solvation[name]
                    result += f"  G_solv({name}): {G_solv:.4f} eV ({G_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solvation" in solvation:
                dG_solv = solvation["delta_G_solvation"]
                result += f"\n  ΔG_solvation: {dG_solv:.4f} eV ({dG_solv * 23.061:.2f} kcal/mol)\n"
            if "delta_G_solution" in solvation:
                dG_sol = solvation["delta_G_solution"]
                result += f"  Solution binding: {dG_sol:.4f} eV ({dG_sol * 23.061:.2f} kcal/mol)\n"

        # List files
        result += "\nAvailable files:\n"
        for f in sorted(sim_dir.glob("*")):
            if f.is_file():
                size = f.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/1024/1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                result += f"  - {f.name} ({size_str})\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error getting results: {str(e)}")]


# ==================== Structure Analysis Handler ====================

async def handle_analyze_structure(args: dict) -> list[TextContent]:
    """Analyze a molecular structure"""
    try:
        from ase.io import read
        import numpy as np

        structure_path = args["structure_path"]

        if not os.path.isabs(structure_path):
            structure_path = Path(__file__).parent / structure_path

        if not Path(structure_path).exists():
            return [TextContent(type="text", text=f"Structure file not found: {structure_path}")]

        # Redirect stdout to avoid MCP pollution
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            atoms = read(structure_path)
        finally:
            sys.stdout = old_stdout

        # Basic info
        result = f"Structure Analysis: {Path(structure_path).name}\n\n"
        result += f"Atoms: {len(atoms)}\n"
        result += f"Formula: {atoms.get_chemical_formula()}\n"

        # Element composition
        symbols = atoms.get_chemical_symbols()
        unique_elements = sorted(set(symbols))
        result += f"Elements: {', '.join(unique_elements)}\n"
        result += "Composition:\n"
        for elem in unique_elements:
            count = symbols.count(elem)
            result += f"  {elem}: {count}\n"

        # Cell info
        cell = atoms.get_cell()
        if cell.any():
            result += f"\nCell dimensions:\n"
            result += f"  a = {cell[0, 0]:.2f} Å\n"
            result += f"  b = {cell[1, 1]:.2f} Å\n"
            result += f"  c = {cell[2, 2]:.2f} Å\n"
            result += f"  Volume = {atoms.get_volume():.2f} Å³\n"

        # Position range
        positions = atoms.get_positions()
        result += f"\nPosition range:\n"
        result += f"  x: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f} Å\n"
        result += f"  y: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f} Å\n"
        result += f"  z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f} Å\n"

        # Center of mass
        com = atoms.get_center_of_mass()
        result += f"\nCenter of mass: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) Å\n"

        return [TextContent(type="text", text=result)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error analyzing structure: {str(e)}")]


# ============================================================
# Resource Definitions
# ============================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources"""
    resources = []

    # Shared molecules from MCP server directory
    molecules_dir = MCP_SERVER_DIR / "molecules"
    if molecules_dir.exists():
        for f in sorted(molecules_dir.glob("*.sdf")):
            if not f.name.startswith('.'):
                resources.append(Resource(
                    uri=f"rapids://molecules/{f.stem}",
                    name=f"Molecule: {f.stem}",
                    description=f"Cached molecule structure",
                    mimeType="chemical/x-mdl-sdfile"
                ))

    # Rare molecules from MCP server directory
    if RARE_MOLECULES_DIR.exists():
        for f in sorted(RARE_MOLECULES_DIR.glob("*.sdf")):
            if not f.name.startswith('.'):
                resources.append(Resource(
                    uri=f"rapids://rare_molecules/{f.stem}",
                    name=f"Rare: {f.stem}",
                    description=f"Pre-optimized complex molecule",
                    mimeType="chemical/x-mdl-sdfile"
                ))

    # Simulations from current workspace
    simulations_dir = get_simulations_dir()
    if simulations_dir and simulations_dir.exists():
        for sim_dir in sorted(simulations_dir.iterdir()):
            if sim_dir.is_dir() and (sim_dir / "summary.txt").exists():
                resources.append(Resource(
                    uri=f"rapids://simulations/{sim_dir.name}",
                    name=f"Simulation: {sim_dir.name}",
                    description=f"Results from simulation run '{sim_dir.name}'",
                    mimeType="text/plain"
                ))

    return resources


@server.read_resource()
async def read_resource(uri) -> str:
    """Read a resource"""
    uri_str = str(uri)

    # Shared molecules
    if uri_str.startswith("rapids://molecules/"):
        mol_name = uri_str.replace("rapids://molecules/", "")
        mol_path = MCP_SERVER_DIR / "molecules" / f"{mol_name}.sdf"
        if mol_path.exists():
            return mol_path.read_text()
        return f"Molecule not found: {mol_name}"

    # Rare molecules
    if uri_str.startswith("rapids://rare_molecules/"):
        mol_name = uri_str.replace("rapids://rare_molecules/", "")
        mol_path = RARE_MOLECULES_DIR / f"{mol_name}.sdf"
        if mol_path.exists():
            return mol_path.read_text()
        return f"Rare molecule not found: {mol_name}"

    # Simulations (require workspace)
    if uri_str.startswith("rapids://simulations/"):
        simulations_dir = get_simulations_dir()
        if simulations_dir is None:
            return "No workspace set. Call set_workspace(path) first."

        run_name = uri_str.replace("rapids://simulations/", "")
        sim_dir = simulations_dir / run_name

        if sim_dir.exists():
            summary_path = sim_dir / "summary.txt"
            if summary_path.exists():
                return summary_path.read_text()

    return f"Resource not found: {uri_str}"


# ============================================================
# Main Entry Point
# ============================================================

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
