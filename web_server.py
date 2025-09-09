#!/usr/bin/env python3
"""
Minimal web server for RAPIDS - Direct integration with existing Python scripts
No Django, just Flask serving the HTML and running simulations
"""

from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import json
import os
import tempfile
import threading
import time
from pathlib import Path
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global process tracking
active_processes = {}

@app.route('/')
def index():
    """Serve the web GUI HTML"""
    return send_from_directory('.', 'web_gui.html')

@app.route('/stream')
def stream_simulation():
    """Stream simulation output using Server-Sent Events"""
    config = json.loads(request.args.get('config', '{}'))
    
    def generate():
        # Create temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Run smart_fairchem_flow.py
            cmd = [sys.executable, 'smart_fairchem_flow.py', config_path]
            
            yield f"data: Running command: {' '.join(cmd)}\n\n"
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}  # Force unbuffered output
            )
            
            # Track process
            active_processes['current'] = process
            
            # Stream output line by line with enhanced progress tracking
            step_count = 0
            start_time = time.time()
            last_output_time = time.time()
            
            # Add initial status
            yield f"data: 🚀 Simulation started, waiting for output...\n\n"
            
            while True:
                line = process.stdout.readline()
                
                # Check if process is still running
                if line == '' and process.poll() is not None:
                    break
                    
                if line:
                    line = line.strip()
                    
                    # Send each line as SSE
                    yield f"data: {line}\n\n"
                    last_output_time = time.time()
                else:
                    # No output, but check if we should send a heartbeat
                    current_time = time.time()
                    if current_time - last_output_time > 5:  # Send heartbeat every 5 seconds
                        elapsed = int(current_time - start_time)
                        yield f"data: ⏳ Still running... ({elapsed}s elapsed)\n\n"
                        last_output_time = current_time
                    time.sleep(0.1)  # Small delay to avoid busy waiting
                    
                    # Enhanced progress tracking
                    if 'Downloading' in line:
                        yield f"data: 📥 {line}\n\n"
                    elif 'Creating' in line or 'Setting up' in line:
                        yield f"data: 🔧 {line}\n\n"
                    elif 'Step' in line and 'Energy' in line:
                        step_count += 1
                        # Parse step number and energy
                        if step_count % 10 == 0:  # Report every 10 steps
                            elapsed = time.time() - start_time
                            yield f"data: ⚡ Progress: Step {step_count}, Time: {elapsed:.1f}s\n\n"
                            
                            # Estimate progress percentage
                            if config.get('max_steps'):
                                percent = min(100, int((step_count / config['max_steps']) * 100))
                                yield f"data: PROGRESS:{percent}\n\n"
                    elif 'Converged' in line:
                        yield f"data: ✅ {line}\n\n"
                    elif 'Error' in line or 'error' in line:
                        yield f"data: ❌ {line}\n\n"
                    elif 'WARNING' in line or 'Warning' in line:
                        yield f"data: ⚠️ {line}\n\n"
                    elif 'Final' in line or 'Complete' in line:
                        yield f"data: 🎉 {line}\n\n"
            
            process.wait()
            
            # Find the actual simulation directory
            # The script creates directories based on probe_target_substrate naming
            probe = config.get('probe', 'unknown')
            target = config.get('target', '')
            substrate = config.get('substrate', 'vacuum')
            
            # Try different possible directory names
            if target:
                sim_name = f"{probe}_{target}_{substrate}"
            else:
                sim_name = f"{probe}_{substrate}"
                
            sim_dir = Path('simulations') / sim_name
            
            # Check if directory exists
            if sim_dir.exists():
                yield f"data: 📁 Found simulation directory: {sim_name}\n\n"
                
                # Read results and parse report
                results = {}
                results_file = sim_dir / 'interactions.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        interactions = json.load(f)
                        results.update(interactions)
                
                # Parse the smart_report.txt for more details
                report_file = sim_dir / 'smart_report.txt'
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        report_text = f.read()
                        
                        # Extract values from report using regex
                        import re
                        
                        # Find total optimization steps (sum all "Total steps" mentions)
                        steps_matches = re.findall(r'Total steps: (\d+)', report_text)
                        if steps_matches:
                            results['optimization_steps'] = sum(int(s) for s in steps_matches)
                        
                        # Extract runtime if available
                        time_match = re.search(r'Total time:\s*([\d.]+)', report_text)
                        if time_match:
                            results['runtime'] = float(time_match.group(1))
                
                yield f"data: RESULTS:{json.dumps(results)}\n\n"
                
                # Generate visualization
                yield f"data: 🔬 Generating 3D visualization...\n\n"
                try:
                    viz_cmd = [sys.executable, 'visualize_structures.py', str(sim_dir)]
                    viz_result = subprocess.run(viz_cmd, capture_output=True, text=True, timeout=10)
                    
                    if viz_result.returncode == 0:
                        viz_file = sim_dir / 'visualization.html'
                        if viz_file.exists():
                            # Send the direct URL
                            viz_url = f"http://localhost:5001/visualization/{sim_name}/visualization.html"
                            yield f"data: VIZURL:{viz_url}\n\n"
                            yield f"data: ✅ 3D visualization ready! Click the button to view.\n\n"
                    else:
                        yield f"data: ❌ Visualization generation failed: {viz_result.stderr}\n\n"
                except Exception as e:
                    yield f"data: ❌ Could not generate visualization: {e}\n\n"
            else:
                yield f"data: ⚠️ Simulation directory not found: {sim_name}\n\n"
            
            yield "data: COMPLETE\n\n"
            
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
        finally:
            # Cleanup
            if os.path.exists(config_path):
                os.unlink(config_path)
            if 'current' in active_processes:
                del active_processes['current']
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop', methods=['POST'])
def stop_simulation():
    """Stop running simulation"""
    if 'current' in active_processes:
        process = active_processes['current']
        process.terminate()
        time.sleep(0.5)
        if process.poll() is None:
            process.kill()
        del active_processes['current']
        return jsonify({"status": "stopped"})
    return jsonify({"status": "no active simulation"})

@app.route('/batch', methods=['POST'])
def run_batch():
    """Run batch screening with streaming output"""
    data = request.json
    molecules = data.get('molecules', [])
    target = data.get('target')
    substrate = data.get('substrate', 'vacuum')
    
    def generate(molecules, target, substrate):
        # Create batch config
        batch_config = {
            "probes": molecules,
            "substrate": substrate
        }
        if target:
            batch_config["target"] = target
        
        # Save config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(batch_config, f)
            config_path = f.name
        
        try:
            yield f"data: Starting batch screening with {len(molecules)} molecules\n\n"
            yield f"data: Configuration: {json.dumps(batch_config, indent=2)}\n\n"
            
            # Run batch_comparison.py
            cmd = [sys.executable, 'batch_comparison.py', config_path]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            
            current_molecule = None
            molecule_index = 0
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    yield f"data: {line}\n\n"
                    
                    # Track which molecule is being processed
                    for i, mol in enumerate(molecules):
                        if mol in line and ('Processing' in line or 'Running' in line or 'Optimizing' in line):
                            current_molecule = mol
                            molecule_index = i
                            yield f"data: MOLECULE_STATUS:{i}:running\n\n"
                            break
                    
                    # Check for completion
                    if current_molecule and ('Complete' in line or 'Done' in line or 'Finished' in line):
                        yield f"data: MOLECULE_STATUS:{molecule_index}:completed\n\n"
            
            process.wait()
            
            # Find the actual comparison results directory
            # batch_comparison.py creates directories with specific naming patterns
            import glob
            
            results = None
            results_dir = None
            
            # Look for the most recent comparison directory
            comparison_dirs = glob.glob('simulations/comparison_*')
            if comparison_dirs:
                # Get the most recent one
                results_dir = Path(max(comparison_dirs, key=os.path.getmtime))
                results_file = results_dir / 'comparison_results.json'
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    yield f"data: BATCH_RESULTS:{json.dumps(results)}\n\n"
                    
                    # Also send the directory name for visualization
                    yield f"data: BATCH_DIR:{results_dir.name}\n\n"
            
            # Generate visualization for each molecule
            # The results should exist and have been sent above
            if results and 'results' in results:
                substrate = batch_config.get('substrate', 'vacuum')
                target = batch_config.get('target', '')
                
                viz_links = []
                yield f"data: 🔍 Looking for molecule directories...\n\n"
                
                for probe in molecules:
                    # Try different possible directory names (handle typos like caffine vs caffeine)
                    possible_dirs = []
                    if target:
                        # Try exact match first
                        possible_dirs.append(Path('simulations') / f"{probe}_{target}_{substrate}")
                        # Try with common variations
                        for target_variant in [target, target.replace('caffine', 'caffeine'), target.replace('caffeine', 'caffine')]:
                            possible_dirs.append(Path('simulations') / f"{probe}_{target_variant}_{substrate}")
                    else:
                        possible_dirs.append(Path('simulations') / f"{probe}_{substrate}")
                    
                    mol_dir = None
                    for possible_dir in possible_dirs:
                        if possible_dir.exists():
                            mol_dir = possible_dir
                            break
                    
                    if mol_dir:
                        yield f"data: 📁 Found directory for {probe}: {mol_dir.name}\n\n"
                        
                        # Generate visualization for this molecule
                        viz_cmd = [sys.executable, 'visualize_structures.py', str(mol_dir)]
                        viz_result = subprocess.run(viz_cmd, capture_output=True, text=True, timeout=10)
                        
                        if viz_result.returncode == 0:
                            viz_file = mol_dir / 'visualization.html'
                            if viz_file.exists():
                                viz_url = f"{mol_dir.name}/visualization.html"
                                viz_links.append({"molecule": probe, "url": viz_url})
                                yield f"data: ✅ Generated 3D visualization for {probe}\n\n"
                            else:
                                yield f"data: ⚠️ Visualization file not created for {probe}\n\n"
                        else:
                            yield f"data: ❌ Failed to generate visualization for {probe}: {viz_result.stderr}\n\n"
                    else:
                        yield f"data: ⚠️ Directory not found for {probe}\n\n"
                
                # Send all visualization links
                if viz_links:
                    yield f"data: VIZ_LINKS:{json.dumps(viz_links)}\n\n"
            
            yield "data: BATCH_COMPLETE\n\n"
            
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    return Response(generate(molecules, target, substrate), mimetype='text/event-stream')

@app.route('/examples')
def get_examples():
    """Get example configurations from example_configs directory"""
    examples = []
    example_dir = Path('example_configs')
    
    if example_dir.exists():
        for category in example_dir.iterdir():
            if category.is_dir():
                for config_file in category.glob('*.json'):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        examples.append({
                            "name": config_file.stem.replace('_', ' ').title(),
                            "category": category.name,
                            "config": config
                        })
    
    return jsonify(examples)

@app.route('/molecules')
def list_molecules():
    """List available molecule files"""
    molecules = []
    mol_dir = Path('molecules')
    
    if mol_dir.exists():
        for mol_file in mol_dir.glob('*.xyz'):
            molecules.append(mol_file.stem)
    
    return jsonify(molecules)

@app.route('/substrates')
def list_substrates():
    """List available substrates"""
    substrates = ['vacuum']  # Start with vacuum
    sub_dir = Path('substrate')
    
    if sub_dir.exists():
        # Get all subdirectories that contain CONTCAR files
        for subdir in sub_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('__'):
                contcar = subdir / 'CONTCAR'
                if contcar.exists():
                    substrates.append(subdir.name)
    
    return jsonify(substrates)

@app.route('/status')
def server_status():
    """Check server and dependencies status"""
    status = {
        "server": "running",
        "active_simulations": len(active_processes),
        "directories": {
            "simulations": os.path.exists('simulations'),
            "molecules": os.path.exists('molecules'),
            "substrates": os.path.exists('substrate'),
            "examples": os.path.exists('example_configs')
        },
        "scripts": {
            "smart_fairchem_flow": os.path.exists('smart_fairchem_flow.py'),
            "batch_comparison": os.path.exists('batch_comparison.py'),
            "molecule_downloader": os.path.exists('molecule_downloader.py')
        }
    }
    
    # Check for fairchem
    try:
        import fairchem
        status["fairchem"] = "installed"
    except ImportError:
        status["fairchem"] = "not installed"
    
    return jsonify(status)

@app.route('/visualization/<path:filename>')
def serve_visualization(filename):
    """Serve visualization HTML files"""
    file_path = Path('simulations') / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory('simulations', filename)
    return jsonify({"error": "Visualization not found"}), 404

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download simulation results"""
    file_path = Path('simulations') / filename
    if file_path.exists() and file_path.is_file():
        return send_from_directory('simulations', filename, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    print("=" * 60)
    print("RAPIDS Web Server - Direct Integration")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print()
    print("Checking environment...")
    
    # Check for required files
    required_files = [
        'smart_fairchem_flow.py',
        'batch_comparison.py',
        'molecule_downloader.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} NOT FOUND - please ensure you're in the auto_CPT_uma_simul directory")
    
    print()
    print("Starting server on http://localhost:5001")
    print("Open your browser to http://localhost:5001 to use the web interface")
    print()
    
    app.run(debug=True, port=5001, threaded=True)