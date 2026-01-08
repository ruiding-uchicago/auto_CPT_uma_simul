#!/usr/bin/env python3
"""
Test the new anchor-based scan_orientations functionality.
Tests:
1. Helper functions (get_principal_axis, generate_anchor_positions, check_contact_state)
2. Full scan_orientations with n_anchors=3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ase.build import molecule
from ase import Atoms

# Test helper functions
print("=" * 60)
print("Testing Helper Functions")
print("=" * 60)

# Import helper functions from mcp_server
from mcp_server import get_principal_axis, generate_anchor_positions, check_contact_state

# Test 1: get_principal_axis with benzene (planar, should have principal axis in-plane)
print("\n[1] Testing get_principal_axis...")
benzene = molecule('C6H6')
axis, extent = get_principal_axis(benzene)
print(f"  Benzene principal axis: {axis}")
print(f"  Benzene extent: {extent:.2f} Å")
assert extent > 2.0, "Benzene extent should be > 2 Å"
print("  ✓ get_principal_axis works!")

# Test 2: get_principal_axis with water (small, nearly spherical)
print("\n[2] Testing get_principal_axis with water...")
water = molecule('H2O')
axis, extent = get_principal_axis(water)
print(f"  Water principal axis: {axis}")
print(f"  Water extent: {extent:.2f} Å")
assert extent > 0.5, "Water extent should be > 0.5 Å"
print("  ✓ Works with small molecules!")

# Test 3: generate_anchor_positions with n_anchors=1 (legacy)
print("\n[3] Testing generate_anchor_positions (n_anchors=1)...")
anchors_1 = generate_anchor_positions(benzene, benzene, n_anchors=1)
print(f"  Single anchor: {anchors_1}")
assert len(anchors_1) == 1, "Should have 1 anchor"
assert anchors_1[0][2] == 4.0, "Default distance should be 4.0 Å"
print("  ✓ Single anchor works!")

# Test 4: generate_anchor_positions with n_anchors=3
print("\n[4] Testing generate_anchor_positions (n_anchors=3)...")
anchors_3 = generate_anchor_positions(benzene, benzene, n_anchors=3)
print(f"  Three anchors: {anchors_3}")
assert len(anchors_3) == 3, "Should have 3 anchors"
print("  ✓ Three anchors generated!")

# Test 5: check_contact_state
print("\n[5] Testing check_contact_state...")
# Create a benzene dimer in contact
benzene1 = molecule('C6H6')
benzene2 = molecule('C6H6')
benzene2.translate([0, 0, 3.5])  # Close contact
dimer_contact = benzene1 + benzene2
dimer_contact.set_cell([20, 20, 20])
dimer_contact.center()

probe_indices = list(range(12, 24))
target_indices = list(range(0, 12))

contact_info = check_contact_state(dimer_contact, probe_indices, target_indices)
print(f"  Contact dimer: is_contact={contact_info['is_contact']}, min_dist={contact_info['min_distance']:.2f} Å")
assert contact_info['is_contact'] == True, "Should be in contact"

# Create a benzene dimer NOT in contact
benzene1 = molecule('C6H6')
benzene2 = molecule('C6H6')
benzene2.translate([0, 0, 10.0])  # Far apart
dimer_far = benzene1 + benzene2
dimer_far.set_cell([20, 20, 20])
dimer_far.center()

contact_info_far = check_contact_state(dimer_far, probe_indices, target_indices)
print(f"  Far dimer: is_contact={contact_info_far['is_contact']}, min_dist={contact_info_far['min_distance']:.2f} Å")
assert contact_info_far['is_contact'] == False, "Should NOT be in contact"
print("  ✓ check_contact_state works!")

print("\n" + "=" * 60)
print("All helper function tests passed!")
print("=" * 60)

# Test full scan_orientations via MCP (optional, requires more setup)
print("\n[Optional] Full integration test requires MCP server running.")
print("To test the full flow, run:")
print("  python mcp_server.py")
print("  Then call scan_orientations with n_anchors=3")
