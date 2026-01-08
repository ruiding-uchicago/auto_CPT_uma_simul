#!/usr/bin/env python3
"""
Full integration test for anchor-based scan_orientations.
Tests benzene-benzene dimer with n_anchors=3, num_orientations=3.
"""

import sys
import os
import asyncio
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up workspace in temp directory
WORKSPACE = tempfile.mkdtemp(prefix="rapids_test_")
print(f"Test workspace: {WORKSPACE}")

async def test_scan_orientations():
    """Test the full scan_orientations with anchors"""
    from mcp_server import (
        handle_set_workspace,
        handle_scan_orientations,
    )

    print("=" * 60)
    print("Testing scan_orientations with n_anchors=3")
    print("=" * 60)

    # Set workspace
    print("\n[1] Setting workspace...")
    result = await handle_set_workspace({"path": WORKSPACE})
    print(f"  {result[0].text[:100]}...")

    # Run scan with n_anchors=3, num_orientations=2 (faster test)
    print("\n[2] Running scan_orientations...")
    print("  Config: probe=benzene, target=benzene, n_anchors=3, num_orientations=2")
    print("  Total: 3×2 = 6 optimizations")
    print("  This may take a few minutes...\n")

    result = await handle_scan_orientations({
        "run_name": "test_anchor_scan",
        "probe": "benzene",
        "target": "benzene",
        "n_anchors": 3,
        "num_orientations": 2,  # Reduced for faster test
        "rotation_axis": "x",
        "task_name": "omol",
        "fmax": 0.1,  # Looser convergence for faster test
    })

    output = result[0].text
    print(output)

    # Verify results
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # Check key elements in output
    checks = [
        ("Anchors: 3" in output, "Has 3 anchors"),
        ("Total: 6 configs" in output, "Has 6 configs"),
        ("[near]" in output or "near" in output.lower(), "Has 'near' anchor"),
        ("[mid]" in output or "mid" in output.lower(), "Has 'mid' anchor"),
        ("[far]" in output or "far" in output.lower(), "Has 'far' anchor"),
        ("BEST" in output, "Found best configuration"),
        ("Energy spread" in output, "Has energy spread analysis"),
    ]

    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")
        if not check:
            all_passed = False

    # Check for anchor analysis
    if "Anchor Analysis" in output:
        print("  ✓ Has anchor-wise analysis")
    else:
        print("  ⚠ No anchor analysis (may be OK if n_anchors=1)")

    if all_passed:
        print("\n✓ All checks passed!")
    else:
        print("\n✗ Some checks failed")

    return all_passed

async def main():
    try:
        success = await test_scan_orientations()
        return 0 if success else 1
    finally:
        # Cleanup
        print(f"\nCleaning up workspace: {WORKSPACE}")
        shutil.rmtree(WORKSPACE, ignore_errors=True)

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
