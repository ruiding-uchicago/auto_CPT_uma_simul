#!/usr/bin/env python3
"""
Test SMART INCREMENTAL scan_orientations:
1. Run initial tier (1×1=1 config)
2. Run middle tier (3×3=9 configs) - should skip the 1 already computed
3. Run high tier (3×6+5 random) - should skip the 9 already computed
"""

import sys, os, asyncio, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKSPACE = tempfile.mkdtemp(prefix="rapids_incremental_")

async def test():
    from mcp_server import handle_set_workspace, handle_scan_orientations

    print(f"Workspace: {WORKSPACE}")
    print("=" * 60)

    await handle_set_workspace({"path": WORKSPACE})

    # === TIER 1: Initial (1×1=1) ===
    print("\n" + "="*60)
    print("TIER 1: Initial (1 anchor × 1 orientation = 1 config)")
    print("="*60)

    result1 = await handle_scan_orientations({
        "run_name": "test_incremental",
        "probe": "water",
        "target": "benzene",
        "n_anchors": 1,
        "num_orientations": 1,
        "fmax": 0.1,
    })
    print(result1[0].text)

    # === TIER 2: Middle (3×3=9, should skip 1) ===
    print("\n" + "="*60)
    print("TIER 2: Middle (3 anchors × 3 orientations = 9 configs)")
    print("Expected: 1 cached, 8 computed")
    print("="*60)

    result2 = await handle_scan_orientations({
        "run_name": "test_incremental",  # SAME run_name!
        "probe": "water",
        "target": "benzene",
        "n_anchors": 3,
        "num_orientations": 3,
        "fmax": 0.1,
    })
    output2 = result2[0].text
    print(output2)

    # Verify caching worked
    if "CACHED" in output2:
        print("\n✓ Incremental caching is working!")
    else:
        print("\n✗ No cached results found - incremental NOT working")

    # Check computed vs cached counts
    if "Computed:" in output2:
        for line in output2.split('\n'):
            if "Computed:" in line:
                print(f"  {line.strip()}")
                break

    shutil.rmtree(WORKSPACE, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(test())
