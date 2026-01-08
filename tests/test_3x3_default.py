#!/usr/bin/env python3
"""Quick test to verify 3×3 default"""
import sys, os, asyncio, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKSPACE = tempfile.mkdtemp(prefix="rapids_3x3_")

async def test():
    from mcp_server import handle_set_workspace, handle_scan_orientations

    await handle_set_workspace({"path": WORKSPACE})

    # Use DEFAULT parameters (no n_anchors, no num_orientations specified)
    result = await handle_scan_orientations({
        "run_name": "test_default",
        "probe": "benzene",
        "target": "benzene",
        # n_anchors and num_orientations NOT specified - should use defaults
        "fmax": 0.1,
    })

    output = result[0].text

    # Check for 9 configs
    if "Total: 9 configs" in output:
        print("✓ Default is 3×3 = 9 configs")
    else:
        print("✗ Default is NOT 9 configs")
        print(output[:500])

    shutil.rmtree(WORKSPACE, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(test())
