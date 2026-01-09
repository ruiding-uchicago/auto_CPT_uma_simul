#!/usr/bin/env python3
"""
Test new batch_screening with use_scan=True (default)
"""

import sys, os, asyncio, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORKSPACE = tempfile.mkdtemp(prefix="rapids_batch_test_")

async def test():
    from mcp_server import handle_set_workspace, handle_batch_screening

    print(f"Workspace: {WORKSPACE}")
    print("=" * 60)

    await handle_set_workspace({"path": WORKSPACE})

    # Test batch_screening with default (use_scan=True)
    print("\nTesting batch_screening with 2 probes, 1 target")
    print("Default: use_scan=True, n_anchors=3, num_orientations=3")
    print("Expected: 2 probes × 9 configs = 18 total optimizations")
    print("=" * 60)

    result = await handle_batch_screening({
        "run_name": "test_batch",
        "probes": ["water", "methanol"],  # 2 simple probes
        "target": "benzene",
        # use_scan defaults to True
        # n_anchors defaults to 3
        # num_orientations defaults to 3
        "fmax": 0.1,  # looser for speed
    })

    print(result[0].text)

    shutil.rmtree(WORKSPACE, ignore_errors=True)

if __name__ == "__main__":
    asyncio.run(test())
