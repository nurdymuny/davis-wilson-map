"""
Test JSON serialization of r_histogram with numpy.int64 keys.
This test validates the fix for the issue where numpy.int64 keys
cannot be serialized to JSON.
"""
import json
import numpy as np


def test_numpy_int64_issue():
    """Demonstrate the issue with numpy.int64 keys."""
    print("\n=== Testing numpy.int64 JSON serialization issue ===\n")
    
    # Simulate the issue - this should fail without the fix
    topo_charges = np.array([0, 1, 0, -1, 0, 0, 1])
    
    # This creates a dict with numpy.int64 keys (the problematic pattern)
    r_histogram_bad = dict(zip(*np.unique(topo_charges, return_counts=True)))
    
    print(f"r_histogram keys types: {[type(k).__name__ for k in r_histogram_bad.keys()]}")
    print(f"r_histogram: {r_histogram_bad}")
    
    try:
        json_str = json.dumps(r_histogram_bad)
        print(f"JSON (should fail): {json_str}")
        return False, "Should have failed but didn't!"
    except TypeError as e:
        print(f"✓ Expected error: {e}")
        return True, str(e)


def test_fixed_pattern():
    """Test the fixed pattern with int conversion."""
    print("\n=== Testing fixed pattern with int conversion ===\n")
    
    topo_charges = np.array([0, 1, 0, -1, 0, 0, 1])
    
    # Fixed pattern: convert keys to Python int
    topo_unique, topo_counts = np.unique(topo_charges, return_counts=True)
    r_histogram_fixed = {int(k): int(v) for k, v in zip(topo_unique, topo_counts)}
    
    print(f"r_histogram keys types: {[type(k).__name__ for k in r_histogram_fixed.keys()]}")
    print(f"r_histogram: {r_histogram_fixed}")
    
    try:
        json_str = json.dumps(r_histogram_fixed)
        print(f"✓ JSON serialization successful: {json_str}")
        return True, None
    except TypeError as e:
        print(f"✗ Unexpected error: {e}")
        return False, str(e)


def test_r_counts_pattern():
    """Test the r_counts loop pattern with int conversion."""
    print("\n=== Testing r_counts loop pattern ===\n")
    
    # Simulate r_histogram_global with numpy.int64 values
    r_histogram_global = [np.int64(0), np.int64(1), np.int64(0), np.int64(-1), np.int64(0)]
    
    # Fixed pattern: convert r to int
    r_counts = {}
    for r in r_histogram_global:
        r_key = int(r)  # Convert numpy.int64 to Python int
        r_counts[r_key] = r_counts.get(r_key, 0) + 1
    
    print(f"r_counts keys types: {[type(k).__name__ for k in r_counts.keys()]}")
    print(f"r_counts: {r_counts}")
    
    try:
        json_str = json.dumps(r_counts)
        print(f"✓ JSON serialization successful: {json_str}")
        return True, None
    except TypeError as e:
        print(f"✗ Unexpected error: {e}")
        return False, str(e)


if __name__ == "__main__":
    print("Testing JSON serialization of r_histogram with numpy.int64 keys")
    print("=" * 70)
    
    # Test 1: Demonstrate the issue
    success1, error1 = test_numpy_int64_issue()
    
    # Test 2: Test the fixed dict(zip(...)) pattern
    success2, error2 = test_fixed_pattern()
    
    # Test 3: Test the fixed r_counts loop pattern
    success3, error3 = test_r_counts_pattern()
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"  Test 1 (numpy.int64 issue): {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"  Test 2 (fixed dict pattern): {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"  Test 3 (fixed r_counts pattern): {'✓ PASS' if success3 else '✗ FAIL'}")
    
    if success2 and success3:
        print("\n✓ All fixes validated successfully!")
        exit(0)
    else:
        print("\n✗ Some tests failed!")
        exit(1)
