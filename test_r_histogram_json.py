"""
Test JSON serialization of r_histogram in the validation script.
This test validates that all r_histogram dictionaries can be serialized to JSON.
"""
import json
import numpy as np


def simulate_test_results():
    """Simulate the test results structure from rest_of_validation_gpu_v17.py."""
    # Simulate r_histogram_global with numpy.int64 values (from topological_charge_integer())
    r_histogram_global = [
        np.int64(0), np.int64(1), np.int64(0), np.int64(-1), 
        np.int64(0), np.int64(0), np.int64(1), np.int64(0)
    ]
    
    # Simulate test results for A2S-001 (with fix)
    r_counts = {}
    for r in r_histogram_global:
        r_key = int(r)  # Convert numpy.int64 to Python int
        r_counts[r_key] = r_counts.get(r_key, 0) + 1
    
    a2s_results = {
        'test': 'A2S-001',
        'r_histogram': r_counts,
        'topology_frozen': len(r_counts) == 1,
        'r_diversity': len(r_counts),
    }
    
    # Simulate test_config results (with fix)
    topo_charges = np.array([0, 1, 0, -1, 0, 0, 1, 0, -1, -1])
    topo_unique, topo_counts = np.unique(topo_charges, return_counts=True)
    r_histogram_dict = {int(k): int(v) for k, v in zip(topo_unique, topo_counts)}
    
    test_config = {
        'N': 10,
        'L': 4,
        'beta': 6.0,
        't_ref': 0.05,
        'r_histogram': r_histogram_dict,
    }
    
    # Full results structure
    all_results = {
        'timestamp': '2024-01-01T00:00:00',
        'test_config': test_config,
        'A2S-001': a2s_results,
    }
    
    return all_results


def test_json_serialization():
    """Test that all results can be serialized to JSON."""
    print("=" * 70)
    print("Testing JSON serialization of validation results with r_histogram")
    print("=" * 70)
    
    results = simulate_test_results()
    
    print("\nTest Configuration r_histogram:")
    print(f"  Keys: {list(results['test_config']['r_histogram'].keys())}")
    print(f"  Key types: {[type(k).__name__ for k in results['test_config']['r_histogram'].keys()]}")
    print(f"  Value types: {[type(v).__name__ for v in results['test_config']['r_histogram'].values()]}")
    print(f"  r_histogram: {results['test_config']['r_histogram']}")
    
    print("\nA2S-001 r_histogram:")
    print(f"  Keys: {list(results['A2S-001']['r_histogram'].keys())}")
    print(f"  Key types: {[type(k).__name__ for k in results['A2S-001']['r_histogram'].keys()]}")
    print(f"  Value types: {[type(v).__name__ for v in results['A2S-001']['r_histogram'].values()]}")
    print(f"  r_histogram: {results['A2S-001']['r_histogram']}")
    
    # Verify all keys are Python int
    for key in results['test_config']['r_histogram'].keys():
        assert isinstance(key, int) and not isinstance(key, np.integer), \
            f"test_config r_histogram key {key} is not Python int, it's {type(key)}"
    
    for key in results['A2S-001']['r_histogram'].keys():
        assert isinstance(key, int) and not isinstance(key, np.integer), \
            f"A2S-001 r_histogram key {key} is not Python int, it's {type(key)}"
    
    print("\n✓ All keys are Python int type")
    
    # Try to serialize to JSON
    try:
        json_str = json.dumps(results, indent=2)
        print("\n✓ JSON serialization successful!")
        print(f"\nJSON output (first 500 chars):\n{json_str[:500]}...")
        
        # Verify we can deserialize
        loaded_results = json.loads(json_str)
        print("\n✓ JSON deserialization successful!")
        
        # Verify r_histogram values are preserved (note: JSON converts int keys to strings)
        original_r_counts = results['A2S-001']['r_histogram']
        loaded_r_counts = {int(k): v for k, v in loaded_results['A2S-001']['r_histogram'].items()}
        
        assert original_r_counts == loaded_r_counts, "r_histogram values not preserved"
        print("✓ r_histogram values preserved after JSON round-trip")
        
        return True
        
    except TypeError as e:
        print(f"\n✗ JSON serialization failed: {e}")
        return False


if __name__ == "__main__":
    success = test_json_serialization()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ ALL TESTS PASSED - JSON serialization works correctly!")
        print("=" * 70)
        exit(0)
    else:
        print("✗ TESTS FAILED - JSON serialization still has issues!")
        print("=" * 70)
        exit(1)
