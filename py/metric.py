"""
Method-Level APFD Calculation for Test Case Prioritization

This module computes APFD (Average Percentage of Faults Detected) for method-level
test case prioritization. Adapted from the FAST paper's class-level APFD calculation.

Author: Adapted for ICST 2026 method-level TCP research
Original: FAST ICSE'18 implementation
"""

from collections import defaultdict
from pickle import load


def apfd(prioritization, fault_matrix_path):
    """
    Calculate APFD for method-level test case prioritization.

    INPUT:
        prioritization (list): List of test method indices in priority order (0-indexed)
        fault_matrix_path (str): Path to fault_matrix pickle file

    OUTPUT:
        float: APFD score (0-1 range, higher is better)

    FORMULA:
        APFD = 1 - (sum_{i=1}^{m} t_i / (n*m)) + (1 / (2n))
        where:
            n = total number of test methods
            m = total number of unique faults detected
            t_i = position (1-indexed) of first test revealing fault i

    FAULT MATRIX FORMAT:
        Dict {method_idx (int): [bug_ids (list of ints)]}
        Example: {0: [1, 5], 142: [1], 500: [], ...}

    NOTE:
        - Prioritization uses 0-indexed method IDs (0 to n-1)
        - Position in APFD formula is 1-indexed (1 to n)
        - This is for Chart which uses single-version fault detection
          (each bug either detected or not by each method)
    """
    # Load fault matrix
    faults_dict = load_fault_matrix(fault_matrix_path)

    # Track which faults have been detected and at what position
    detected_faults = set()
    numerator = 0.0  # Sum of positions where faults are first detected
    position = 1  # 1-indexed position in prioritization

    # Iterate through prioritized test methods
    for method_idx in prioritization:
        # Check what faults this method detects
        for fault_id in faults_dict.get(method_idx, []):
            if fault_id not in detected_faults:
                # This is the first time we've seen this fault
                detected_faults.add(fault_id)
                numerator += position
        position += 1

    # Calculate APFD
    n = len(prioritization)  # Total number of test methods
    m = len(detected_faults)  # Total number of unique faults detected

    if m == 0:
        # No faults detected at all
        return 0.0

    apfd_score = 1.0 - (numerator / (n * m)) + (1.0 / (2 * n))
    return apfd_score


def load_fault_matrix(fault_matrix_path):
    """
    Load fault matrix from pickle file.

    INPUT:
        fault_matrix_path (str): Path to fault_matrix pickle file

    OUTPUT:
        dict: {method_idx (int): [bug_ids (list of ints)]}

    NOTES:
        - Keys are method indices (0 to n-1)
        - Values are lists of bug IDs (1 to num_bugs) that the method detects
        - Methods that detect no bugs may be absent from dict (treated as empty list)
    """
    faults_dict = defaultdict(list)

    with open(fault_matrix_path, "rb") as picklefile:
        pickledict = load(picklefile)

    # Ensure all keys are integers
    for key in pickledict.keys():
        faults_dict[int(key)] = pickledict[key]

    return faults_dict


def get_fault_statistics(fault_matrix_path):
    """
    Get statistics about the fault matrix.

    Returns:
        dict: Statistics including:
            - num_methods: Total methods with fault data
            - num_methods_detecting: Number of methods that detect ≥1 fault
            - num_faults: Total unique faults
            - total_detections: Total fault detections (sum across all methods)
            - avg_faults_per_method: Average faults detected per detecting method
    """
    faults_dict = load_fault_matrix(fault_matrix_path)

    all_faults = set()
    total_detections = 0
    methods_detecting = 0

    for method_idx, bug_list in faults_dict.items():
        if len(bug_list) > 0:
            methods_detecting += 1
            total_detections += len(bug_list)
            all_faults.update(bug_list)

    stats = {
        'num_methods': len(faults_dict),
        'num_methods_detecting': methods_detecting,
        'num_faults': len(all_faults),
        'total_detections': total_detections,
        'avg_faults_per_method': total_detections / methods_detecting if methods_detecting > 0 else 0.0
    }

    return stats


if __name__ == "__main__":
    # Test with chart_method data
    import sys

    fault_matrix_path = "input/chart_method/fault_matrix.pickle"

    print("=" * 70)
    print("Method-Level APFD Calculator - Test Run")
    print("=" * 70)

    # Get statistics
    stats = get_fault_statistics(fault_matrix_path)
    print("\nFault Matrix Statistics:")
    print(f"  Total methods: {stats['num_methods']}")
    print(f"  Methods detecting faults: {stats['num_methods_detecting']}")
    print(f"  Total unique faults: {stats['num_faults']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Avg faults per detecting method: {stats['avg_faults_per_method']:.2f}")

    # Test with random prioritization
    import random
    num_methods = stats['num_methods']
    random_prioritization = list(range(num_methods))
    random.shuffle(random_prioritization)

    random_apfd = apfd(random_prioritization, fault_matrix_path)
    print(f"\nRandom prioritization APFD: {random_apfd:.4f}")

    # Test with optimal prioritization (fault-detecting methods first)
    faults_dict = load_fault_matrix(fault_matrix_path)
    detecting_methods = [idx for idx, bugs in faults_dict.items() if len(bugs) > 0]
    non_detecting_methods = [idx for idx in range(num_methods) if idx not in detecting_methods]
    optimal_prioritization = detecting_methods + non_detecting_methods

    optimal_apfd = apfd(optimal_prioritization, fault_matrix_path)
    print(f"Optimal prioritization APFD: {optimal_apfd:.4f}")

    print("\n" + "=" * 70)
