"""
Method-Level Test Case Prioritization - Cleaned Script

This script implements TCP approaches for method-level testing:

SUBMODULAR METHODS (with code embeddings, lazy greedy only):
- FALCON: Facility Location
- LOGDET: Log Determinant
- GRAPHCUT: Graph Cut

FAST METHODS (with LSH-based similarity):
- FAST-pw: Pairwise comparison
- FAST-all, FAST-sqrt, FAST-log, FAST-one: Candidate set variants

Usage:
    python py/prioritize_method.py <project> <algorithm> [model] [repetitions] [metric]

Examples:
    # Submodular with embeddings (require model)
    python py/prioritize_method.py Chart FALCON codexembed 1
    python py/prioritize_method.py Chart LOGDET unixcoder 1 cosine
    python py/prioritize_method.py Chart GRAPHCUT codebert 1 euclidean

    # FAST methods (use 'none' for model, stochastic so 50 reps recommended)
    python py/prioritize_method.py Chart FAST-pw none 50
    python py/prioritize_method.py Chart FAST-sqrt none 50

Arguments:
    project: Project name (e.g., Chart)
    algorithm: One of FALCON, LOGDET, GRAPHCUT, FAST-pw, FAST-all, FAST-sqrt, FAST-log, FAST-one
    model: Embedding model (unixcoder/codexembed/graphcodebert/codebert/starencoder) or 'none'
    repetitions: Number of runs (default: 1 for submodular, 50 for FAST)
    metric: Distance metric for embeddings ('cosine' or 'euclidean', default: 'cosine')
"""

import os
import sys
import time
import pickle
import subprocess
import random
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

import submodtcp
import metric
import fast
import math


def load_or_generate_embeddings(project, model_key='codexembed'):
    """
    Load existing embeddings or generate them if they don't exist.

    Args:
        project (str): Project name (e.g., 'Chart')
        model_key (str): Embedding model to use

    Returns:
        tuple: (embeddings array, embedding generation time)
    """
    embeddings_dir = f"input/{project}/embeddings"
    embeddings_file = f"{embeddings_dir}/{model_key}_test_embeddings.npy"

    if not os.path.exists(embeddings_file):
        print(f"\n{'='*70}")
        print(f"Embeddings not found at {embeddings_file}")
        print(f"Generating embeddings using {model_key}...")
        print(f"{'='*70}")

        os.makedirs(embeddings_dir, exist_ok=True)

        embed_start = time.time()
        result = subprocess.run([
            'python3', 'py/embedding_generator.py',
            '--project', project,
            '--models', model_key,
            '--device', 'cuda'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"\nERROR: Embedding generation failed")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError("Embedding generation failed")

        embed_time = time.time() - embed_start
        print(f"\nEmbeddings generated in {embed_time:.2f} seconds")
    else:
        print(f"\nLoading existing embeddings from {embeddings_file}")
        embed_time = 0.0

    embeddings = np.load(embeddings_file)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, embed_time


def falcon(project, model_key='codexembed', metric='cosine'):
    """
    Facility Location black-box prioritization with lazy greedy.

    Args:
        project (str): Project name
        model_key (str): Embedding model
        metric (str): Distance metric - 'cosine' or 'euclidean'

    Returns:
        tuple: (embedding_time, prioritization_time, prioritization_list)
    """
    embeddings, embed_time = load_or_generate_embeddings(project, model_key)

    ptime_start = time.perf_counter()

    # Determine device
    device = 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    print(f"Using device: {device}")
    print(f"Using metric: {metric}")

    # Create Facility Location function
    fl = submodtcp.FacilityLocation(
        source_embeddings=embeddings,
        target_embeddings=None,
        metric=metric,
        device=device
    )

    # Run lazy greedy algorithm
    budget = embeddings.shape[0]
    print(f"Running Facility Location with lazy greedy, budget={budget}...")

    selected_indices, _ = fl.lazy_greedy(budget=budget)

    # Prioritization is 0-indexed (as returned by greedy)
    prioritization = selected_indices if isinstance(selected_indices, list) else selected_indices.tolist()

    ptime = time.perf_counter() - ptime_start

    return embed_time, ptime, prioritization


def logdet(project, model_key='codexembed', lambda_param=1e-6, metric='cosine'):
    """
    Log Determinant black-box prioritization with lazy greedy.

    Args:
        project (str): Project name
        model_key (str): Embedding model
        lambda_param (float): Regularization parameter
        metric (str): Distance metric - 'cosine' or 'euclidean'

    Returns:
        tuple: (embedding_time, prioritization_time, prioritization_list)
    """
    embeddings, embed_time = load_or_generate_embeddings(project, model_key)

    ptime_start = time.perf_counter()

    device = 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    print(f"Using device: {device}")
    print(f"Using metric: {metric}")

    logdet = submodtcp.LogDeterminant(
        source_embeddings=embeddings,
        target_embeddings=None,
        lambda_param=lambda_param,
        metric=metric,
        device=device
    )

    # Run lazy greedy algorithm
    budget = embeddings.shape[0]
    print(f"Running Log Determinant with lazy greedy, budget={budget}...")

    selected_indices, _ = logdet.lazy_greedy(budget=budget)

    prioritization = selected_indices if isinstance(selected_indices, list) else selected_indices.tolist()

    ptime = time.perf_counter() - ptime_start

    return embed_time, ptime, prioritization


def graphcut(project, model_key='codexembed', metric='cosine'):
    """
    Graph Cut black-box prioritization with lazy greedy.

    Args:
        project (str): Project name
        model_key (str): Embedding model
        metric (str): Distance metric - 'cosine' or 'euclidean'

    Returns:
        tuple: (embedding_time, prioritization_time, prioritization_list)
    """
    embeddings, embed_time = load_or_generate_embeddings(project, model_key)

    ptime_start = time.perf_counter()

    device = 'cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
    print(f"Using device: {device}")
    print(f"Using metric: {metric}")

    gc = submodtcp.GraphCut(
        source_embeddings=embeddings,
        target_embeddings=None,
        metric=metric,
        device=device
    )

    # Run lazy greedy algorithm
    budget = embeddings.shape[0]
    print(f"Running Graph Cut with lazy greedy, budget={budget}...")

    selected_indices, _ = gc.lazy_greedy(budget=budget)

    prioritization = selected_indices if isinstance(selected_indices, list) else selected_indices.tolist()

    ptime = time.perf_counter() - ptime_start

    return embed_time, ptime, prioritization


# ============================================================================
# FAST ALGORITHMS
# ============================================================================

def fast_pw(project, k=5, r=1, b=10):
    """
    FAST-pw (pairwise) prioritization.

    Args:
        project (str): Project name
        k (int): k-shingle size
        r (int): number of rows for LSH
        b (int): number of bands for LSH

    Returns:
        tuple: (signature_time, prioritization_time, prioritization_list)
    """
    input_file = f"input/{project}/{project.split('_')[0]}-bbox.txt"
    print(f"Running FAST-pw on {input_file}...")

    stime, ptime, prioritization = fast.fast_pw(
        input_file, r=r, b=b, bbox=True, k=k, memory=True
    )

    # Convert 1-indexed to 0-indexed
    prioritization = [idx - 1 for idx in prioritization]

    return stime, ptime, prioritization


def fast_variant(project, selsize, k=5, r=1, b=10):
    """
    FAST variants (all, sqrt, log, one) prioritization.

    Args:
        project (str): Project name
        selsize (function): Selection size function
        k (int): k-shingle size
        r (int): number of rows for LSH
        b (int): number of bands for LSH

    Returns:
        tuple: (signature_time, prioritization_time, prioritization_list)
    """
    input_file = f"input/{project}/{project.split('_')[0]}-bbox.txt"
    print(f"Running FAST-{selsize.__name__[:-1]} on {input_file}...")

    stime, ptime, prioritization = fast.fast_(
        input_file, selsize, r=r, b=b, bbox=True, k=k, memory=True
    )

    # Convert 1-indexed to 0-indexed
    prioritization = [idx - 1 for idx in prioritization]

    return stime, ptime, prioritization


def get_method_directory(algorithm):
    """
    Extract base method name for directory organization.

    Examples:
        FAST-pw -> FAST
        FALCON -> FALCON
        LOGDET -> LOGDET
        GRAPHCUT -> GRAPHCUT
    """
    # For FAST variants, use FAST
    if algorithm.startswith("FAST-"):
        return "FAST"

    # For submodular methods
    for method in ["FALCON", "LOGDET", "GRAPHCUT"]:
        if algorithm.startswith(method):
            return method

    return algorithm


def save_prioritization(project, algorithm, model_key, run, prioritization, distance_metric='cosine'):
    """Save prioritization to pickle file."""
    method_dir = get_method_directory(algorithm)
    output_dir = f"output/{project}/{method_dir}/prioritized"
    os.makedirs(output_dir, exist_ok=True)

    if model_key != 'none':
        output_name = f"{algorithm}-{model_key}-{distance_metric}"
    else:
        output_name = algorithm
    filename = f"{output_dir}/{output_name}-bbox-{run}.pickle"

    with open(filename, 'wb') as f:
        pickle.dump(prioritization, f)

    print(f"Saved prioritization to {filename}")


def save_results(project, algorithm, model_key, embed_times, prioritization_times, apfds, distance_metric='cosine'):
    """Save results to TSV file."""
    method_dir = get_method_directory(algorithm)
    output_dir = f"output/{project}/{method_dir}"
    os.makedirs(output_dir, exist_ok=True)

    if model_key != 'none':
        output_name = f"{algorithm}-{model_key}-{distance_metric}"
    else:
        output_name = algorithm
    filename = f"{output_dir}/{output_name}-bbox.tsv"

    with open(filename, 'w') as f:
        if model_key != 'none':
            # For embedding-based methods
            f.write("EmbeddingTime\tPrioritizationTime\tAPFD\n")
        else:
            # For FAST methods (shingles)
            f.write("SignatureTime\tPrioritizationTime\tAPFD\n")
        for embed_time, ptime, apfd in zip(embed_times, prioritization_times, apfds):
            f.write(f"{embed_time}\t{ptime}\t{apfd}\n")

    print(f"Saved results to {filename}")


def run_experiment(project, algorithm, model_key, repetitions=1, distance_metric='cosine'):
    """
    Run prioritization experiment.

    Args:
        project (str): Project name
        algorithm (str): Algorithm name (FALCON, LOGDET, GRAPHCUT, FAST-*)
        model_key (str): Embedding model (for submodular) or 'none' (for FAST)
        repetitions (int): Number of runs (default 1)
        distance_metric (str): Distance metric for embedding-based methods ('cosine' or 'euclidean')
    """
    print("\n" + "="*70)
    print(f"METHOD-LEVEL TEST CASE PRIORITIZATION")
    print("="*70)
    print(f"Project: {project}")
    print(f"Algorithm: {algorithm}")
    if model_key != 'none':
        print(f"Embedding Model: {model_key}")
        print(f"Distance Metric: {distance_metric}")
    print(f"Repetitions: {repetitions}")
    print("="*70)

    # Check if already run
    if model_key != 'none':
        output_name = f"{algorithm}-{model_key}-{distance_metric}"
    else:
        output_name = algorithm

    method_dir = get_method_directory(algorithm)
    output_file = f"output/{project}/{method_dir}/{output_name}-bbox.tsv"
    if os.path.exists(output_file):
        print(f"\nExperiment already completed: {output_file}")
        print("Skipping...")
        return

    fault_matrix_path = f"input/{project}/fault_matrix.pickle"

    embed_times = []
    prioritization_times = []
    apfds = []

    # Define FAST selection size functions
    def all_(x): return x
    def sqrt_(x): return int(math.sqrt(x)) + 1
    def log_(x): return int(math.log(x, 2)) + 1
    def one_(x): return 1

    for run in range(repetitions):
        print(f"\n{'─'*70}")
        print(f"Run {run + 1}/{repetitions}")
        print(f"{'─'*70}")

        # Run prioritization algorithm
        if algorithm == "FALCON":
            embed_time, ptime, prioritization = falcon(project, model_key, metric=distance_metric)
        elif algorithm == "LOGDET":
            embed_time, ptime, prioritization = logdet(project, model_key, metric=distance_metric)
        elif algorithm == "GRAPHCUT":
            embed_time, ptime, prioritization = graphcut(project, model_key, metric=distance_metric)
        elif algorithm == "FAST-pw":
            embed_time, ptime, prioritization = fast_pw(project)
        elif algorithm == "FAST-all":
            embed_time, ptime, prioritization = fast_variant(project, all_)
        elif algorithm == "FAST-sqrt":
            embed_time, ptime, prioritization = fast_variant(project, sqrt_)
        elif algorithm == "FAST-log":
            embed_time, ptime, prioritization = fast_variant(project, log_)
        elif algorithm == "FAST-one":
            embed_time, ptime, prioritization = fast_variant(project, one_)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Calculate APFD
        print(f"\nCalculating APFD...")
        apfd_score = metric.apfd(prioritization, fault_matrix_path)

        # Save prioritization
        save_prioritization(project, algorithm, model_key, run + 1, prioritization, distance_metric=distance_metric)

        # Record results
        embed_times.append(embed_time)
        prioritization_times.append(ptime)
        apfds.append(apfd_score)

        # Print results
        print(f"\n{'─'*70}")
        print(f"Results for Run {run + 1}:")
        if model_key != 'none':
            print(f"  Embedding time: {embed_time:.2f}s")
        else:
            print(f"  Signature time: {embed_time:.2f}s")
        print(f"  Prioritization time: {ptime:.2f}s")
        print(f"  Total time: {embed_time + ptime:.2f}s")
        print(f"  APFD: {apfd_score:.4f}")
        print(f"{'─'*70}")

    # Save results
    save_results(project, algorithm, model_key, embed_times, prioritization_times, apfds, distance_metric=distance_metric)

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    if model_key != 'none':
        print(f"Average Embedding Time: {np.mean(embed_times):.2f}s")
    else:
        print(f"Average Signature Time: {np.mean(embed_times):.2f}s")
    print(f"Average Prioritization Time: {np.mean(prioritization_times):.2f}s")
    print(f"Average APFD: {np.mean(apfds):.4f} ± {np.std(apfds):.4f}")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("USAGE: python py/prioritize_method.py <project> <algorithm> [model] [repetitions] [metric]")
        print("\nOPTIONS:")
        print("  <project>: Project name (e.g., Chart)")
        print("  <algorithm>: Algorithm to use")
        print("  [model]: Embedding model (for submodular methods) or 'none' (for FAST)")
        print("  [repetitions]: Number of runs (default: 1 for submodular, 50 for FAST)")
        print("  [metric]: Distance metric (default: 'cosine', options: 'cosine' or 'euclidean')")
        print("\nALGORITHMS:")
        print("  Submodular with embeddings (lazy greedy):")
        print("    FALCON, LOGDET, GRAPHCUT (require model, support metric)")
        print("  FAST with shingles:")
        print("    FAST-pw, FAST-all, FAST-sqrt, FAST-log, FAST-one (use 'none' for model)")
        print("\nMODELS (for submodular methods):")
        print("  unixcoder, codexembed, graphcodebert, codebert, starencoder")
        print("\nMETRICS (for submodular methods):")
        print("  cosine (default) - Semantic similarity, range [0,1]")
        print("  euclidean - Absolute distance in embedding space")
        print("\nEXAMPLES:")
        print("  # Submodular with embeddings and cosine distance (default)")
        print("  python py/prioritize_method.py Chart FALCON codexembed 1")
        print("  python py/prioritize_method.py Chart FALCON codexembed 1 cosine")
        print("  # Submodular with embeddings and euclidean distance")
        print("  python py/prioritize_method.py Chart LOGDET unixcoder 1 euclidean")
        print("  # FAST with shingles (stochastic, 50 reps recommended)")
        print("  python py/prioritize_method.py Chart FAST-pw none 50")
        print("  python py/prioritize_method.py Chart FAST-sqrt none 50")
        sys.exit(1)

    project = sys.argv[1]
    algorithm = sys.argv[2]

    # Determine defaults based on algorithm type
    submodular_algorithms = {"FALCON", "LOGDET", "GRAPHCUT"}
    fast_algorithms = {"FAST-pw", "FAST-all", "FAST-sqrt", "FAST-log", "FAST-one"}

    if algorithm in submodular_algorithms:
        # Submodular methods with embeddings (deterministic, 1 rep)
        if len(sys.argv) < 4:
            print(f"ERROR: {algorithm} requires a model parameter")
            print("Valid models: unixcoder, codexembed, graphcodebert, codebert, starencoder")
            sys.exit(1)
        model_key = sys.argv[3]
        repetitions = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        distance_metric = sys.argv[5] if len(sys.argv) > 5 else 'cosine'
    elif algorithm in fast_algorithms:
        # FAST methods with shingles (stochastic, 50 reps)
        model_key = sys.argv[3] if len(sys.argv) > 3 else 'none'
        repetitions = int(sys.argv[4]) if len(sys.argv) > 4 else 50
        distance_metric = 'cosine'  # Not used for FAST methods
    else:
        print(f"ERROR: Invalid algorithm '{algorithm}'")
        print(f"Valid algorithms: {', '.join(sorted(submodular_algorithms | fast_algorithms))}")
        sys.exit(1)

    # Validate inputs
    valid_models = {"unixcoder", "codexembed", "graphcodebert", "codebert", "starencoder", "none"}
    valid_metrics = {"cosine", "euclidean"}

    if model_key not in valid_models:
        print(f"ERROR: Invalid model '{model_key}'")
        print(f"Valid options: {', '.join(valid_models)}")
        sys.exit(1)

    if distance_metric not in valid_metrics:
        print(f"ERROR: Invalid distance metric '{distance_metric}'")
        print(f"Valid options: {', '.join(valid_metrics)}")
        sys.exit(1)

    if algorithm in submodular_algorithms and model_key == 'none':
        print(f"ERROR: {algorithm} requires an embedding model")
        print("Valid models: unixcoder, codexembed, graphcodebert, codebert, starencoder")
        sys.exit(1)

    if repetitions <= 0:
        print(f"ERROR: Repetitions must be positive")
        sys.exit(1)

    # Run experiment
    try:
        run_experiment(project, algorithm, model_key, repetitions, distance_metric)
    except Exception as e:
        print(f"\nERROR: Experiment failed")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)