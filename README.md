# FALCON: Efficient Test Case Prioritization via Submodular Optimization

FALCON is the **first similarity-based test case prioritization approach grounded in submodular optimization**, using the facility location function to provide a principled mathematical foundation for test ordering. By representing test cases as code embeddings and maximizing a facility location objective, FALCON achieves superior fault detection effectiveness while maintaining practical scalability.

## Overview

**Zenodo DOI**: [10.5281/zenodo.18897073](https://doi.org/10.5281/zenodo.18897073)

This repository implements FALCON and evaluates it against state-of-the-art test case prioritization (TCP) baselines:

**FALCON** (Primary Contribution):
- **Novel approach**: Facility location-based submodular optimization
- **Principled**: Mathematically grounded with provable approximation guarantees
- **Effective**: ~16.4% median APFD improvement over best FAST baseline
- **Scalable**: Deterministic greedy algorithm, median 5s prioritization time

**Alternative Submodular Functions** (for comparison):
- **LOGDET**: Log Determinant maximization
- **GRAPHCUT**: Graph Cut maximization

**FAST Baselines** (state-of-the-art similarity-based TCP):
- **FAST-pw**: Pairwise comparison
- **FAST** variants: FAST-all, FAST-sqrt, FAST-log, FAST-one

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- CUDA-capable GPU (recommended for embedding generation)

## Quick Start

### Option A: Run All Experiments (Automated)

The fastest way to reproduce all experiments:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh full    # Full experiments (recommended)
./run_all_experiments.sh quick   # Quick test (1 rep for all)
```

This script will:
- Run FALCON, LOGDET, and GRAPHCUT with 5 embedding models × 2 metrics (30 configs)
- Run 5 FAST baseline variants with 50 repetitions each
- Generate embeddings automatically if not present
- Save results to `output/<project>/` directories
- Create comprehensive summary tables

**Total**: 35 experiment configurations per project across 6 Defects4J projects (Chart, Closure, Lang, Math, Mockito, Time).

### Option B: Manual Execution

For running individual experiments:

#### 1. Generate Embeddings

Generate code embeddings for a project using a pre-trained model:

```bash
python py/embedding_generator.py --project Chart --models codexembed
```

Available models: `unixcoder` (recommended), `codexembed`, `graphcodebert`, `codebert`, `starencoder`

**Note**: UniXcoder achieves the best balance between fault detection (median APFD: 0.731) and efficiency (17.23s preparation time).

#### 2. Run Prioritization

**FALCON** (recommended):

```bash
python py/prioritize_method.py Chart FALCON unixcoder 1 cosine
```

**Other submodular methods** (for comparison):

```bash
python py/prioritize_method.py Chart LOGDET unixcoder 1 cosine
python py/prioritize_method.py Chart GRAPHCUT unixcoder 1 euclidean
```

**FAST baselines** (stochastic, 50 runs recommended):

```bash
python py/prioritize_method.py Chart FAST-pw none 50
python py/prioritize_method.py Chart FAST-sqrt none 50
```

#### 3. Generate Summary

Analyze results across all experiments:

```bash
python generate_summary.py
```

Results are saved to `output/<project>/summary.txt` and `output/summary.md`.

## Project Structure

```
falcon/
├── py/                          # Core implementation
│   ├── prioritize_method.py     # Main experiment runner
│   ├── submodtcp.py            # Submodular TCP methods
│   ├── fast.py                 # FAST baseline methods
│   ├── embedding_generator.py  # Code embedding generation
│   ├── metric.py               # APFD calculation
│   └── lsh.py                  # LSH utilities
├── input/                       # Input data (test methods, fault matrices)
├── output/                      # Experiment results
├── generate_summary.py          # Results analysis
├── statistical_analysis.py      # Statistical comparison
└── requirements.txt
```

## Input Format

Each project requires:
- `<project>-bbox.txt`: Test methods (one per line)
- `fault_matrix.pickle`: Fault detection matrix (method_id -> [bug_ids])
- `embeddings/`: Generated code embeddings (auto-created)

## Output Format

Results are saved as TSV files:
```
output/<project>/<method>/<algorithm>-<model>-<metric>-bbox.tsv
```

Columns: `EmbeddingTime | PrioritizationTime | APFD`

## Key Results

- **16.4% median APFD improvement** over best FAST baseline
- **Median prioritization time**: ~5 seconds (2,193-8,006 test methods)
- **Best embedding model**: UniXcoder (0.731 median APFD, 17.23s setup)
- **Deterministic**: Reproducible prioritization orders (vs. stochastic FAST)

## Evaluation Metric

**APFD** (Average Percentage of Faults Detected): Measures how quickly prioritized test suites detect faults. Higher is better (range: 0-1).

**Formula**: APFD = 1 - (Σ position_of_first_fault_detection) / (n × m) + 1/(2n)

where n = number of test methods, m = number of unique faults.

## Distance Metrics

For embedding-based methods:
- `cosine`: Semantic similarity (default, range [0,1])
- `euclidean`: Absolute distance in embedding space

## Theoretical Foundation

FALCON formulates TCP as maximizing a **facility location function**, a monotone submodular function from operations research. Each test case acts as a "facility" that "covers" similar tests. The function exhibits diminishing returns: selecting similar tests contributes less marginal value than selecting dissimilar ones—naturally encoding the dissimilarity-based prioritization objective.

## License

This project builds upon the [FAST](https://github.com/icse18-FAST) implementation (ICSE'18). See individual files for license details.
