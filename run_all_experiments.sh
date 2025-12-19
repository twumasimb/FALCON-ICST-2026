#!/bin/bash

################################################################################
# Comprehensive Method-Level Test Case Prioritization 

# This script runs TCP algorithms for fair comparison across representations:
#
# SUBMODULAR METHODS (with embeddings only):
# - 3 algorithms (FALCON, LOGDET, GRAPHCUT) × 5 models × 2 metrics = 30 experiments
#
# FAST METHODS (with shingles):
# - 5 algorithms × 50 reps = 5 experiments
#
# Total: 35 unique experiment configurations per project
#
# Usage:
#   ./run_all_experiments_cleaned.sh [quick|full]
#
# Options:
#   quick: Run 1 repetition for all (for testing)
#   full:  Run full experiments (1 rep for deterministic, 50 for stochastic)
################################################################################

set -e  # Exit on error

# Auto-detect available projects from input directory
PROJECTS=()

# If no projects found, use default list
if [ ${#PROJECTS[@]} -eq 0 ]; then
    PROJECTS=("Chart" "Lang" "Math" "Time" "Mockito" "Closure")
    echo "⚠ No projects found in input/, using default list"
fi

# Experiment mode
MODE="${1:-full}"

if [ "$MODE" == "quick" ]; then
    echo "QUICK MODE: Running 1 repetition for all experiments"
    STOCHASTIC_REPS=1
elif [ "$MODE" == "full" ]; then
    echo "FULL MODE: Running proper repetitions (1 for deterministic, 50 for stochastic)"
    STOCHASTIC_REPS=50
else
    echo "ERROR: Invalid mode '$MODE'"
    echo "Usage: ./run_all_experiments_cleaned.sh [quick|full]"
    exit 1
fi

echo "================================================================================"
echo "METHOD-LEVEL TCP - CLEANED EXPERIMENT SUITE"
echo "================================================================================"
echo "Projects: ${PROJECTS[@]}"
echo "Number of projects: ${#PROJECTS[@]}"
echo "Mode: ${MODE}"
echo "Stochastic repetitions: ${STOCHASTIC_REPS}"
echo ""

# Define algorithms
SUBMODULAR_ALGORITHMS=("FALCON" "LOGDET" "GRAPHCUT")
EMBEDDING_MODELS=("codexembed" "unixcoder" "graphcodebert" "codebert" "starencoder")
DISTANCE_METRICS=("cosine" "euclidean")
FAST_ALGORITHMS=("FAST-pw" "FAST-all" "FAST-sqrt" "FAST-log" "FAST-one")

# Calculate total experiments
SUBMODULAR_EMB_COUNT=$((${#SUBMODULAR_ALGORITHMS[@]} * ${#EMBEDDING_MODELS[@]} * ${#DISTANCE_METRICS[@]}))
FAST_SHINGLES_COUNT=${#FAST_ALGORITHMS[@]}
TOTAL_EXPERIMENTS_PER_PROJECT=$((SUBMODULAR_EMB_COUNT + FAST_SHINGLES_COUNT))
TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS_PER_PROJECT * ${#PROJECTS[@]}))

echo "Experiment breakdown per project:"
echo "  Submodular methods (with embeddings only):"
echo "    - ${SUBMODULAR_EMB_COUNT} experiments (${#SUBMODULAR_ALGORITHMS[@]} algorithms × ${#EMBEDDING_MODELS[@]} models × ${#DISTANCE_METRICS[@]} metrics)"
echo "  FAST methods (with shingles):"
echo "    - ${FAST_SHINGLES_COUNT} experiments (${#FAST_ALGORITHMS[@]} algorithms, ${STOCHASTIC_REPS} reps each)"
echo "  ─────────────────────"
echo "  Per project: ${TOTAL_EXPERIMENTS_PER_PROJECT} configurations"
echo "  TOTAL (all ${#PROJECTS[@]} projects): ${TOTAL_EXPERIMENTS} configurations"
echo "================================================================================"
echo ""

# Counter for progress tracking
current_experiment=0
total_project_counter=0

# ============================================================================
# MAIN PROJECT LOOP
# ============================================================================

for PROJECT in "${PROJECTS[@]}"; do
    total_project_counter=$((total_project_counter + 1))

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║ PROJECT ${total_project_counter}/${#PROJECTS[@]}: ${PROJECT}"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Create output directories for this project
    mkdir -p output/${PROJECT}/prioritized
    mkdir -p input/${PROJECT}/embeddings

# ============================================================================
# SUBMODULAR METHODS WITH EMBEDDINGS (Deterministic, 1 repetition)
# Testing both cosine and euclidean distance metrics
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "PHASE 1: SUBMODULAR METHODS WITH EMBEDDINGS (Deterministic)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

for algorithm in "${SUBMODULAR_ALGORITHMS[@]}"; do
    for model in "${EMBEDDING_MODELS[@]}"; do
        for metric in "${DISTANCE_METRICS[@]}"; do
            current_experiment=$((current_experiment + 1))

            echo ""
            echo "────────────────────────────────────────────────────────────────────────────"
            echo "Experiment ${current_experiment}/${TOTAL_EXPERIMENTS}: [${PROJECT}] ${algorithm} with ${model} (${metric})"
            echo "────────────────────────────────────────────────────────────────────────────"

            # Check if already run
            output_file="output/${PROJECT}/${algorithm}/${algorithm}-${model}-${metric}-bbox.tsv"
            if [ -f "${output_file}" ]; then
                echo "✓ Already completed: ${output_file}"
                echo "  Skipping..."
                continue
            fi

            # Run experiment (1 repetition for deterministic)
            python3 py/prioritize_method.py ${PROJECT} ${algorithm} ${model} 1 ${metric}

            if [ $? -eq 0 ]; then
                echo "✓ Completed: ${algorithm} with ${model} (${metric})"
            else
                echo "✗ Failed: ${algorithm} with ${model} (${metric})"
                exit 1
            fi
        done
    done
done


# ============================================================================
# FAST METHODS WITH SHINGLES (Stochastic, 50 repetitions or 1 for quick mode)
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "PHASE 2: FAST METHODS WITH SHINGLES (Stochastic, ${STOCHASTIC_REPS} repetitions)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

for algorithm in "${FAST_ALGORITHMS[@]}"; do
    current_experiment=$((current_experiment + 1))

    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "Experiment ${current_experiment}/${TOTAL_EXPERIMENTS}: [${PROJECT}] ${algorithm}"
    echo "────────────────────────────────────────────────────────────────────────────"

    # Check if already run
    output_file="output/${PROJECT}/${algorithm}-bbox.tsv"
    if [ -f "${output_file}" ]; then
        echo "✓ Already completed: ${output_file}"
        echo "  Skipping..."
        continue
    fi

    # Run experiment
    python3 py/prioritize_method.py ${PROJECT} ${algorithm} none ${STOCHASTIC_REPS}

    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${algorithm}"
    else
        echo "✗ Failed: ${algorithm}"
        exit 1
    fi
done


    # ========================================================================
    # PER-PROJECT SUMMARY
    # ========================================================================

    echo ""
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "PROJECT ${PROJECT} COMPLETED"
    echo "────────────────────────────────────────────────────────────────────────────"
    echo "Results saved to:"
    echo "  - output/${PROJECT}/*.tsv"
    echo "  - output/${PROJECT}/prioritized/*.pickle"
    echo ""

done  # End of project loop

# ============================================================================
# GENERATE COMPREHENSIVE RESULTS SUMMARY (ALL PROJECTS)
# ============================================================================

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETED FOR ALL PROJECTS"
echo "================================================================================"
echo ""

echo "Generating comprehensive results summary..."
python3 -c "
import os
import pandas as pd
import numpy as np

# Get projects list from environment
projects_str = '${PROJECTS[*]}'
projects = projects_str.split()

def get_method_directory(algorithm):
    \"\"\"Extract base method name for directory organization.\"\"\"
    if algorithm.startswith('FAST-'):
        return 'FAST'
    for base in ['FALCON', 'LOGDET', 'GRAPHCUT']:
        if algorithm.startswith(base):
            return base
    return algorithm

# Define algorithm groups
submodular_emb = [('FALCON', 'codexembed'), ('FALCON', 'unixcoder'), ('FALCON', 'graphcodebert'), ('FALCON', 'codebert'), ('FALCON', 'starencoder'),
                  ('LOGDET', 'codexembed'), ('LOGDET', 'unixcoder'), ('LOGDET', 'graphcodebert'), ('LOGDET', 'codebert'), ('LOGDET', 'starencoder'),
                  ('GRAPHCUT', 'codexembed'), ('GRAPHCUT', 'unixcoder'), ('GRAPHCUT', 'graphcodebert'), ('GRAPHCUT', 'codebert'), ('GRAPHCUT', 'starencoder')]
fast_shingles = ['FAST-pw', 'FAST-all', 'FAST-sqrt', 'FAST-log', 'FAST-one']

# Define distance metrics
distance_metrics = ['cosine', 'euclidean']

results = []

# Loop through all projects
for project in projects:
    # Submodular with embeddings
    for alg, model in submodular_emb:
        for metric in distance_metrics:
            method_dir = get_method_directory(alg)
            tsv_file = f'output/{project}/{method_dir}/{alg}-{model}-{metric}-bbox.tsv'
            if os.path.exists(tsv_file):
                df = pd.read_csv(tsv_file, sep='\t')
                apfd_median = df['APFD'].median()
                apfd_mad = (df['APFD'] - apfd_median).abs().median() if len(df) > 1 else 0.0

                # Setup time: one-time cost (embeddings)
                time_col = 'EmbeddingTime' if 'EmbeddingTime' in df.columns else 'SignatureTime'
                setup_time = df[time_col].median()
                prior_time = df['PrioritizationTime'].median()
                total_time = setup_time + prior_time

                results.append({
                    'Project': project,
                    'Method': f'{alg}-{model}',
                    'Metric': metric,
                    'Type': 'Submodular-Emb',
                    'APFD': f'{apfd_median:.4f} ± {apfd_mad:.4f}',
                    'Setup (s)': f'{setup_time:.2f}',
                    'Prior (s)': f'{prior_time:.2f}',
                    'Total (s)': f'{total_time:.2f}',
                    'Runs': len(df)
                })

    # FAST with shingles
    for alg in fast_shingles:
        method_dir = get_method_directory(alg)
        tsv_file = f'output/{project}/{method_dir}/{alg}-bbox.tsv'
        if os.path.exists(tsv_file):
            df = pd.read_csv(tsv_file, sep='\t')
            apfd_median = df['APFD'].median()
            apfd_mad = (df['APFD'] - apfd_median).abs().median() if len(df) > 1 else 0.0

            # Setup time: k-shingles signature creation
            setup_time = df['SignatureTime'].median()
            prior_time = df['PrioritizationTime'].median()
            total_time = setup_time + prior_time

            results.append({
                'Project': project,
                'Method': alg,
                'Metric': 'N/A',
                'Type': 'FAST-Shingles',
                'APFD': f'{apfd_median:.4f} ± {apfd_mad:.4f}',
                'Setup (s)': f'{setup_time:.2f}',
                'Prior (s)': f'{prior_time:.2f}',
                'Total (s)': f'{total_time:.2f}',
                'Runs': len(df)
            })

if results:
    summary_df = pd.DataFrame(results)

    print('\n' + '='*140)
    print('COMPREHENSIVE RESULTS SUMMARY - ALL PROJECTS')
    print('='*140)
    print(summary_df.to_string(index=False))
    print('='*140)

    # Save to file
    summary_file = 'output/comprehensive_results_summary_all_projects.txt'
    os.makedirs('output', exist_ok=True)
    with open(summary_file, 'w') as f:
        f.write('METHOD-LEVEL TCP - COMPREHENSIVE RESULTS SUMMARY - ALL PROJECTS\n')
        f.write('='*140 + '\n')
        f.write(summary_df.to_string(index=False))
        f.write('\n' + '='*140 + '\n')
        f.write(f'\n\nTotal experiments: {len(results)}\n')
        f.write(f'Projects: {len(projects)}\n')
    print(f'\nSummary saved to: {summary_file}')

    # Also save as CSV for easier analysis
    csv_file = 'output/comprehensive_results_all_projects.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f'CSV saved to: {csv_file}')
else:
    print('No results found.')
"

echo ""
echo "✓ All done!"
echo ""