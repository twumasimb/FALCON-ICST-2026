#!/usr/bin/env python3
"""
Bootstrap Confidence Interval Analysis for FALCON vs FAST
Treats FALCON single values as reference points to compare against FAST distributions
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Create statistics directory if it doesn't exist
os.makedirs("statistics", exist_ok=True)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's Delta effect size."""
    n_x, n_y = len(x), len(y)
    dominance = 0
    for x_i in x:
        for y_i in y:
            if x_i > y_i:
                dominance += 1
            elif x_i < y_i:
                dominance -= 1
    delta = dominance / (n_x * n_y)
    return delta


def interpret_cliffs_delta(delta: float) -> str:
    """Interpret Cliff's Delta magnitude."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000,
                 alpha: float = 0.05, statistic='median') -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: Original data
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (0.05 for 95% CI)
        statistic: 'median' or 'mean'

    Returns:
        (lower_bound, upper_bound) of the confidence interval
    """
    np.random.seed(42)  # For reproducibility

    bootstrap_statistics = []
    n = len(data)

    stat_func = np.median if statistic == 'median' else np.mean

    for _ in range(n_bootstrap):
        # Resample WITH REPLACEMENT
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_statistics.append(stat_func(sample))

    # Calculate percentiles for CI
    lower = np.percentile(bootstrap_statistics, (alpha/2) * 100)
    upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)

    return lower, upper


def load_apfd_data(project: str, method: str, base_dir: str = "output") -> np.ndarray:
    """Load APFD data from TSV file."""
    if method.startswith('FALCON'):
        subdir = 'FALCON'
    elif method.startswith('FAST'):
        subdir = 'FAST'
    else:
        subdir = method.split('-')[0]

    file_path = Path(base_dir) / project / subdir / f"{method}-bbox.tsv"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, sep='\t')
    return df['APFD'].values


def holm_bonferroni_correction(p_values: list, alpha: float = 0.05) -> list:
    """
    Apply Holm-Bonferroni correction.
    For bootstrap CI, we convert "outside CI" to pseudo p-values for correction.
    """
    n = len(p_values)
    indexed_p_values = [(p, i) for i, p in enumerate(p_values)]
    indexed_p_values.sort(key=lambda x: x[0])

    significant = [False] * n

    for rank, (p_val, orig_idx) in enumerate(indexed_p_values, start=1):
        adjusted_alpha = alpha / (n - rank + 1)
        if p_val <= adjusted_alpha:
            significant[orig_idx] = True
        else:
            break

    return significant


def analyze_project_bootstrap(project: str, base_dir: str = "output",
                               alpha: float = 0.05, n_bootstrap: int = 10000) -> pd.DataFrame:
    """
    Perform bootstrap CI analysis for a single project.
    """
    project_dir = Path(base_dir) / project

    falcon_variants = []
    fast_variants = []

    # Get variants
    falcon_dir = project_dir / 'FALCON'
    if falcon_dir.exists():
        for file in falcon_dir.glob('FALCON-*-bbox.tsv'):
            method_name = file.stem.replace('-bbox', '')
            falcon_variants.append(method_name)

    fast_dir = project_dir / 'FAST'
    if fast_dir.exists():
        for file in fast_dir.glob('FAST-*-bbox.tsv'):
            method_name = file.stem.replace('-bbox', '')
            fast_variants.append(method_name)

    print(f"\n{'='*80}")
    print(f"Project: {project}")
    print(f"{'='*80}")
    print(f"FALCON variants: {len(falcon_variants)}")
    print(f"FAST variants: {len(fast_variants)}")
    print(f"Bootstrap iterations: {n_bootstrap}")

    results = []

    # Compare each FALCON variant against each FAST variant
    for falcon in sorted(falcon_variants):
        falcon_data = load_apfd_data(project, falcon, base_dir)
        falcon_value = falcon_data[0]  # Single value

        for fast in sorted(fast_variants):
            fast_data = load_apfd_data(project, fast, base_dir)

            # Calculate bootstrap CI for FAST median
            ci_lower, ci_upper = bootstrap_ci(fast_data, n_bootstrap, alpha, 'median')

            # Check if FALCON falls outside CI
            outside_ci = falcon_value < ci_lower or falcon_value > ci_upper

            # For Holm-Bonferroni, assign pseudo p-value:
            # If outside CI → p = 0.01 (significant)
            # If inside CI → p = 0.50 (not significant)
            # This is a simplification; alternatively we could use bootstrap p-value
            pseudo_p = 0.01 if outside_ci else 0.50

            # Calculate Cliff's Delta
            delta = cliffs_delta(falcon_data, fast_data)
            effect_size_interpretation = interpret_cliffs_delta(delta)

            # Stats for FAST
            fast_median = np.median(fast_data)
            fast_mean = np.mean(fast_data)
            fast_std = np.std(fast_data)

            # Determine comparison direction
            if falcon_value > fast_median:
                comparison = "FALCON > FAST"
                advantage = "FALCON"
            elif falcon_value < fast_median:
                comparison = "FALCON < FAST"
                advantage = "FAST"
            else:
                comparison = "FALCON = FAST"
                advantage = "Tie"

            # How far outside CI (if applicable)
            if falcon_value < ci_lower:
                ci_distance = ci_lower - falcon_value
            elif falcon_value > ci_upper:
                ci_distance = falcon_value - ci_upper
            else:
                ci_distance = 0.0

            results.append({
                'FALCON': falcon,
                'FAST': fast,
                'falcon_value': falcon_value,
                'fast_median': fast_median,
                'fast_mean': fast_mean,
                'fast_std': fast_std,
                'fast_n': len(fast_data),
                'ci_lower_95': ci_lower,
                'ci_upper_95': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'outside_ci': outside_ci,
                'ci_distance': ci_distance,
                'pseudo_p_value': pseudo_p,
                'cliffs_delta': delta,
                'effect_size': effect_size_interpretation,
                'comparison': comparison,
                'advantage': advantage
            })

    return pd.DataFrame(results)


def format_results_table(df: pd.DataFrame) -> str:
    """Format results as a readable table."""
    output = []

    output.append(f"\n{'FALCON':<30} {'FAST':<10} {'F_val':<7} {'FAST_med':<9} {'95% CI':<22} {'Outside':<8} {'δ':<7} {'Effect':<10}")
    output.append("-" * 115)

    for _, row in df.iterrows():
        falcon_short = row['FALCON'].replace('FALCON-', '')[:28]
        fast_short = row['FAST'].replace('FAST-', '')

        ci_str = f"[{row['ci_lower_95']:.4f}, {row['ci_upper_95']:.4f}]"
        outside = "YES***" if row['outside_ci'] else "no"

        line = (f"{falcon_short:<30} {fast_short:<10} "
                f"{row['falcon_value']:.4f}  "
                f"{row['fast_median']:.4f}    "
                f"{ci_str:<22} "
                f"{outside:<8} "
                f"{row['cliffs_delta']:>6.3f}  "
                f"{row['effect_size']:<10}")

        output.append(line)

    return '\n'.join(output)


def main():
    """Main function to run bootstrap CI analysis."""
    base_dir = "output"
    alpha = 0.05
    n_bootstrap = 10000

    # Find all projects
    projects = []
    for item in Path(base_dir).iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            projects.append(item.name)

    projects = sorted(projects)

    print("="*80)
    print("BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
    print("="*80)
    print(f"Method: Bootstrap 95% Confidence Intervals for FAST median")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Significance criterion: FALCON outside CI → significant at α={alpha}")
    print(f"Multiple comparison correction: Holm-Bonferroni")
    print(f"Effect size: Cliff's Delta (δ)")
    print("="*80)

    all_results = {}

    for project in projects:
        try:
            results_df = analyze_project_bootstrap(project, base_dir, alpha, n_bootstrap)

            # Apply Holm-Bonferroni correction
            p_values = results_df['pseudo_p_value'].tolist()
            significant_flags = holm_bonferroni_correction(p_values, alpha)
            results_df['significant_corrected'] = significant_flags

            all_results[project] = results_df

            # Print formatted results
            print(format_results_table(results_df))

            # Summary for this project
            outside_ci = results_df['outside_ci'].sum()
            sig_corrected = results_df['significant_corrected'].sum()
            total = len(results_df)

            print(f"\nSummary:")
            print(f"  Outside 95% CI (uncorrected): {outside_ci}/{total} ({100*outside_ci/total:.1f}%)")
            print(f"  Significant after correction: {sig_corrected}/{total} ({100*sig_corrected/total:.1f}%)")

            # Save to CSV
            output_file = f"statistics/bootstrap_results_{project}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"  Results saved to: {output_file}")

        except Exception as e:
            print(f"\nError processing {project}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Overall summary
    print("\n" + "="*80)
    print("CROSS-PROJECT SUMMARY")
    print("="*80)

    total_comparisons = 0
    total_outside_ci = 0
    total_sig_corrected = 0
    total_falcon_advantage = 0

    for project, df in all_results.items():
        outside_ci = df['outside_ci'].sum()
        sig_corrected = df['significant_corrected'].sum()
        falcon_wins = (df['advantage'] == 'FALCON').sum()
        total = len(df)

        total_comparisons += total
        total_outside_ci += outside_ci
        total_sig_corrected += sig_corrected
        total_falcon_advantage += falcon_wins

        print(f"\n{project}:")
        print(f"  Outside CI: {outside_ci}/{total} ({100*outside_ci/total:.1f}%)")
        print(f"  Significant (corrected): {sig_corrected}/{total} ({100*sig_corrected/total:.1f}%)")
        print(f"  FALCON advantage: {falcon_wins}/{total} ({100*falcon_wins/total:.1f}%)")

    print(f"\n{'='*80}")
    print(f"OVERALL ({total_comparisons} comparisons):")
    print(f"  Outside CI: {total_outside_ci}/{total_comparisons} ({100*total_outside_ci/total_comparisons:.1f}%)")
    print(f"  Significant (corrected): {total_sig_corrected}/{total_comparisons} ({100*total_sig_corrected/total_comparisons:.1f}%)")
    print(f"  FALCON advantage: {total_falcon_advantage}/{total_comparisons} ({100*total_falcon_advantage/total_comparisons:.1f}%)")

    # Save combined results
    combined_df = pd.concat([df.assign(Project=proj) for proj, df in all_results.items()])
    combined_df.to_csv('statistics/bootstrap_results_all_projects.csv', index=False)
    print(f"\nCombined results saved to: statistics/bootstrap_results_all_projects.csv")


if __name__ == "__main__":
    main()
