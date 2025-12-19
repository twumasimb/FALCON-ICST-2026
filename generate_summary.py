#!/usr/bin/env python3
"""
Experiments Results Summary Generator

Analyzes existing TSV files in output/ directory and generates:
1. Per-project summaries
2. Cross-project summary with APFD median±mad and runtime tables
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(project):
    """Load all TSV results for a project."""
    results = {}
    output_dir = Path(f"output/{project}")

    if not output_dir.exists():
        return results

    # Recursively search for TSV files in subdirectories
    for tsv_file in output_dir.rglob("*.tsv"):
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            if len(df) > 0:
                # Extract method identifier from filename
                method_key = tsv_file.stem.replace('-bbox', '')
                results[method_key] = df
        except Exception as e:
            print(f"Warning: Could not load {tsv_file}: {e}", file=sys.stderr)

    return results


def compute_mad(series):
    """Compute Median Absolute Deviation."""
    if len(series) <= 1:
        return 0.0
    median = series.median()
    return (series - median).abs().median()


def compute_stats(df):
    """Compute summary statistics for a dataframe."""
    stats = {
        'n_runs': len(df),
        'n_tests': df['NTests'].iloc[0] if 'NTests' in df.columns else 'N/A',
        'apfd_median': df['APFD'].median(),
        'apfd_mad': compute_mad(df['APFD']),
        'apfd_min': df['APFD'].min(),
        'apfd_max': df['APFD'].max(),
    }

    # Time columns - median
    sig_time = df['SignatureTime'].median() if 'SignatureTime' in df.columns else 0.0
    emb_time = df['EmbeddingTime'].median() if 'EmbeddingTime' in df.columns else 0.0
    prior_time = df['PrioritizationTime'].median() if 'PrioritizationTime' in df.columns else 0.0

    # Time columns - MAD
    sig_time_mad = compute_mad(df['SignatureTime']) if 'SignatureTime' in df.columns else 0.0
    emb_time_mad = compute_mad(df['EmbeddingTime']) if 'EmbeddingTime' in df.columns else 0.0
    prior_time_mad = compute_mad(df['PrioritizationTime']) if 'PrioritizationTime' in df.columns else 0.0

    stats['sig_time_median'] = sig_time
    stats['sig_time_mad'] = sig_time_mad
    stats['emb_time_median'] = emb_time
    stats['emb_time_mad'] = emb_time_mad
    stats['prior_time_median'] = prior_time
    stats['prior_time_mad'] = prior_time_mad
    stats['total_time_median'] = sig_time + emb_time + prior_time

    # Device info
    if 'Device' in df.columns:
        stats['device'] = df['Device'].iloc[0]
    else:
        stats['device'] = 'N/A'

    return stats


def generate_project_summary(project):
    """Generate summary for a single project."""
    results = load_results(project)

    if not results:
        return None

    summary_rows = []

    for method_key, df in sorted(results.items()):
        stats = compute_stats(df)

        summary_rows.append({
            'Method': method_key,
            'Runs': stats['n_runs'],
            'APFD': f"{stats['apfd_median']:.4f} ± {stats['apfd_mad']:.4f}",
            'APFD_Range': f"[{stats['apfd_min']:.4f}, {stats['apfd_max']:.4f}]",
            'Sig_Time': f"{stats['sig_time_median']:.2f}",
            'Emb_Time': f"{stats['emb_time_median']:.2f}",
            'Prior_Time': f"{stats['prior_time_median']:.2f}",
            'Total_Time': f"{stats['total_time_median']:.2f}"
        })

    return pd.DataFrame(summary_rows)


def generate_cross_project_summary(projects):
    """Generate cross-project comparison summary."""

    # Collect all method keys across projects
    all_methods = set()
    project_results = {}

    for project in projects:
        results = load_results(project)
        project_results[project] = results
        all_methods.update(results.keys())

    # Build comparison tables
    apfd_table = []
    time_table = []

    for method in sorted(all_methods):
        apfd_row = {'Method': method}
        time_row = {'Method': method}

        apfd_values = []
        time_values = []

        for project in projects:
            if method in project_results[project]:
                df = project_results[project][method]
                stats = compute_stats(df)

                # APFD
                apfd_row[project] = f"{stats['apfd_median']:.4f} ± {stats['apfd_mad']:.4f}"
                apfd_values.append(stats['apfd_median'])

                # # Time
                # time_row[project] = f"{stats['total_time_median']:.2f}s"
                # time_values.append(stats['total_time_median'])

                # Time
                time_row[project] = f"{stats['prior_time_median']:.2f} ± {stats['prior_time_mad']:.2f}s"
                time_values.append(stats['prior_time_median'])
            else:
                apfd_row[project] = "—"
                time_row[project] = "—"

        # Cross-project median
        if apfd_values:
            apfd_median = np.median(apfd_values)
            apfd_mad = np.median(np.abs(np.array(apfd_values) - apfd_median))
            apfd_row['Median'] = f"{apfd_median:.4f}"
            apfd_row['MAD'] = f"{apfd_mad:.4f}"
        else:
            apfd_row['Median'] = "—"
            apfd_row['MAD'] = "—"

        # Median time
        if time_values:
            time_median = np.median(time_values)
            time_mad = np.median(np.abs(np.array(time_values) - time_median))
            time_row['Median'] = f"{time_median:.2f} ± {time_mad:.2f}s"
        else:
            time_row['Median'] = "—"

        apfd_table.append(apfd_row)
        time_table.append(time_row)

    return pd.DataFrame(apfd_table), pd.DataFrame(time_table)


def save_project_summary(project, df):
    """Save per-project summary to file."""
    output_file = f"output/{project}/summary.txt"

    with open(output_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write(f"EXPERIMENT RESULTS SUMMARY — {project}\n")
        f.write("=" * 120 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "=" * 120 + "\n")

    return output_file


def save_cross_project_summary(apfd_df, time_df):
    """Save cross-project summary to markdown file."""
    output_file = f"output/summary.md"

    with open(output_file, 'w') as f:
        f.write(f"# 2026 Experiments — Results Summary\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## APFD Comparison (Median ± MAD)\n\n")
        f.write(apfd_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Runtime Comparison (Prioritization Time in seconds)\n\n")
        f.write(time_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("---\n")
        f.write("*Note: FAST methods use stochastic selection (50 reps); FALCON is deterministic (1 rep)*\n")

    return output_file


def main():
    # Define projects
    method_projects = ['Chart', 'Closure', 'Lang', 'Math', 'Mockito', 'Time']

    print("=" * 100)
    print("EXPERIMENTS RESULTS SUMMARY GENERATOR")
    print("=" * 100)
    print()

    # ========================================================================
    # METHOD-LEVEL SUMMARIES
    # ========================================================================
    print("=" * 100)
    print("METHOD-LEVEL SUMMARIES")
    print("=" * 100)
    print()

    print("Generating method-level per-project summaries...")
    for project in method_projects:
        df = generate_project_summary(project)
        if df is not None:
            output_file = save_project_summary(project, df)
            print(f"  ✓ {project}: {output_file}")
        else:
            print(f"  ✗ {project}: No results found")

    print()
    print("Generating method-level cross-project summary...")
    apfd_df, time_df = generate_cross_project_summary(method_projects)
    output_file = save_cross_project_summary(apfd_df, time_df)
    print(f"  ✓ Cross-project summary: {output_file}")
    print()

    # ========================================================================
    # FINAL MESSAGE
    # ========================================================================
    print("=" * 100)
    print("✓ Summary generation complete!")
    print("=" * 100)
    print()
    print("Output locations:")
    print("  • Summaries: output/method/")
    print("  • Per-project summaries:  output/<Project>/summary.txt")
    print("  • Cross-project summaries: output/summary.md")
    print()
    print("=" * 100)


if __name__ == '__main__':
    main()
