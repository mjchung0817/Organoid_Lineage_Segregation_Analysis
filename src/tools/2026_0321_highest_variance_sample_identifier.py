import pandas as pd
import os
import shutil
import glob
from collections import defaultdict

# ==============================================================================
# CONFIG
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CELL_TYPE_COL     = 'cell_type_dapi_adusted'   # typo present in dataset (no 'j')
CELL_TYPE_COL_ALT = 'cell_type_dapi_adjusted'

MAX_SAMPLES = 3   # keep at most this many organoids per condition

DRY_RUN = True    # set False to actually move files


# ==============================================================================
# HELPER: auto-detect condition_levels from experiment folder name
# Exp3 has morphogen conditions (BMP4/Wnt5a) → need 2 parts (dox + morphogen)
# Exp1/2 only differ by dox                  → need 1 part (dox only)
# ==============================================================================
def condition_levels_for(exp_folder_name):
    name = exp_folder_name.upper()
    if 'BMP4' in name or 'WNT' in name:
        return 2
    return 1


# ==============================================================================
# CORE: identify + optionally move highest-variance samples per condition
# ==============================================================================
def prune_rep_folder(rep_path, cell_type_col, max_samples, condition_levels, dry_run):
    """
    For each (dox [x morphogen]) condition group in rep_path:
      - compute cell-type % per sample
      - iteratively remove the sample whose removal most reduces mean variance
        across cell-type compositions (i.e. the statistical outlier)
      - move removed files to _archive/ inside rep_path
    Returns dict: condition -> list of archived filenames
    """
    archive_dir = os.path.join(rep_path, '_archive')
    if not dry_run:
        os.makedirs(archive_dir, exist_ok=True)

    # Group CSVs by condition key
    condition_files = defaultdict(list)
    for fname in sorted(f for f in os.listdir(rep_path) if f.endswith('.csv')):
        parts = fname.replace('.csv', '').split('_')
        condition = '_'.join(parts[:condition_levels])
        condition_files[condition].append(fname)

    pruned = {}

    for condition, files in sorted(condition_files.items()):
        n = len(files)
        if n <= max_samples:
            print(f"  {condition}: {n} samples — OK")
            pruned[condition] = []
            continue

        print(f"  {condition}: {n} samples — removing {n - max_samples}")

        # Build pivot: rows=sample index, cols=cell type, values=% composition
        sample_pcts = []
        for i, fname in enumerate(files):
            df = pd.read_csv(os.path.join(rep_path, fname))
            col = cell_type_col if cell_type_col in df.columns else CELL_TYPE_COL_ALT
            if col not in df.columns:
                print(f"    WARNING: cell type column not found in {fname}, skipping")
                continue
            pct = df[col].value_counts(normalize=True).mul(100)
            pct.name = i
            sample_pcts.append(pct)

        if len(sample_pcts) < 2:
            pruned[condition] = []
            continue

        pivot = pd.DataFrame(sample_pcts).fillna(0)
        to_remove = []  # list of (sample_index, filename, var_drop)

        while pivot.shape[0] > max_samples:
            full_var = pivot.var(axis=0).mean()
            var_drops = {}
            for idx in pivot.index:
                reduced = pivot.drop(index=idx)
                var_drops[idx] = full_var - reduced.var(axis=0).mean()
            worst = max(var_drops, key=lambda k: var_drops[k])
            pivot = pivot.drop(index=worst)
            to_remove.append((worst, files[worst], var_drops[worst]))

        pruned[condition] = [fname for _, fname, _ in to_remove]

        for _, fname, vdrop in to_remove:
            src = os.path.join(rep_path, fname)
            dst = os.path.join(archive_dir, fname)
            tag = '[DRY RUN] ' if dry_run else ''
            print(f"    {tag}-> archive: {fname}  (var_drop={vdrop:.3f})")
            if not dry_run:
                shutil.move(src, dst)

    return pruned


# ==============================================================================
# MAIN: discover all experiment folders, then all Rep* folders within each
# ==============================================================================
def run(root_dir, cell_type_col, max_samples, dry_run):
    # Find all Rep* folders recursively (works whether script is at root or inside one exp)
    rep_folders = sorted(glob.glob(os.path.join(root_dir, '**', '*Rep*'), recursive=True))
    # Exclude _archive dirs and the script itself
    rep_folders = [p for p in rep_folders if os.path.isdir(p) and '_archive' not in p]

    if not rep_folders:
        print(f"No Rep* folders found under {root_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Root: {root_dir}")
    print(f"Mode: {'DRY RUN (no files moved)' if dry_run else 'LIVE (files will be moved)'}")
    print(f"Max samples per condition: {max_samples}")
    print(f"Found {len(rep_folders)} Rep folder(s)")
    print(f"{'='*70}\n")

    all_pruned = {}

    # Group rep folders by their parent experiment folder
    exp_groups = defaultdict(list)
    for rep_path in rep_folders:
        exp_folder = os.path.dirname(rep_path)
        exp_groups[exp_folder].append(rep_path)

    for exp_folder, reps in sorted(exp_groups.items()):
        exp_name = os.path.basename(exp_folder)
        cond_levels = condition_levels_for(exp_name)
        print(f"\n[Experiment: {exp_name}]  (condition_levels={cond_levels})")

        for rep_path in sorted(reps):
            rep_name = os.path.basename(rep_path)
            print(f"\n  Rep: {rep_name}")
            pruned = prune_rep_folder(rep_path, cell_type_col, max_samples, cond_levels, dry_run)
            all_pruned[f"{exp_name}/{rep_name}"] = pruned

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — files {'that would be ' if dry_run else ''}archived:")
    total = 0
    for rep_key, conditions in all_pruned.items():
        for cond, files in conditions.items():
            for f in files:
                print(f"  {rep_key}/_archive/{f}")
                total += 1
    print(f"\nTotal: {total} file(s)")
    if dry_run:
        print("\nSet DRY_RUN = False at the top of the script to apply changes.")


if __name__ == '__main__':
    run(
        root_dir=PROJECT_ROOT,
        cell_type_col=CELL_TYPE_COL,
        max_samples=MAX_SAMPLES,
        dry_run=DRY_RUN,
    )
