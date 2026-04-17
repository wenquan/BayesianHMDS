"""
Run original + surrogate embedding trials for a single fish.

Intended to be called by a SLURM array job (one job per fish).

For each fish the script:
  1. Loads the correlation matrix (no uncertainty used).
  2. Runs run_embedding_trials on the real chord-distance matrix.
  3. Builds a surrogate distance matrix (spectrum-preserving, random geometry).
  4. Runs run_embedding_trials on the surrogate chord-distance matrix.
  5. Saves both best-BIC fits to results/.

Usage:
    python run_surrogate_trials.py --fish 1 [--dim 6] [--trials 10] [--seed 42]
"""

import argparse
import os
import numpy as np
from analysis_from_mat import (
    load_dmat_from_mat,
    corr_to_distance,
    surrogate_distance_matrix,
    run_embedding_trials,
)

MATRIX_VAR = 'distance_laserOn'
CONDITION  = 'laserOn'


def main():
    parser = argparse.ArgumentParser(
        description="Run original + surrogate HMDS embedding trials for one fish."
    )
    parser.add_argument('--fish',   type=int, required=True, help='Fish number (e.g. 1-7)')
    parser.add_argument('--dim',    type=int, default=6,     help='Embedding dimension (default: 6)')
    parser.add_argument('--trials', type=int, default=10,    help='Number of trials (default: 10)')
    parser.add_argument('--seed',   type=int, default=42,    help='Random seed for surrogate (default: 42)')
    args = parser.parse_args()

    mat_file    = f'data/distance_matrix_fish{args.fish}.mat'
    context     = f'{CONDITION}(d={args.dim})fish{args.fish}'
    output_orig = f'results/fit_best__{context}_nounc.pkl'
    output_surr = f'results/surrogate_fit_best__{context}.pkl'

    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Data file not found: {mat_file}")

    os.makedirs('results', exist_ok=True)

    print(f"=== Fish {args.fish} | dim={args.dim} | trials={args.trials} | seed={args.seed} ===")
    print(f"Loading {mat_file} ...")

    distance_matrix = load_dmat_from_mat(mat_file, MATRIX_VAR)
    corr_mat = 1.0 - distance_matrix
    corr_mat = (corr_mat + corr_mat.T) / 2.0
    dmat_chord = corr_to_distance(corr_mat, method='chord')

    # --- Original embedding (no uncertainty) ---
    print(f"\n--- Original embedding ---")
    trials_orig = run_embedding_trials(
        dmat_chord,
        embedding_dim=args.dim,
        n_trials=args.trials,
        dmat_unc=None,
        output_path=output_orig,
    )
    print(f"\nOriginal: lambda = {trials_orig['lambda_mean']:.4f} +/- {trials_orig['lambda_std']:.4f}")
    print(f"Best BIC = {trials_orig['bic_all'].min():.2f}")
    print(f"Results saved to: {output_orig}")

    # --- Surrogate embedding (no uncertainty) ---
    print(f"\n--- Surrogate embedding ---")
    surr = surrogate_distance_matrix(corr_mat, corr_unc=None,
                                     seed=args.seed, distance_method='chord')
    trials_surr = run_embedding_trials(
        surr['dmat_surrogate'],
        embedding_dim=args.dim,
        n_trials=args.trials,
        dmat_unc=None,
        output_path=output_surr,
    )
    print(f"\nSurrogate: lambda = {trials_surr['lambda_mean']:.4f} +/- {trials_surr['lambda_std']:.4f}")
    print(f"Best BIC = {trials_surr['bic_all'].min():.2f}")
    print(f"Results saved to: {output_surr}")


if __name__ == '__main__':
    main()
