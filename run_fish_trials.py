"""
Run run_embedding_trials for a single fish data file.

Usage:
    python run_fish_trials.py --fish 1 [--dim 6] [--trials 10] [--no-unc]

Intended to be called by a SLURM array job (one job per fish).
"""

import argparse
import os
import numpy as np
from analysis_from_mat import (
    load_dmat_from_mat,
    corr_to_distance,
    corr_unc_to_dist_unc,
    run_embedding_trials,
)

MATRIX_VAR   = 'distance_laserOff'
UNC_VAR      = 'var_dij_laserOff'
CONDITION    = 'laserOff'

def main():
    parser = argparse.ArgumentParser(description="Run HMDS embedding trials for one fish.")
    parser.add_argument('--fish',   type=int, required=True, help='Fish number (e.g. 1-7)')
    parser.add_argument('--dim',    type=int, default=6,     help='Embedding dimension (default: 6)')
    parser.add_argument('--trials', type=int, default=10,    help='Number of trials (default: 10)')
    parser.add_argument('--no-unc', action='store_true',     help='Ignore uncertainty matrix')
    args = parser.parse_args()

    mat_file = f'data/distance_matrix_fish{args.fish}.mat'
    context  = f'{CONDITION}(d={args.dim})fish{args.fish}'
    output   = f'results/fit_best__{context}.pkl'

    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"Data file not found: {mat_file}")

    os.makedirs('results', exist_ok=True)

    print(f"=== Fish {args.fish} | dim={args.dim} | trials={args.trials} ===")
    print(f"Loading {mat_file} ...")

    distance_matrix = load_dmat_from_mat(mat_file, MATRIX_VAR)
    corr_mat = 1.0 - distance_matrix
    corr_mat = (corr_mat + corr_mat.T) / 2.0
    dmat_chord = corr_to_distance(corr_mat, method='chord')

    dmat_unc = None
    if not args.no_unc:
        try:
            unc_matrix = load_dmat_from_mat(mat_file, UNC_VAR)
            dmat_unc = corr_unc_to_dist_unc(corr_mat, unc_matrix, method='chord', reg=0.01)
        except Exception as e:
            print(f"Warning: could not load uncertainty matrix ({e}). Continuing without it.")

    trials = run_embedding_trials(
        dmat_chord,
        embedding_dim=args.dim,
        n_trials=args.trials,
        dmat_unc=dmat_unc,
        output_path=output,
    )

    print(f"\nFish {args.fish} done.")
    print(f"lambda = {trials['lambda_mean']:.4f} +/- {trials['lambda_std']:.4f}")
    print(f"Best BIC = {trials['bic_all'].min():.2f}")
    print(f"Results saved to: {output}")


if __name__ == '__main__':
    main()
