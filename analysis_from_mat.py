import metric_HMDS as HMDS
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque
import sys
import argparse
import contextlib
import pickle
import os
import io


# --- Code Quality & Clarity Suggestion: Use h5py for modern .mat files ---
# MATLAB v7.3+ files are HDF5 format and require h5py to be read.
# We import it here and will use it in the loading function.
try:
    import h5py
except ImportError:
    h5py = None
# Add matplotlib and scikit-learn for plotting and MDS/PCA
try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances
except ImportError:
    plt = None
    PCA = None
    MDS = None


class MatReadError(Exception):
    """Custom exception for MATLAB file reading errors."""
    pass

def plot_shepard_diagram(original_dmat, embedded_dmat, lambda_val=None, title="Shepard Diagram", output_file=None):
    """
    Creates and displays a Shepard diagram to compare original and embedded distances.

    Args:
        original_dmat (np.ndarray): The original, normalized distance matrix.
        embedded_dmat (np.ndarray): The distance matrix from the embedding.
        lambda_val (float): The fitted curvature scale parameter.
        title (str): The title for the plot.
        output_file (str, optional): If provided, saves the plot to this file path as a PDF.
    """
    if plt is None:
        print("\nWarning: matplotlib is not installed. Skipping plot. Please run 'pip install matplotlib'.")
        return

    print("Generating Shepard diagram...")

    # We use the upper triangle to avoid plotting each pair twice and the diagonal.
    N = original_dmat.shape[0]
    indices = np.triu_indices(N, k=1)

    original_distances = original_dmat[indices]
    if lambda_val is not None:
        embedded_distances = embedded_dmat[indices] / lambda_val
        ylabel = "Embedded Hyperbolic Distances / λ"
    else:
        embedded_distances = embedded_dmat[indices]
        ylabel = "Embedded Euclidean Distances"

    # Calculate R-squared (coefficient of determination)
    if len(original_distances) > 1 and len(embedded_distances) > 1:
        r_value = np.corrcoef(original_distances, embedded_distances)[0, 1]
        r_squared = r_value**2
    else:
        r_squared = np.nan

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(original_distances, embedded_distances, alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
    ax.plot([0, max(original_distances.max(), embedded_distances.max())], [0, max(original_distances.max(), embedded_distances.max())], 'r--', label=f'Perfect Match (y=x)\n$R^2 = {r_squared:.2f}$')
    ax.set_xlabel("Original Pairwise Distances (Normalized)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    ax.legend()
    
    if output_file:
        print(f"Saving Shepard diagram to {output_file}...")
        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_poincare_2d(poincare_coords, title="2D Hyperbolic Embedding (Poincare Disk)", colors=None, output_file=None):
    """
    Creates a 2D visualization of points in the Poincare disk.

    Args:
        poincare_coords (np.ndarray): An N x 2 array of Poincare coordinates.
        title (str): The title for the plot.
        colors (optional): An array of colors for the points.
        output_file (str, optional): If provided, saves the plot to this file path.
    """
    if plt is None:
        print("\nWarning: matplotlib is not installed. Skipping 2D plot. Please run 'pip install matplotlib'.")
        return

    print("Generating 2D Poincare disk visualization...")
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the boundary circle
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, alpha=0.5)
    ax.add_artist(circle)

    # Use provided colors (and a colormap) or default to blue
    point_colors = colors if colors is not None else 'b'
    scatter = ax.scatter(poincare_coords[:, 0], poincare_coords[:, 1], c=point_colors, s=20, cmap='Reds')

    ax.set_title(title)
    ax.set_aspect('equal', 'box')

    # Add a colorbar if a color sequence is provided
    if colors is not None:
        fig.colorbar(scatter, ax=ax, label="Point Index")

    if output_file:
        print(f"Saving 2D plot to {output_file}...")
        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_poincare_3d(poincare_coords, colors=None, output_file=None):
    """Creates an interactive 3D visualization of points in the Poincare ball.

    Args:
        poincare_coords (np.ndarray): An N x 3 array of Poincare coordinates.
        colors (optional): An array of colors for the points.
        output_file (str, optional): If provided, saves the plot to this file path.
    """
    if plt is None:
        print("\nWarning: matplotlib is not installed. Skipping 3D plot. Please run 'pip install matplotlib'.")
        return

    if poincare_coords.shape[1] != 3:
        print(f"\nWarning: 3D plot is only available for 3D embeddings. Found {poincare_coords.shape[1]} dimensions. Skipping plot.")
        return

    print("Generating interactive 3D Poincare ball visualization...")
    
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the boundary sphere (wireframe)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    # Use provided colors (and a colormap) or default to blue
    point_colors = colors if colors is not None else 'b'
    scatter = ax.scatter(poincare_coords[:, 0], poincare_coords[:, 1], poincare_coords[:, 2], c=point_colors, s=20, cmap='Reds')

    # Add a colorbar if a color sequence is provided
    if colors is not None:
        fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, label="Point Index")

    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Z coordinate")
    ax.set_title("3D Hyperbolic Embedding (Poincare Ball)")
    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])

    if output_file:
        print(f"Saving 3D plot to {output_file}...")
        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_poincare_3d_projections(poincare_coords, colors=None, output_file=None):
    """Creates 2D projections (XY, XZ, YZ) of a 3D Poincare embedding.

    Args:
        poincare_coords (np.ndarray): An N x 3 array of Poincare coordinates.
        colors (optional): An array of colors for the points.
        output_file (str, optional): If provided, saves the plot to this file path.
    """
    if plt is None:
        print("\nWarning: matplotlib is not installed. Skipping 2D projections plot.")
        return

    if poincare_coords.shape[1] != 3:
        return # Silently fail as the calling function should handle this.

    print("Generating 2D projections of the 3D embedding...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle("2D Projections of 3D Embedding", fontsize=16)

    projections = [
        (0, 1, 'X', 'Y'),
        (0, 2, 'X', 'Z'),
        (1, 2, 'Y', 'Z')
    ]

    for ax, (d1, d2, d1_name, d2_name) in zip(axes, projections):
        circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, alpha=0.5)
        ax.add_artist(circle)
        scatter = ax.scatter(poincare_coords[:, d1], poincare_coords[:, d2], c=colors, s=20, cmap='Reds')
        ax.set_title(f"{d1_name} vs {d2_name} Projection")
        ax.set_xlabel(f"{d1_name} coordinate")
        ax.set_ylabel(f"{d2_name} coordinate")
        ax.set_aspect('equal', 'box')
        ax.grid(True)

    if colors is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.8, label="Point Index")

    if output_file:
        print(f"Saving 3D projections plot to {output_file}...")
        fig.savefig(output_file, format='pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def visualize_embedding(fit_results, output_prefix=None):
    """
    Visualizes the embedding. For d=2 or d=3, it plots directly.
    For d > 3, it uses PCA to project the data to 2D for visualization.

    Args:
        fit_results (dict): The dictionary returned by run_embedding.
        output_prefix (str, optional): If provided, saves plots to files with this prefix.
    """
    if plt is None:
        print("\nWarning: matplotlib is not installed. Skipping visualization.")
        return

    coords = fit_results['cp']
    dim = coords.shape[1]

    # Generate a color array based on the index of each point
    num_points = coords.shape[0]
    colors = np.arange(num_points)

    def get_path(suffix):
        return f"{output_prefix}_{suffix}.pdf" if output_prefix else None

    if dim == 2:
        plot_poincare_2d(coords, colors=colors, output_file=get_path("2d"))
    elif dim == 3:
        plot_poincare_3d(coords, colors=colors, output_file=get_path("3d"))
        plot_poincare_3d_projections(coords, colors=colors, output_file=get_path("3d_projections"))
    elif dim > 3:
        print(f"\nEmbedding dimension is {dim} (>3). Projecting to 2D using PCA for visualization.")
        # Use PCA to find the three principal components
        pca = PCA(n_components=3)
        projected_coords = pca.fit_transform(coords)
        explained_variance = sum(pca.explained_variance_ratio_) * 100
        print(f"The 3D PCA projection explains {explained_variance:.2f}% of the variance.")

        title = f"PCA Projection of {dim}D Embedding to 3D Poincare Ball"
        plot_poincare_3d(projected_coords, colors=colors, output_file=get_path("pca_3d"))
        plot_poincare_3d_projections(projected_coords, colors=colors, output_file=get_path("pca_3d_projections")) 

def load_dmat_from_mat(mat_file_path, matrix_variable_name):
    """
    Loads a distance matrix from a .mat file, handling both old and v7.3 formats.

    Args:
        mat_file_path (str): Path to the .mat file.
        matrix_variable_name (str): The name of the distance matrix variable inside the .mat file.

    Returns:
        np.ndarray: The loaded distance matrix.
    """

    try:
        # First, try reading with scipy's loadmat, which handles older formats.
        mat_data = loadmat(mat_file_path, simplify_cells=True)
    except NotImplementedError:
        # This error is raised for v7.3 .mat files. We switch to h5py.
        print("Detected MATLAB v7.3 file, switching to h5py reader...")
        if h5py is None:
            raise MatReadError("MATLAB v7.3 file detected, but 'h5py' is not installed. Please run 'pip install h5py'.")
        
        try:
            with h5py.File(mat_file_path, 'r') as f:
                # h5py doesn't load the whole file into a dict, so we access the variable directly.
                if matrix_variable_name not in f:
                    available_vars = list(f.keys())
                    raise MatReadError(f"Variable '{matrix_variable_name}' not found in HDF5 file. Available variables: {available_vars}")
                
                # Extract the data and convert to a NumPy array.
                # Note: h5py may read data transposed compared to MATLAB. For a symmetric
                # distance matrix, this is not an issue.
                dmat = np.array(f[matrix_variable_name], dtype=float)
        except Exception as e:
            raise MatReadError(f"Failed to read HDF5 file with h5py: {e}")

    except FileNotFoundError:
        raise MatReadError(f"The file '{mat_file_path}' was not found.")
    
    except Exception as e:
        raise MatReadError(f"An unexpected error occurred while reading the .mat file: {e}")

    else:
        # This block runs if scipy.io.loadmat succeeded.
        if matrix_variable_name not in mat_data:
            available_vars = [k for k in mat_data.keys() if not k.startswith('__')]
            raise MatReadError(f"Variable '{matrix_variable_name}' not found in .mat file. Available variables: {available_vars}")
        
        dmat = mat_data[matrix_variable_name]

        # Handle complex data structures that can arise from MATLAB structs/cells
        if isinstance(dmat, MatlabOpaque):
            raise MatReadError("The variable is a complex MATLAB object. Please simplify it in MATLAB before saving.")

    # Ensure the matrix is a numpy array of the correct type (float)
    dmat = np.array(dmat, dtype=float)

    print(f"Successfully loaded matrix '{matrix_variable_name}' with shape: {dmat.shape}")

    return dmat

def calculate_bic(dmat, emb_mat, n_params, is_hyperbolic=False, lambda_val=None, sig_vals=None, dmat_unc=None):
    """
    Calculates the Bayesian Information Criterion (BIC) for an embedding.

    Args:
        dmat (np.ndarray): The original distance matrix.
        dmat_unc (np.ndarray, optional): The matrix of uncertainties for each distance.
        emb_mat (np.ndarray): The distance matrix from the embedding.
        n_params (int): The number of parameters in the model.
        is_hyperbolic (bool): Flag to indicate if the model is hyperbolic.
        lambda_val (float, optional): The lambda scale parameter for hyperbolic models.
        sig_vals (np.ndarray, optional): The uncertainty parameters for hyperbolic models.

    Returns:
        float: The calculated BIC value.
    """
    N = dmat.shape[0]
    n_pairs = N * (N - 1) / 2
    indices = np.triu_indices(N, k=1)
    
    original_distances = dmat[indices]
    embedded_distances = emb_mat[indices]

    if is_hyperbolic:
        # Combine inferred and data uncertainties, matching the Stan model
        inferred_seff_sq = np.array([sig_vals[i]**2 + sig_vals[j]**2 for i in range(N) for j in range(i + 1, N)])
        data_unc_sq = dmat_unc[indices] if dmat_unc is not None else 0
        seff_sq = inferred_seff_sq + data_unc_sq

        residuals = original_distances - (embedded_distances / lambda_val)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * seff_sq) + (residuals**2 / seff_sq))
    else: # Euclidean
        # For the Euclidean case, we assume a single global model variance (sigma_model^2)
        # plus the known data variance for each pair.
        residuals = original_distances - embedded_distances
        rss = np.sum(residuals**2)
        
        # Estimate the single model variance parameter via MLE.
        # This is equivalent to the average residual sum of squares.
        sigma2_model_mle = rss / n_pairs
        
        data_unc_sq = dmat_unc[indices] if dmat_unc is not None else 0
        total_variance = sigma2_model_mle + data_unc_sq
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * total_variance) + (residuals**2 / total_variance))

    return n_params * np.log(n_pairs) - 2 * log_likelihood

def run_embedding(dmat, embedding_dim, dmat_unc=None, verbose=False, output_path=None):
    """
    Takes a distance matrix, normalizes it, and runs the HMDS embedding.

    Args:
        dmat (np.ndarray): The input distance matrix.
        dmat_unc (np.ndarray, optional): The matrix of uncertainties for each distance.
        embedding_dim (int): The target dimension for the hyperbolic embedding.
        verbose (bool): If True, prints Stan's optimization progress. Defaults to False.
        output_path (str, optional): If provided, saves the fit_results dictionary 
                                     to this path using pickle.

    Returns:
        dict: The 'fit' dictionary containing all embedding results.
    """
    # Ensure the matrix is a numpy array of the correct type (float)
    if not isinstance(dmat, np.ndarray) or dmat.dtype != float:
        dmat = np.array(dmat, dtype=float)

    if dmat.ndim != 2 or dmat.shape[0] != dmat.shape[1]:
        raise ValueError(f"Input matrix must be square. Got shape: {dmat.shape}")

    # --- Pre-processing (as seen in tst.py) ---
    # It's often a good idea to normalize the distance matrix.
    # The original paper and tst.py normalize it to have a max value of 2.0.
    print("Normalizing distance matrix...")
    dmat = 2.0 * dmat / np.max(dmat)

    # --- Run the Embedding ---
    print(f"Starting HMDS embedding into {embedding_dim} dimensions...")
    # This step can take several minutes, especially the first time.
    if verbose:
        fit = HMDS.embed(embedding_dim, dmat, dij_unc=dmat_unc)
    else:
        # Suppress the C++ output from Stan by redirecting stdout and stderr
        print("Stan optimization output is suppressed. This may take a while...")
        f = io.StringIO()
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            fit = HMDS.embed(embedding_dim, dmat, dij_unc=dmat_unc)
    
    # --- Print Results ---
    print("\n--- Embedding Complete ---")
    print(f"Fitted curvature scale parameter (lambda): {fit['lambda']:.4f}")
    print(f"Embedding contains {len(fit['euc'])} points.")
    print("Results are available in the 'fit' dictionary.")

    # --- Calculate BIC ---
    n_params = dmat.shape[0] * embedding_dim + dmat.shape[0] + 1 - embedding_dim*(embedding_dim - 1)/2 # N*D + N + 1 - D*(D-1)/2
    fit['bic'] = calculate_bic(fit['dmat'], fit['emb_mat'], n_params, 
                               is_hyperbolic=True, lambda_val=fit['lambda'], 
                               sig_vals=fit['sig'], dmat_unc=dmat_unc)

    # --- Save Results to Disk ---
    if output_path:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f_out:
                pickle.dump(fit, f_out)
            print(f"\nEmbedding results successfully saved to: {output_path}")
        except Exception as e:
            print(f"\nWarning: Could not save results to '{output_path}'. Error: {e}")

    return fit

def run_euclidean_embedding(dmat, embedding_dim, dmat_unc=None):
    """
    Performs classical multidimensional scaling (MDS) for a Euclidean embedding.

    Args:
        dmat (np.ndarray): The input distance matrix.
        embedding_dim (int): The target dimension for the Euclidean embedding.
        dmat_unc (np.ndarray, optional): The matrix of uncertainties for each distance.

    Returns:
        dict or None: A dictionary containing the embedding results, or None if
                      scikit-learn is not installed.
    """
    if MDS is None:
        print("\nWarning: scikit-learn is not installed. Skipping Euclidean MDS. Please run 'pip install scikit-learn'.")
        return None

    print(f"\nStarting Euclidean MDS embedding into {embedding_dim} dimensions...")

    # Normalize the distance matrix for fair comparison with hyperbolic results
    dmat = 2.0 * dmat / np.max(dmat)

    # scikit-learn's MDS expects dissimilarities, which is what dmat is.
    # We use metric=True for classical MDS.
    mds = MDS(n_components=embedding_dim, dissimilarity='precomputed', metric=True,
              random_state=0, normalized_stress=False)

    # Fit the model and get the embedded coordinates
    coords = mds.fit_transform(dmat)

    # Calculate the pairwise distances in the new Euclidean embedding
    embedded_dmat = pairwise_distances(coords, metric='euclidean')

    # --- Calculate BIC ---
    n_params = dmat.shape[0] * embedding_dim + 1 # N*D + 1 (for variance)
    bic = calculate_bic(dmat, embedded_dmat, n_params, dmat_unc=dmat_unc)


    return {'dmat': dmat, 'emb_mat': embedded_dmat, 'coords': coords, 'bic': bic}


if __name__ == '__main__':
    # --- Code Quality & Clarity Suggestion: Using argparse ---
    # This makes the script more reusable by allowing you to pass file paths
    # and parameters from the command line instead of editing the code.
    parser = argparse.ArgumentParser(description="Run Hyperbolic MDS on a distance matrix from a .mat file.")
    parser.add_argument("mat_file", help="Path to the input .mat file.")
    parser.add_argument("matrix_name", help="Name of the distance matrix variable within the .mat file.")
    parser.add_argument("--unc-name", help="Name of the distance uncertainty matrix variable within the .mat file (optional).")
    parser.add_argument("-d", "--dim", type=int, default=3, help="Target dimension for the embedding (default: 3).")
    parser.add_argument("--no-plot", action="store_true", help="Suppress the Shepard diagram plot.")
    parser.add_argument("-o", "--output", help="Output file prefix to save plots as PDF files instead of displaying them.")

    args = parser.parse_args()
    
    try:
        # Command-line workflow
        print(f"Loading data from: {args.mat_file}")
        distance_matrix = load_dmat_from_mat(args.mat_file, args.matrix_name)
        distance_unc_matrix = None
        if args.unc_name:
            print(f"Loading uncertainty data from variable: {args.unc_name}")
            distance_unc_matrix = load_dmat_from_mat(args.mat_file, args.unc_name)
        
        # --- Hyperbolic Embedding ---
        fit_results = run_embedding(distance_matrix, args.dim, 
                                    dmat_unc=distance_unc_matrix, 
                                    verbose=True)
        print(f"Hyperbolic Embedding BIC: {fit_results['bic']:.2f}")

        if not args.no_plot:
            # Shepard diagram for Hyperbolic embedding
            plot_shepard_diagram(fit_results['dmat'], fit_results['emb_mat'],
                                 fit_results['lambda'], title="Shepard Diagram: Hyperbolic Embedding",
                                 output_file=f"{args.output}_hyperbolic_shepard.pdf" if args.output else None)
            # Visualize the embedding, using PCA if dimension > 3
            visualize_embedding(fit_results, output_prefix=args.output)

        # --- Euclidean Embedding ---
        euclidean_results = run_euclidean_embedding(distance_matrix, args.dim)
        print(f"Euclidean Embedding BIC: {euclidean_results['bic']:.2f}")

        if euclidean_results and not args.no_plot:
            # Shepard diagram for Euclidean embedding
            plot_shepard_diagram(euclidean_results['dmat'], euclidean_results['emb_mat'],
                                 title="Shepard Diagram: Euclidean Embedding",
                                 output_file=f"{args.output}_euclidean_shepard.pdf" if args.output else None)

    except MatReadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)