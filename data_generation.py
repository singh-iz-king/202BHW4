import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def generate_er_graphical_model_data(n, p, prob, rng):
    """
    Generates data from a Gaussian Graphical Model based on an Erdős-Rényi graph.

    Parameters:
    n (int): Number of samples
    p (int): Number of nodes (variables)
    prob (float): Probability of edge creation
    seed (int): Random seed for reproducibility

    Returns:
    dict: Contains 'data', 'precision_matrix', and 'adjacency_matrix'
    """
    seed = 42
    np.random.seed(seed)

    # Generate a random Erdős-Rényi graph
    # NetworkX uses (n, p) notation where n=nodes, p=probability
    G = nx.erdos_renyi_graph(n=p, p=prob, seed=seed)
    adjacency_matrix = nx.to_numpy_array(G)

    # Construct a symmetric precision matrix with random weights
    random_weights = np.random.uniform(low=-0.5, high=0.5, size=(p, p))

    # Symmetrize the weights: (W + W.T) / 2
    symmetric_weights = (random_weights + random_weights.T) / 2

    # Apply the adjacency mask to the weights
    precision_matrix = adjacency_matrix * symmetric_weights

    # Ensure positive definiteness by adjusting the diagonal (Diagonal dominance)
    # diag(P) = sum(|row values|) + 0.01
    row_sums = np.sum(np.abs(precision_matrix), axis=1)
    np.fill_diagonal(precision_matrix, row_sums + 1)

    # Compute the covariance matrix as the inverse of the precision matrix
    covariance_matrix = np.linalg.inv(precision_matrix)

    # Generate multivariate normal data
    mean_vector = np.zeros(p)
    data = np.random.multivariate_normal(mean_vector, covariance_matrix, size=n)

    return {
        "data": data,
        "precision_matrix": precision_matrix,
        "adjacency_matrix": adjacency_matrix,
    }


def generate_data(n, p, q, Omega_density, B_density, rng):
    result = generate_er_graphical_model_data(n, p, Omega_density, rng)

    E = result["data"]
    Omega = result["precision_matrix"]

    B = np.zeros((q, p))
    mask = rng.random((q, p)) < B_density
    signs = rng.choice([-1.0, 1.0], size=(q, p))
    mags = rng.uniform(0.5, 2.0, size=(q, p))
    B[mask] = (signs * mags)[mask]

    X = rng.standard_normal((n, q))

    X = X - X.mean(axis=0, keepdims=True)

    Y = X @ B + E
    Y = Y - Y.mean(axis=0, keepdims=True)

    return {"Y": Y, "X": X, "B": B, "Omega": Omega}


def plot_precision_matrix(precision_matrix, problem, B=True):
    """
    Visualizes the precision matrix as a heatmap.
    """
    plt.figure(figsize=(8, 6))

    # Create a heatmap matching the R code's style (Blue-White-Red)
    sns.heatmap(
        precision_matrix,
        cmap="bwr",  # Blue-White-Red colormap
        center=0,  # Center the colormap at 0
        annot=False,  # Turn off number annotations for cleaner look
        square=True,  # Force square cells
        cbar_kws={"label": "Precision"},
    )

    title = f"Difference Heatmap for {"Beta" if B else "Omega"} for problem: {problem}"
    plt.title(title)
    plt.xlabel("Variable Index")
    plt.ylabel("Variable Index")
    plt.savefig(title.strip())
