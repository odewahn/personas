"""
Vector Embeddings Processor
---------------------------
This module provides functionality to load and analyze vector embeddings from CSV files.
It supports computing centroids and variance of embedding vectors.

Usage:
    from vector_embeddings import load_embeddings, compute_centroid, compute_variance

    # Load embeddings from a CSV file
    embeddings_data = load_embeddings("embeddings.csv")

    # Compute centroid of all vectors
    centroid = compute_centroid(embeddings_data)

    # Compute variance of all vectors
    variance = compute_variance(embeddings_data, centroid)
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def load_embeddings(csv_file: str) -> pd.DataFrame:
    """
    Load vector embeddings from a CSV file.

    The CSV file should have columns:
    group_tag, block_tag, block_id, position, v0, v1, v2, ..., v1535

    Args:
        csv_file: Path to the CSV file containing embeddings

    Returns:
        DataFrame containing the embeddings data
    """
    print(f"Loading embeddings from {csv_file}...")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Embeddings file not found: {csv_file}")

    # Load the CSV file
    try:
        df = pd.read_csv(csv_file)

        print(f"Loaded {len(df)} rows from {csv_file}")

        # Verify the expected columns are present
        expected_columns = ["group_tag", "block_tag", "block_id", "position"]
        vector_columns = [f"v{i}" for i in range(1536)]  # v0 through v1535

        # Check for metadata columns
        missing_meta = [col for col in expected_columns if col not in df.columns]
        if missing_meta:
            print(f"Warning: Missing metadata columns: {missing_meta}")

        # Check for vector columns
        vector_cols_present = [col for col in vector_columns if col in df.columns]
        if not vector_cols_present:
            raise ValueError("No vector columns (v0, v1, ...) found in the CSV file")

        vector_dimension = len(vector_cols_present)
        print(f"Loaded {len(df)} embeddings with dimension {vector_dimension}")

        return df

    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise


def compute_centroid(embeddings_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the centroid (mean vector) of the embeddings.

    Args:
        embeddings_df: DataFrame containing the embeddings

    Returns:
        Numpy array representing the centroid vector
    """

    # Get only the vector columns (v0, v1, ...)
    vector_columns = [col for col in embeddings_df.columns if col.startswith("v")]

    if not vector_columns:
        raise ValueError("No vector columns found in the DataFrame")

    # Compute the mean of each vector dimension
    centroid = embeddings_df[vector_columns].mean().values

    print(
        f"Computed centroid from {len(embeddings_df)} vectors with dimension {len(centroid)}"
    )
    return centroid


def compute_variance(
    embeddings_df: pd.DataFrame,
    centroid: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute the variance of embeddings from the centroid.

    Args:
        embeddings_df: DataFrame containing the embeddings
        centroid: Optional pre-computed centroid. If None, it will be computed

    Returns:
        Tuple containing:
            - Total variance (scalar)
            - Per-dimension variance (numpy array)
    """
    # Get only the vector columns
    vector_columns = [col for col in embeddings_df.columns if col.startswith("v")]

    if not vector_columns:
        raise ValueError("No vector columns found in the DataFrame")

    # Compute centroid if not provided
    if centroid is None:
        centroid = compute_centroid(embeddings_df)

    # Extract vectors as a numpy array
    vectors = embeddings_df[vector_columns].values

    # Compute squared differences from centroid
    squared_diffs = np.square(vectors - centroid)

    # Compute variance per dimension
    per_dim_variance = np.mean(squared_diffs, axis=0)

    # Compute total variance (mean of per-dimension variances)
    total_variance = np.mean(per_dim_variance)

    print(f"Computed variance: {total_variance:.6f} from {len(embeddings_df)} vectors")
    return total_variance, per_dim_variance


def group_statistics(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each group in the embeddings.

    Args:
        embeddings_df: DataFrame containing the embeddings

    Returns:
        DataFrame with statistics for each group
    """
    stats = []
    for group, group_df in embeddings_df.groupby("group_tag"):
        group_centroid = compute_centroid(group_df)
        group_variance, _ = compute_variance(group_df, group_centroid)
        stats.append(
            {"group": group, "count": len(group_df), "variance": group_variance}
        )

    return pd.DataFrame(stats)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process vector embeddings from CSV files"
    )
    parser.add_argument("csv_file", help="Path to the CSV file containing embeddings")
    parser.add_argument("--stats", action="store_true", help="Show group statistics")

    args = parser.parse_args()

    try:
        # Load the embeddings
        embeddings = load_embeddings(args.csv_file)

        # Show basic info
        print(f"\nLoaded {len(embeddings)} vectors from {args.csv_file}")

        # Compute centroid and variance
        centroid = compute_centroid(embeddings)
        total_variance, per_dim_variance = compute_variance(embeddings, centroid)

        print(f"\nResults:")
        print(f"Total variance: {total_variance:.6f}")
        print(f"Max dimension variance: {np.max(per_dim_variance):.6f}")
        print(f"Min dimension variance: {np.min(per_dim_variance):.6f}")

        # Show group statistics if requested
        if args.stats and "group_tag" in embeddings.columns:
            print("\nGroup Statistics:")
            stats = group_statistics(embeddings)
            print(stats.sort_values("variance"))

    except Exception as e:
        print(f"Error: {e}")
