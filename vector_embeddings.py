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
        
        # Verify the expected columns are present
        expected_columns = ['group_tag', 'block_tag', 'block_id', 'position']
        vector_columns = [f'v{i}' for i in range(1536)]  # v0 through v1535
        
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


def compute_centroid(embeddings_df: pd.DataFrame, 
                    group_filter: Optional[str] = None,
                    block_filter: Optional[str] = None) -> np.ndarray:
    """
    Compute the centroid (mean vector) of the embeddings.
    
    Args:
        embeddings_df: DataFrame containing the embeddings
        group_filter: Optional filter for specific group_tag
        block_filter: Optional filter for specific block_tag
        
    Returns:
        Numpy array representing the centroid vector
    """
    # Apply filters if specified
    filtered_df = embeddings_df
    if group_filter:
        filtered_df = filtered_df[filtered_df['group_tag'] == group_filter]
    if block_filter:
        filtered_df = filtered_df[filtered_df['block_tag'] == block_filter]
    
    # Get only the vector columns (v0, v1, ...)
    vector_columns = [col for col in filtered_df.columns if col.startswith('v')]
    
    if not vector_columns:
        raise ValueError("No vector columns found in the DataFrame")
    
    # Compute the mean of each vector dimension
    centroid = filtered_df[vector_columns].mean().values
    
    print(f"Computed centroid from {len(filtered_df)} vectors with dimension {len(centroid)}")
    return centroid


def compute_variance(embeddings_df: pd.DataFrame, 
                    centroid: Optional[np.ndarray] = None,
                    group_filter: Optional[str] = None,
                    block_filter: Optional[str] = None) -> Tuple[float, np.ndarray]:
    """
    Compute the variance of embeddings from the centroid.
    
    Args:
        embeddings_df: DataFrame containing the embeddings
        centroid: Optional pre-computed centroid. If None, it will be computed
        group_filter: Optional filter for specific group_tag
        block_filter: Optional filter for specific block_tag
        
    Returns:
        Tuple containing:
            - Total variance (scalar)
            - Per-dimension variance (numpy array)
    """
    # Apply filters if specified
    filtered_df = embeddings_df
    if group_filter:
        filtered_df = filtered_df[filtered_df['group_tag'] == group_filter]
    if block_filter:
        filtered_df = filtered_df[filtered_df['block_tag'] == block_filter]
    
    # Get only the vector columns
    vector_columns = [col for col in filtered_df.columns if col.startswith('v')]
    
    if not vector_columns:
        raise ValueError("No vector columns found in the DataFrame")
    
    # Compute centroid if not provided
    if centroid is None:
        centroid = compute_centroid(filtered_df)
    
    # Extract vectors as a numpy array
    vectors = filtered_df[vector_columns].values
    
    # Compute squared differences from centroid
    squared_diffs = np.square(vectors - centroid)
    
    # Compute variance per dimension
    per_dim_variance = np.mean(squared_diffs, axis=0)
    
    # Compute total variance (mean of per-dimension variances)
    total_variance = np.mean(per_dim_variance)
    
    print(f"Computed variance: {total_variance:.6f} from {len(filtered_df)} vectors")
    return total_variance, per_dim_variance


def find_closest_vectors(embeddings_df: pd.DataFrame, 
                         target_vector: np.ndarray,
                         n: int = 5) -> pd.DataFrame:
    """
    Find the n closest vectors to a target vector using cosine similarity.
    
    Args:
        embeddings_df: DataFrame containing the embeddings
        target_vector: The vector to compare against
        n: Number of closest vectors to return
        
    Returns:
        DataFrame with the n closest vectors and their similarity scores
    """
    # Get only the vector columns
    vector_columns = [col for col in embeddings_df.columns if col.startswith('v')]
    
    if not vector_columns:
        raise ValueError("No vector columns found in the DataFrame")
    
    # Normalize the target vector
    target_norm = np.linalg.norm(target_vector)
    if target_norm > 0:
        normalized_target = target_vector / target_norm
    else:
        normalized_target = target_vector
    
    # Compute cosine similarity for each vector
    similarities = []
    
    for idx, row in embeddings_df.iterrows():
        vector = row[vector_columns].values
        
        # Normalize the vector
        vector_norm = np.linalg.norm(vector)
        if vector_norm > 0:
            normalized_vector = vector / vector_norm
        else:
            normalized_vector = vector
        
        # Compute cosine similarity
        similarity = np.dot(normalized_target, normalized_vector)
        similarities.append((idx, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top n results
    top_indices = [idx for idx, _ in similarities[:n]]
    result_df = embeddings_df.loc[top_indices].copy()
    
    # Add similarity scores
    result_df['similarity'] = [sim for _, sim in similarities[:n]]
    
    return result_df


def group_statistics(embeddings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each group in the embeddings.
    
    Args:
        embeddings_df: DataFrame containing the embeddings
        
    Returns:
        DataFrame with statistics for each group
    """
    # Check if group_tag column exists
    if 'group_tag' not in embeddings_df.columns:
        raise ValueError("group_tag column not found in the DataFrame")
    
    # Get unique groups
    groups = embeddings_df['group_tag'].unique()
    
    results = []
    for group in groups:
        group_df = embeddings_df[embeddings_df['group_tag'] == group]
        
        # Compute centroid and variance for this group
        centroid = compute_centroid(group_df)
        total_variance, _ = compute_variance(group_df, centroid)
        
        # Count blocks in this group
        if 'block_tag' in group_df.columns:
            unique_blocks = group_df['block_tag'].nunique()
        else:
            unique_blocks = 0
        
        results.append({
            'group_tag': group,
            'vector_count': len(group_df),
            'unique_blocks': unique_blocks,
            'variance': total_variance
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process vector embeddings from CSV files")
    parser.add_argument("csv_file", help="Path to the CSV file containing embeddings")
    parser.add_argument("--group", help="Filter by group_tag")
    parser.add_argument("--block", help="Filter by block_tag")
    parser.add_argument("--stats", action="store_true", help="Show group statistics")
    
    args = parser.parse_args()
    
    try:
        # Load the embeddings
        embeddings = load_embeddings(args.csv_file)
        
        # Show basic info
        print(f"\nLoaded {len(embeddings)} vectors from {args.csv_file}")
        
        if 'group_tag' in embeddings.columns:
            group_counts = embeddings['group_tag'].value_counts()
            print(f"Found {len(group_counts)} unique groups")
        
        if 'block_tag' in embeddings.columns:
            block_counts = embeddings['block_tag'].value_counts()
            print(f"Found {len(block_counts)} unique blocks")
        
        # Compute centroid and variance
        centroid = compute_centroid(embeddings, args.group, args.block)
        total_variance, per_dim_variance = compute_variance(embeddings, centroid, args.group, args.block)
        
        print(f"\nResults:")
        print(f"Total variance: {total_variance:.6f}")
        print(f"Max dimension variance: {np.max(per_dim_variance):.6f}")
        print(f"Min dimension variance: {np.min(per_dim_variance):.6f}")
        
        # Show group statistics if requested
        if args.stats and 'group_tag' in embeddings.columns:
            print("\nGroup Statistics:")
            stats = group_statistics(embeddings)
            print(stats.sort_values('variance'))
        
    except Exception as e:
        print(f"Error: {e}")
