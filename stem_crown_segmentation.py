"""
Forestry LiDAR Analysis Pipeline - Stem/Crown Segmentation Module.

This module splits a tree point cloud into two distinct semantic groups:
1. Stem (trunk, limbs, major branches - Class 1)
2. Crown (foliage, leaves, canopy - Class 0)
It supports explicit class label extraction and a spatial heuristic fallback.
"""

import numpy as np
from typing import Tuple, Dict, Any


def segment_stem_and_crown(
    points: np.ndarray, 
    labels: np.ndarray,
    vertical_axis: str = "y"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments the point cloud into stem and crown points.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3), coordinate arrays of the tree.
    labels : np.ndarray
        Shape (N,), point classifications. Class 1 = Wood, Class 0 = Leaf.
    vertical_axis : str, optional
        The vertical coordinate axis of the tree ("x", "y", or "z"). Default is "y".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - stem_points: np.ndarray of shape (M, 3)
        - crown_points: np.ndarray of shape (K, 3)
    """
    # Map vertical axis index
    v_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis.lower()]
    h_indices = [i for i in [0, 1, 2] if i != v_idx]

    # Check if we have valid classification labels from the generator
    unique_labels = np.unique(labels)
    if 1 in unique_labels:
        print("Explicit semantic classifications found. Segmenting via class labels...")
        stem_mask = (labels == 1)
        crown_mask = (labels == 0)
        return points[stem_mask], points[crown_mask]

    # Unsupervised Fallback Segmentation:
    # If no classification is present, apply a distance-to-axis spatial classification heuristic.
    print("Warning: No class labels found. Initiating unsupervised spatial fallback segmentation...")
    
    # 1. Identify coordinate boundaries
    min_val = points[:, v_idx].min()
    max_val = points[:, v_idx].max()
    height = max_val - min_val

    # 2. Slice the base of the trunk (lowest 15% of the height) to find the trunk core coordinate
    base_mask = (points[:, v_idx] < (min_val + 0.15 * height))
    base_pts = points[base_mask]

    if len(base_pts) > 0:
        trunk_center = np.median(base_pts[:, h_indices], axis=0)
    else:
        trunk_center = np.median(points[:, h_indices], axis=0)

    # 3. Spatial Class Separation:
    # Points near the trunk center axis are classified as stem. As height increases,
    # the canopy flares outwards, so we use a conical envelope where the stem radius
    # tapers, and any far outlying points are marked as crown.
    dist_to_axis = np.linalg.norm(points[:, h_indices] - trunk_center, axis=1)
    
    # Heuristic boundary: within 0.35m radius at base, tapering slowly with height
    norm_height = (points[:, v_idx] - min_val) / height
    stem_threshold = np.maximum(0.08, 0.40 * (1.0 - 0.7 * norm_height))

    stem_mask = (dist_to_axis < stem_threshold) & (norm_height < 0.85)
    crown_mask = ~stem_mask

    stem_points = points[stem_mask]
    crown_points = points[crown_mask]

    print(f"Fallback segmentation complete:")
    print(f"  Stem: {len(stem_points)} points")
    print(f"  Crown: {len(crown_points)} points")

    return stem_points, crown_points


if __name__ == "__main__":
    # Test fallback
    mock_points = np.random.uniform(-2, 2, (1000, 3))
    mock_labels = np.zeros(1000, dtype=np.int32)
    s, c = segment_stem_and_crown(mock_points, mock_labels, "y")
