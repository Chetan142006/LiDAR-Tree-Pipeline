"""
Forestry LiDAR Analysis Pipeline - Crown Base Height (CBH) Module.

This module estimates the Crown Base Height (CBH) of a tree, which is the height
above the ground where the living foliage crown begins. It uses a vertical density
profiling method to handle stray points or noise robustly.
"""

import numpy as np
from typing import Tuple


def estimate_crown_base_height(
    points: np.ndarray,
    classification: np.ndarray,
    vertical_axis: str = "y",
    bin_size: float = 0.1,
    noise_threshold_ratio: float = 0.015
) -> float:
    """
    Estimates the Crown Base Height (CBH) above ground level.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3), coordinate arrays of the entire tree.
    classification : np.ndarray
        Shape (N,), point classifications (Class 0 = Leaves).
    vertical_axis : str, optional
        The vertical coordinate axis ("x", "y", or "z"). Default is "y".
    bin_size : float, optional
        Vertical height thickness of the bins (meters) for density profiling.
    noise_threshold_ratio : float, optional
        The minimum proportion of leaf points (e.g. 1.5%) a height bin must contain
        to be considered the true start of the canopy (filtering out noise/strays).

    Returns
    -------
    float
        The estimated Crown Base Height (CBH) in meters above ground level.
    """
    v_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis.lower()]

    # Ground elevation is the lowest vertical stem point coordinate
    ground_y = points[:, v_idx].min()

    # Extract vertical coordinates of leaf points
    leaf_mask = (classification == 0)
    leaf_v = points[leaf_mask][:, v_idx]

    if len(leaf_v) == 0:
        print("Warning: No leaf points found. Crown Base Height is undefined.")
        return 0.0

    # 1. Direct minimum height approach (as reference baseline)
    min_leaf_y = leaf_v.min()
    cbh_baseline = min_leaf_y - ground_y

    # 2. Robust Vertical Density Profiling
    # Bin the vertical coordinates of the leaves
    num_bins = int(np.ceil((leaf_v.max() - leaf_v.min()) / bin_size))
    if num_bins <= 1:
        return cbh_baseline

    counts, bin_edges = np.histogram(leaf_v, bins=num_bins)
    total_leaves = len(leaf_v)

    # Find the first bin starting from the lowest height that exceeds the noise threshold
    threshold_count = int(noise_threshold_ratio * total_leaves)
    
    cbh_absolute = min_leaf_y
    for i, count in enumerate(counts):
        if count >= threshold_count:
            # The canopy base is estimated at the lower boundary of this bin
            cbh_absolute = bin_edges[i]
            break

    cbh = cbh_absolute - ground_y
    
    # Bound check
    cbh = np.clip(cbh, 0.0, leaf_v.max() - ground_y)

    print(f"Crown Base Height (CBH) Estimation Successful:")
    print(f"  Ground Elevation: {ground_y:.3f} meters")
    print(f"  Lowest Leaf Elevation: {min_leaf_y:.3f} meters")
    print(f"  Estimated CBH (Density Filtered): {cbh:.3f} meters ({cbh * 100:.1f} cm)")
    
    return float(cbh)


if __name__ == "__main__":
    # Test CBH estimation
    # Mock ground at 0.0
    mock_stems = np.column_stack((np.zeros(100), np.linspace(0, 10, 100), np.zeros(100)))
    # Leaves start at 3.5m, plus a stray outlier leaf at 1.0m
    leaf_heights = np.concatenate([[1.0], np.random.uniform(3.5, 10.0, 99)])
    mock_leaves = np.column_stack((np.random.normal(0, 1, 100), leaf_heights, np.random.normal(0, 1, 100)))
    
    all_pts = np.vstack((mock_stems, mock_leaves))
    all_classes = np.concatenate([np.ones(100), np.zeros(100)]) # 1 for wood, 0 for leaf
    
    estimate_crown_base_height(all_pts, all_classes, "y")
