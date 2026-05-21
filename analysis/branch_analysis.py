"""
Forestry LiDAR Analysis Pipeline - Branch Analysis Module.

This module extracts and analyzes branch structures from the stem point cloud.
By grouping branch points above the trunk and using Principal Component Analysis
(PCA via Singular Value Decomposition), we fit 3D direction vectors to estimate
bifurcation/branching angles relative to the vertical trunk axis.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def fit_pca_direction(points: np.ndarray) -> np.ndarray:
    """
    Fits the principal direction vector of a 3D point cloud using SVD/PCA.

    Parameters
    ----------
    points : np.ndarray
        Shape (M, 3), coordinate points of a branch segment.

    Returns
    -------
    np.ndarray
        The 3D unit vector representing the primary direction of the branch.
    """
    # Center the points
    centered_pts = points - np.mean(points, axis=0)

    # Compute Singular Value Decomposition (SVD)
    _, _, vh = np.linalg.svd(centered_pts, full_matrices=False)
    
    # The first row of vh (first right singular vector) corresponds to the principal component
    direction = vh[0, :]
    return direction / np.linalg.norm(direction)


def analyze_branches(
    stem_points: np.ndarray,
    vertical_axis: str = "y",
    trunk_length: float = 4.0
) -> List[Dict[str, Any]]:
    """
    Identifies major branch structures and calculates their branching angles.

    Parameters
    ----------
    stem_points : np.ndarray
        Shape (N, 3), coordinate arrays of segmented wood/stem.
    vertical_axis : str, optional
        The vertical coordinate axis ("x", "y", or "z"). Default is "y".
    trunk_length : float, optional
        Height threshold above which branching starts (meters).

    Returns
    -------
    List[dict]
        A list of analyzed branches, each containing:
        - "center": 3D center point of the branch.
        - "direction": 3D unit direction vector.
        - "angle_deg": branching angle in degrees relative to the vertical axis.
        - "point_count": number of points representing this branch.
    """
    v_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis.lower()]
    h_indices = [i for i in [0, 1, 2] if i != v_idx]

    # 1. Establish trunk reference axis vector
    trunk_vector = np.zeros(3)
    trunk_vector[v_idx] = 1.0

    # 2. Extract branch points (stem points above clear trunk height)
    ground_y = stem_points[:, v_idx].min()
    crown_stem_mask = stem_points[:, v_idx] > (ground_y + trunk_length * 0.9)
    branch_pts = stem_points[crown_stem_mask]

    if len(branch_pts) < 30:
        print("Warning: Insufficient stem points above the main trunk to analyze lateral branches.")
        return []

    # 3. Simple spatial grouping of branch points:
    # Separate points into angular sectors around the trunk axis to isolate limbs.
    trunk_center = np.median(stem_points[stem_points[:, v_idx] < (ground_y + 1.0)][:, h_indices], axis=0)
    centered_h_pts = branch_pts[:, h_indices] - trunk_center
    
    # Compute horizontal angles (azimuth)
    angles_rad = np.arctan2(centered_h_pts[:, 1], centered_h_pts[:, 0])
    
    # Divide into 4 quadrants to segment primary structural branches
    quadrants = [
        ("North-East", (angles_rad >= 0) & (angles_rad < np.pi/2)),
        ("North-West", (angles_rad >= np.pi/2) & (angles_rad <= np.pi)),
        ("South-West", (angles_rad >= -np.pi) & (angles_rad < -np.pi/2)),
        ("South-East", (angles_rad >= -np.pi/2) & (angles_rad < 0))
    ]

    analyzed_limbs = []
    
    for name, mask in quadrants:
        limb_pts = branch_pts[mask]
        
        # Ensure we have a significant cluster to prevent noise PCA fitting
        if len(limb_pts) >= 15:
            # Fit primary orientation vector using PCA
            direction = fit_pca_direction(limb_pts)
            
            # Ensure branch vector points upwards/outwards (positive along vertical axis)
            if direction[v_idx] < 0:
                direction = -direction

            # Calculate branching angle relative to the trunk axis
            dot_product = np.dot(direction, trunk_vector)
            # Clip dot product to handle floating-point precision bounds
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            analyzed_limbs.append({
                "quadrant": name,
                "center": np.mean(limb_pts, axis=0),
                "direction": direction,
                "angle_deg": angle_deg,
                "point_count": len(limb_pts)
            })

    print(f"Branch Analysis complete: Detected {len(analyzed_limbs)} major structural branch systems.")
    for limb in analyzed_limbs:
        print(f"  Branch System [{limb['quadrant']}]:")
        print(f"    Points: {limb['point_count']}")
        print(f"    3D Vector: [{limb['direction'][0]:.2f}, {limb['direction'][1]:.2f}, {limb['direction'][2]:.2f}]")
        print(f"    Bifurcation Angle: {limb['angle_deg']:.1f}°")

    return analyzed_limbs


if __name__ == "__main__":
    # Test SVD directional fitting on artificial slanted line points
    t = np.linspace(-1, 1, 100)
    # Slanted branch along Y and X
    x = t * np.sin(np.radians(30))
    y = t * np.cos(np.radians(30))
    z = np.zeros_like(t)
    mock_limb = np.column_stack((x, y, z)) + 0.02 * np.random.randn(100, 3)
    
    dir_vec = fit_pca_direction(mock_limb)
    angle = np.degrees(np.arccos(dir_vec[1]))
    print(f"PCA Fitted Direction: {dir_vec}, Bifurcation Angle: {angle:.2f}° (True: 30.00°)")
