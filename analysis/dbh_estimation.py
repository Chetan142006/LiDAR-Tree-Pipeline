"""
Forestry LiDAR Analysis Pipeline - DBH Estimation Module.

This module estimates the Diameter at Breast Height (DBH) of a tree trunk.
It extracts a horizontal slice at 1.3 meters above the ground and fits a 2D
circle using an algebraic least-squares algorithm with robust outlier rejection.
"""

import numpy as np
from typing import Tuple, Dict, Any


def fit_circle_2d(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits a 2D circle to a set of points (x, y) using algebraic least squares.
    Equation: (x - xc)^2 + (y - yc)^2 = R^2

    Parameters
    ----------
    x : np.ndarray
        1D array of X coordinates.
    y : np.ndarray
        1D array of Y/Z coordinates.

    Returns
    -------
    Tuple[float, float, float]
        - xc: center X coordinate.
        - yc: center Y/Z coordinate.
        - R: radius of the fitted circle.
    """
    # Linearized system: 2*x*xc + 2*y*yc + (R^2 - xc^2 - yc^2) = x^2 + y^2
    # Let u = [2*xc, 2*yc, R^2 - xc^2 - yc^2]^T
    # A * u = b
    A = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2

    # Solve least squares
    u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    xc = u[0] / 2.0
    yc = u[1] / 2.0
    R = np.sqrt(u[2] + xc**2 + yc**2)

    return xc, yc, R


def estimate_dbh(
    stem_points: np.ndarray,
    vertical_axis: str = "y",
    target_height: float = 1.3,
    slice_thickness: float = 0.15
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimates the DBH of the tree trunk by horizontal slicing and circle fitting.

    Parameters
    ----------
    stem_points : np.ndarray
        Shape (N, 3), coordinate arrays of segmented wood/stem.
    vertical_axis : str, optional
        The vertical coordinate axis ("x", "y", or "z"). Default is "y".
    target_height : float, optional
        The height at which to measure DBH (meters). Standard is 1.3m.
    slice_thickness : float, optional
        The vertical thickness of the horizontal slice (meters).

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        - dbh: Estimated diameter in meters (2 * R).
        - center: 2D coordinates [xc, yc] of the trunk center on the horizontal plane.
        - inliers: subset of points (M, 2) that were used in the final circle fit.
    """
    # Map vertical axis index
    v_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis.lower()]
    h_indices = [i for i in [0, 1, 2] if i != v_idx]

    # Ground height is assumed to be the minimum vertical coordinate of the stem points
    ground_y = stem_points[:, v_idx].min()
    absolute_target = ground_y + target_height

    # Extract points inside the slice
    h_min = absolute_target - (slice_thickness / 2.0)
    h_max = absolute_target + (slice_thickness / 2.0)

    slice_mask = (stem_points[:, v_idx] >= h_min) & (stem_points[:, v_idx] <= h_max)
    slice_pts = stem_points[slice_mask]

    # If too few points, expand the slice dynamically
    if len(slice_pts) < 15:
        print(f"Warning: Low point count in standard DBH slice. Expanding thickness to {slice_thickness * 2}m...")
        h_min = absolute_target - slice_thickness
        h_max = absolute_target + slice_thickness
        slice_mask = (stem_points[:, v_idx] >= h_min) & (stem_points[:, v_idx] <= h_max)
        slice_pts = stem_points[slice_mask]

    if len(slice_pts) < 5:
        raise ValueError(f"Insufficient points ({len(slice_pts)}) in the trunk slice to fit a DBH circle.")

    # 2D coordinates on the horizontal plane
    pts_2d = slice_pts[:, h_indices]

    # Robust Iterative Circle Fit:
    # 1. Fit initial circle
    xc, yc, R = fit_circle_2d(pts_2d[:, 0], pts_2d[:, 1])

    # 2. Calculate point distances from fitted circle radius (residuals)
    distances = np.sqrt((pts_2d[:, 0] - xc)**2 + (pts_2d[:, 1] - yc)**2)
    residuals = np.abs(distances - R)

    # 3. Filter out outliers (points further than 2.0 * standard deviation)
    std_dev = np.std(residuals)
    inlier_mask = residuals < np.maximum(0.02, 2.0 * std_dev) # at least 2cm tolerance
    inliers = pts_2d[inlier_mask]

    # Refit with only inliers if we threw away outliers
    if len(inliers) >= 5 and len(inliers) < len(pts_2d):
        xc, yc, R = fit_circle_2d(inliers[:, 0], inliers[:, 1])
    else:
        inliers = pts_2d

    dbh = 2.0 * R
    print(f"DBH Estimation Successful:")
    print(f"  Trunk center at Z={xc:.3f}, X={yc:.3f} (on horizontal plane)")
    print(f"  Estimated DBH: {dbh:.3f} meters ({dbh * 100:.1f} cm)")
    
    return dbh, np.array([xc, yc]), inliers


if __name__ == "__main__":
    # Test circular fit
    theta = np.linspace(0, 2*np.pi, 100)
    test_x = 0.5 * np.cos(theta) + 0.05 * np.random.randn(100)
    test_y = 0.5 * np.sin(theta) + 0.05 * np.random.randn(100)
    xc, yc, R = fit_circle_2d(test_x, test_y)
    print(f"Test Circle: Center ({xc:.3f}, {yc:.3f}), Radius: {R:.3f} (True R=0.500)")
