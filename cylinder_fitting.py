"""
Forestry LiDAR Analysis Pipeline - Cylinder Fitting Module.

This module fits 3D cylinder primitives to slices of the tree trunk.
By slicing the main stem at regular vertical intervals and fitting 2D circles,
we reconstruct the tree's cylindrical taper profile (height vs. radius).
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dbh_estimation import fit_circle_2d


def fit_trunk_cylinders(
    stem_points: np.ndarray,
    vertical_axis: str = "y",
    slice_height: float = 0.5,
    max_height_ratio: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Fits 3D cylinders along horizontal slices of the trunk to capture tapering.

    Parameters
    ----------
    stem_points : np.ndarray
        Shape (N, 3), coordinate arrays of segmented wood/stem.
    vertical_axis : str, optional
        The vertical coordinate axis ("x", "y", or "z"). Default is "y".
    slice_height : float, optional
        Vertical thickness of each cylinder slice in meters.
    max_height_ratio : float, optional
        Maximum height ratio of the tree up to which trunk cylinder fitting is done
        (to avoid fitting branches instead of the main trunk, default is 70% of tree height).

    Returns
    -------
    List[dict]
        A list of cylinder parameters, each dictionary containing:
        - "center_3d": numpy.ndarray of shape (3,) representing the 3D center of the cylinder segment.
        - "radius": float, fitted cylinder radius.
        - "height": float, height thickness of this segment.
        - "elevation": float, vertical center coordinate of this segment.
    """
    v_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis.lower()]
    h_indices = [i for i in [0, 1, 2] if i != v_idx]

    # Find boundaries
    min_v = stem_points[:, v_idx].min()
    max_v = stem_points[:, v_idx].max()
    total_height = max_v - min_v

    # Limit trunk cylinder fitting to the main trunk section
    limit_v = min_v + max_height_ratio * total_height

    cylinders = []
    
    # Slice trunk sequentially
    current_v = min_v
    while current_v + slice_height <= limit_v:
        v_start = current_v
        v_end = current_v + slice_height
        v_mid = (v_start + v_end) / 2.0

        # Extract points in this slice
        mask = (stem_points[:, v_idx] >= v_start) & (stem_points[:, v_idx] < v_end)
        slice_pts = stem_points[mask]

        # Ensure we have enough points for a stable circle fit
        if len(slice_pts) >= 10:
            pts_2d = slice_pts[:, h_indices]
            
            try:
                # Fit circle to XZ / XY slice
                xc, yc, R = fit_circle_2d(pts_2d[:, 0], pts_2d[:, 1])
                
                # Check for physical realism (trunk radius shouldn't exceed trunk base size)
                # Max radius should be reasonable, e.g. < 1.0 meter for standard trees
                if 0.01 < R < 1.0:
                    center_3d = np.zeros(3)
                    center_3d[v_idx] = v_mid
                    center_3d[h_indices[0]] = xc
                    center_3d[h_indices[1]] = yc

                    cylinders.append({
                        "center_3d": center_3d,
                        "radius": R,
                        "height": slice_height,
                        "elevation": v_mid - min_v
                    })
            except Exception:
                # Catch mathematical exceptions in circle fitting
                pass

        current_v += slice_height

    print(f"Cylinder Fitting complete: Successfully fitted {len(cylinders)} trunk cylinders.")
    print("Trunk Tapering Profile (Elevation vs. Radius):")
    for cyl in cylinders:
        print(f"  Height: {cyl['elevation']:.1f}m -> Radius: {cyl['radius']:.3f}m")

    return cylinders


if __name__ == "__main__":
    # Test cylinder fitting on a mock cylinder
    mock_theta = np.linspace(0, 2*np.pi, 50)
    mock_slices = []
    # Make a tapering cylinder
    for h in np.linspace(0, 3, 6):
        r = 0.25 - 0.03 * h
        x = r * np.cos(mock_theta)
        z = r * np.sin(mock_theta)
        y = np.full_like(x, h)
        mock_slices.append(np.column_stack((x, y, z)))
    
    mock_stem = np.vstack(mock_slices)
    cyls = fit_trunk_cylinders(mock_stem, "y", 0.5, 1.0)
