"""
Forestry LiDAR Analysis Pipeline - Canopy Analysis Module.

This module computes key structural canopy parameters including:
1. Canopy Height (vertical crown length)
2. Canopy Spread (horizontal dimensions along primary axes)
3. 2D Canopy Projection Area (using 2D Convex Hull)
4. 3D Canopy Volume (using 3D Convex Hull)
"""

import numpy as np
from typing import Dict, Any, Tuple

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None


def analyze_canopy(
    crown_points: np.ndarray,
    vertical_axis: str = "y"
) -> Dict[str, Any]:
    """
    Computes statistical and spatial structural properties of the tree crown.

    Parameters
    ----------
    crown_points : np.ndarray
        Shape (K, 3), coordinate arrays of foliage/leaves.
    vertical_axis : str, optional
        The vertical coordinate axis ("x", "y", or "z"). Default is "y".

    Returns
    -------
    dict
        A dictionary containing:
        - "canopy_height": float, total vertical length of foliage in meters.
        - "canopy_spread_axis1": float, spread along first horizontal axis (meters).
        - "canopy_spread_axis2": float, spread along second horizontal axis (meters).
        - "canopy_volume_m3": float, 3D envelope volume.
        - "projection_area_m2": float, 2D ground coverage area.
        - "hull_vertices": numpy.ndarray, coordinates of the hull boundary (for visualization).
    """
    if len(crown_points) < 4:
        raise ValueError(f"Insufficient foliage points ({len(crown_points)}) to perform canopy hull analysis.")

    # Determine vertical and horizontal coordinate indices
    v_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis.lower()]
    h_indices = [i for i in [0, 1, 2] if i != v_idx]

    # 1. Height Analysis
    min_v = crown_points[:, v_idx].min()
    max_v = crown_points[:, v_idx].max()
    canopy_height = max_v - min_v

    # 2. Spread Analysis
    spread_h1 = crown_points[:, h_indices[0]].max() - crown_points[:, h_indices[0]].min()
    spread_h2 = crown_points[:, h_indices[1]].max() - crown_points[:, h_indices[1]].min()

    # 3. 3D Volume and 2D Projection Area Calculations (with SciPy ConvexHull)
    use_scipy = (ConvexHull is not None)
    volume_m3 = 0.0
    area_m2 = 0.0
    hull_pts_3d = None

    if use_scipy:
        try:
            # 3D Convex Hull for volume
            hull_3d = ConvexHull(crown_points)
            volume_m3 = float(hull_3d.volume)
            # Store points of the 3D hull vertices for plotting
            hull_pts_3d = crown_points[hull_3d.vertices]

            # 2D horizontal projection Convex Hull for projection area
            pts_2d = crown_points[:, h_indices]
            # Remove duplicate coordinate projections to avoid collinear errors
            pts_2d_unique = np.unique(pts_2d, axis=0)
            
            if len(pts_2d_unique) >= 3:
                hull_2d = ConvexHull(pts_2d_unique)
                area_m2 = float(hull_2d.volume) # in 2D, hull.volume represents the 2D area
            else:
                area_m2 = spread_h1 * spread_h2 * 0.785 # elliptical approximation fallback
        except Exception as e:
            print(f"Warning: SciPy ConvexHull calculation encountered a numerical error: {e}. Falling back to bounding ellipsoid.")
            use_scipy = False
    
    # Fallback to Ellipsoid approximations if SciPy is missing or failed
    if not use_scipy:
        print("Warning: Using bounding ellipsoid approximations for canopy volume/area...")
        # Ellipsoid Volume: 4/3 * pi * a * b * c
        a = spread_h1 / 2.0
        b = spread_h2 / 2.0
        c = canopy_height / 2.0
        volume_m3 = (4.0 / 3.0) * np.pi * a * b * c * 0.75 # multiplied by density factor
        
        # Ellipse Area: pi * a * b
        area_m2 = np.pi * a * b

    results = {
        "canopy_height": canopy_height,
        "canopy_spread_axis1": spread_h1,
        "canopy_spread_axis2": spread_h2,
        "canopy_volume_m3": volume_m3,
        "projection_area_m2": area_m2,
        "hull_vertices": hull_pts_3d
    }

    print("Canopy Structure Metrics:")
    print(f"  Canopy Height: {results['canopy_height']:.2f} meters")
    print(f"  Horizontal Spread: {results['canopy_spread_axis1']:.2f}m x {results['canopy_spread_axis2']:.2f}m")
    print(f"  Estimated Canopy Volume (Hull): {results['canopy_volume_m3']:.2f} m³")
    print(f"  Projected Ground Coverage Area: {results['projection_area_m2']:.2f} m²")

    return results


if __name__ == "__main__":
    # Test canopy analysis on mock spherical crown
    phi = np.random.uniform(0, 2*np.pi, 500)
    theta = np.random.uniform(0, np.pi, 500)
    r = np.random.uniform(0.5, 1.5, 500)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi) + 5.0 # elevation
    z = r * np.cos(theta)
    
    mock_crown = np.column_stack((x, y, z))
    analyze_canopy(mock_crown, "y")
