"""
Forestry LiDAR Analysis Pipeline - Visualization Dashboard Module.

This module renders an interactive multi-panel 3D/2D dashboard displaying:
1. 3D Semantic Classification (Stem vs. Crown)
2. 2D DBH horizontal circle fit at 1.3m elevation
3. 3D Canopy volume Convex Hull envelope
4. 3D Trunk taper cylinder reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from matplotlib.patches import Circle

try:
    from scipy.spatial import ConvexHull
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    ConvexHull = None
    Poly3DCollection = None


def render_forestry_dashboard(
    points: np.ndarray,
    classification: np.ndarray,
    dbh_val: float,
    dbh_center: np.ndarray,
    dbh_inliers: np.ndarray,
    canopy_results: Dict[str, Any],
    cylinders: List[Dict[str, Any]],
    vertical_axis: str = "y"
) -> None:
    """
    Renders an interactive, premium 2x2 multi-panel forestry analysis dashboard.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3), coordinate arrays of the entire tree.
    classification : np.ndarray
        Shape (N,), point classifications (Class 1 = Wood, Class 0 = Leaf).
    dbh_val : float
        Estimated Diameter at Breast Height in meters.
    dbh_center : np.ndarray
        Horizontal plane center [xc, yc] of the DBH circle.
    dbh_inliers : np.ndarray
        Horizontal coordinates of slice points used in DBH circle fitting.
    canopy_results : dict
        Output from analyze_canopy including projection area, volume, etc.
    cylinders : List[dict]
        Output list of dictionaries representing fitted trunk cylinders.
    vertical_axis : str, optional
        The vertical coordinate axis ("x", "y", or "z"). Default is "y".
    """
    print("Launching interactive multi-panel 3D/2D Visual Analysis Dashboard...")

    # Determine vertical and horizontal coordinate indices
    v_idx = {"x": 0, "y": 1, "z": 2}[vertical_axis.lower()]
    h_indices = [i for i in [0, 1, 2] if i != v_idx]

    stem_pts = points[classification == 1]
    leaf_pts = points[classification == 0]

    # Create figure
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Procedural Forest Tree - LiDAR Analysis Dashboard", fontsize=18, fontweight='bold', y=0.98)

    # ----------------------------------------------------
    # PANEL 1: 3D Semantic Classification (Top-Left)
    # ----------------------------------------------------
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(stem_pts[:, 0], stem_pts[:, 1], stem_pts[:, 2], c='#8B4513', s=1.5, label='Stem/Wood (Class 1)', alpha=0.5)
    ax1.scatter(leaf_pts[:, 0], leaf_pts[:, 1], leaf_pts[:, 2], c='#228B22', s=0.8, label='Crown/Foliage (Class 0)', alpha=0.4)
    ax1.set_title("1. 3D Semantic Segmentation", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.set_zlabel("Z (meters)")
    ax1.legend(loc='upper right')
    
    # Aspect ratio scaling
    max_range = np.array([points[:, 0].max() - points[:, 0].min(), 
                          points[:, 1].max() - points[:, 1].min(), 
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # ----------------------------------------------------
    # PANEL 2: 2D DBH Fitting at 1.3m Height (Top-Right)
    # ----------------------------------------------------
    ax2 = fig.add_subplot(222)
    ax2.scatter(dbh_inliers[:, 0], dbh_inliers[:, 1], c='#8B4513', s=15, alpha=0.6, label='Slice Points (1.3m)')
    
    # Draw fitted circle ring
    r_fit = dbh_val / 2.0
    circle = Circle((dbh_center[0], dbh_center[1]), r_fit, edgecolor='red', facecolor='none', linewidth=2.5, linestyle='--', label=f'DBH circle (D={dbh_val*100:.1f}cm)')
    ax2.add_patch(circle)
    
    # Plot center cross
    ax2.scatter([dbh_center[0]], [dbh_center[1]], c='black', marker='+', s=100, label='Trunk Center')
    
    ax2.set_title(f"2. 2D DBH Circle Fit (Height = 1.3m)", fontsize=12, fontweight='bold')
    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    ax2.set_xlabel(axis_labels[h_indices[0]])
    ax2.set_ylabel(axis_labels[h_indices[1]])
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    # ----------------------------------------------------
    # PANEL 3: 3D Canopy Convex Hull Envelope (Bottom-Left)
    # ----------------------------------------------------
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(leaf_pts[:, 0], leaf_pts[:, 1], leaf_pts[:, 2], c='#228B22', s=0.8, alpha=0.2, label='Foliage Points')
    
    # Overlay translucent 3D Convex Hull faces
    if ConvexHull is not None and Poly3DCollection is not None:
        try:
            hull = ConvexHull(leaf_pts)
            # Reconstruct 3D triangulation face coordinates
            for simplex in hull.simplices:
                tri = leaf_pts[simplex]
                poly = Poly3DCollection([tri], facecolors='g', alpha=0.10, edgecolors='#196F3D', linewidths=0.2)
                ax3.add_collection3d(poly)
            ax3.text(mid_x, mid_y, leaf_pts[:, v_idx].max() + 0.5, f"Canopy Volume: {canopy_results['canopy_volume_m3']:.1f} m³", color='green', ha='center', fontweight='bold')
        except Exception:
            pass

    ax3.set_title("3. 3D Canopy Envelope Volume", fontsize=12, fontweight='bold')
    ax3.set_xlabel("X (meters)")
    ax3.set_ylabel("Y (meters)")
    ax3.set_zlabel("Z (meters)")
    ax3.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3.set_zlim(mid_z - max_range, mid_z + max_range)

    # ----------------------------------------------------
    # PANEL 4: 3D Fitted Cylinders Profile (Bottom-Right)
    # ----------------------------------------------------
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(stem_pts[:, 0], stem_pts[:, 1], stem_pts[:, 2], c='#8B4513', s=1.5, alpha=0.2, label='Wood Points')
    
    # Draw cylinder wireframe boundaries/circles
    for cyl in cylinders:
        c = cyl["center_3d"]
        r = cyl["radius"]
        h = cyl["height"]
        
        # Draw top and bottom circles of each fitted cylinder segment
        theta = np.linspace(0, 2*np.pi, 30)
        c_x = c[h_indices[0]] + r * np.cos(theta)
        c_z = c[h_indices[1]] + r * np.sin(theta)
        
        h_bottom = c[v_idx] - h/2.0
        h_top = c[v_idx] + h/2.0
        
        # Build circular arrays
        circ_bottom = np.zeros((30, 3))
        circ_bottom[:, v_idx] = h_bottom
        circ_bottom[:, h_indices[0]] = c_x
        circ_bottom[:, h_indices[1]] = c_z
        
        circ_top = np.zeros((30, 3))
        circ_top[:, v_idx] = h_top
        circ_top[:, h_indices[0]] = c_x
        circ_top[:, h_indices[1]] = c_z
        
        # Plot circles
        ax4.plot(circ_bottom[:, 0], circ_bottom[:, 1], circ_bottom[:, 2], color='red', linewidth=1.2, alpha=0.7)
        ax4.plot(circ_top[:, 0], circ_top[:, 1], circ_top[:, 2], color='red', linewidth=1.2, alpha=0.7)
        
        # Draw side connectors for visualization
        for idx in [0, 7, 15, 22]:
            ax4.plot([circ_bottom[idx, 0], circ_top[idx, 0]], 
                     [circ_bottom[idx, 1], circ_top[idx, 1]], 
                     [circ_bottom[idx, 2], circ_top[idx, 2]], color='red', linewidth=0.8, alpha=0.5)

    ax4.set_title("4. 3D Cylinder Taper Reconstruction", fontsize=12, fontweight='bold')
    ax4.set_xlabel("X (meters)")
    ax4.set_ylabel("Y (meters)")
    ax4.set_zlabel("Z (meters)")
    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()
