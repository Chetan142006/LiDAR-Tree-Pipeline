"""
Forestry LiDAR Analysis Pipeline - LiDAR Loader Module.

This module reads LAS formatted point clouds, extracts 3D coordinates, 
point classifications, and returns standard numpy representations.
"""

import os
import numpy as np
import laspy
from typing import Dict, Any


def load_las(file_path: str) -> Dict[str, Any]:
    """
    Loads a .las point cloud file, extracting coordinates and classification attributes.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the .las file.

    Returns
    -------
    dict
        A dictionary containing:
        - "points": numpy.ndarray of shape (N, 3) with float coordinate arrays.
        - "classification": numpy.ndarray of shape (N,) with integer classification IDs.
        - "header_stats": dict with metadata from the LAS file header.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LiDAR data file not found at: {file_path}")

    print(f"Opening LiDAR point cloud file: {file_path}")
    las = laspy.read(file_path)

    # Extract coordinates directly (laspy handles scale and offset correction automatically)
    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    points = np.vstack((x, y, z)).T

    # Extract classification if present
    try:
        classification = np.array(las.classification, dtype=np.int32)
    except AttributeError:
        print("Warning: Classification field missing in LAS file. Defaulting to unclassified (Class 0).")
        classification = np.zeros(points.shape[0], dtype=np.int32)

    header_stats = {
        "version": las.header.version,
        "point_format": las.header.point_format.id,
        "point_count": las.header.point_count,
        "min_bounds": list(las.header.min),
        "max_bounds": list(las.header.max)
    }

    print(f"Successfully loaded {points.shape[0]} points.")
    print(f"Bounding box: Min {header_stats['min_bounds']}, Max {header_stats['max_bounds']}")
    
    # Check classes
    unique_classes, counts = np.unique(classification, return_counts=True)
    print("Point counts per classification class:")
    for cls_id, count in zip(unique_classes, counts):
        class_name = "Leaf/Canopy (Class 0)" if cls_id == 0 else "Wood/Stem (Class 1)" if cls_id == 1 else f"Class {cls_id}"
        print(f"  {class_name}: {count} points ({count/points.shape[0]*100:.2f}%)")

    return {
        "points": points,
        "classification": classification,
        "header_stats": header_stats
    }


if __name__ == "__main__":
    # Self-test if executed directly
    test_path = "realistic_synthetic_tree.las"
    if os.path.exists(test_path):
        load_las(test_path)
    else:
        print("Self-test: Create a .las file to test this module.")
