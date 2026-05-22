"""
Procedural Realistic 3D Tree Generator with Species-Specific L-System Branching,
Gravitropism, Spiral Phyllotaxis Flat Canopy Leaf Distribution, and Semantic LAS Export.

This script implements:
1. Procedural L-system recursive branching tree geometry (Trunk + Branches + Canopy).
2. Architectural presets: Oak (Broadleaf), Pine (Conifer), Cypress (Columnar).
3. Physical Gravitropism (gravitational branch droop).
4. Spiral Foliar Phyllotaxis: Flat box leaves distributed along the twig shafts in golden spirals.
5. Surface point sampling of meshes using Trimesh.
6. Point-level semantic class annotation (Class 1 = Wood, Class 0 = Leaf).
7. Standard ASPRS LAS format point cloud export using laspy.
"""

import argparse
import os
import numpy as np
import trimesh
import laspy
from typing import List, Tuple, Dict, Any


def get_rotation_matrix_to_align_z_with(direction: np.ndarray) -> np.ndarray:
    """
    Computes the 3x3 rotation matrix that rotates the unit Z-axis vector [0, 0, 1]
    to align with the target direction vector using Rodrigues' rotation formula.
    """
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.eye(3)
    b = direction / norm
    a = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    c = np.dot(a, b)

    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float64)

    v = np.cross(a, b)
    V_x = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=np.float64)

    R = np.eye(3) + V_x + np.dot(V_x, V_x) * (1.0 / (1.0 + c))
    return R


def spawn_flat_box_leaf(
    position: np.ndarray,
    orientation: np.ndarray,
    leaf_meshes: List[trimesh.Trimesh]
) -> None:
    """
    Creates a single flat box leaf mesh at a target position and orientation.
    """
    # Flat, thin box representing leaf segment
    leaf = trimesh.creation.box(extents=[0.15, 0.25, 0.01])

    # Orientation
    d = orientation / np.linalg.norm(orientation)
    R_leaf = get_rotation_matrix_to_align_z_with(d)

    # Roll rotation around leaf axis
    roll_angle = np.random.uniform(0, 2 * np.pi)
    cos_r = np.cos(roll_angle)
    sin_r = np.sin(roll_angle)
    K = np.array([
        [0, -d[2], d[1]],
        [d[2], 0, -d[0]],
        [-d[1], d[0], 0]
    ], dtype=np.float64)
    R_roll = np.eye(3) + sin_r * K + (1 - cos_r) * np.dot(K, K)
    
    R_final = np.dot(R_roll, R_leaf)

    T = np.eye(4)
    T[:3, :3] = R_final
    T[:3, 3] = position

    leaf.apply_transform(T)
    leaf_meshes.append(leaf)


def distribute_leaves_spiral_canopy(
    start_pt: np.ndarray,
    direction: np.ndarray,
    length: float,
    leaf_meshes: List[trimesh.Trimesh],
    num_leaves: int = 5
) -> None:
    """
    Distributes flat box leaves along the twig segment using a golden spiral phyllotaxis pattern.
    """
    d = direction / np.linalg.norm(direction)
    
    # Orthonormal basis
    if abs(d[0]) < 0.9:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        r = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(d, r)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)

    golden_angle = np.radians(137.5)

    for i in range(1, num_leaves + 1):
        # Spacing along twig (leaving some gap near base)
        fraction = 0.2 + 0.8 * (i / num_leaves)
        # Position with a slight translational jitter (+/- 0.02m)
        jitter = np.random.uniform(-0.02, 0.02, 3)
        pos = start_pt + fraction * d * length + jitter

        # Golden spiral angle around branch axis
        phi = i * golden_angle
        # Outward flare from branch axis
        theta = np.random.uniform(np.radians(20), np.radians(45))

        # Leaf direction vector
        leaf_dir = np.cos(theta) * d + np.sin(theta) * np.cos(phi) * u + np.sin(theta) * np.sin(phi) * v
        spawn_flat_box_leaf(pos, leaf_dir, leaf_meshes)


def generate_branch_recursive(
    start_pt: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    depth: int,
    max_depth: int,
    wood_meshes: List[trimesh.Trimesh],
    leaf_meshes: List[trimesh.Trimesh],
    preset_cfg: Dict[str, Any],
    gravity_vector: np.ndarray
) -> None:
    """
    Recursively generates L-system branch cylinders, incorporating gravitropism and presets.
    """
    direction = direction / np.linalg.norm(direction)

    # 1. Create cylinder centered at origin along local Z
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)

    # 2. Position cylinder along direction vector
    end_pt = start_pt + direction * length
    center = start_pt + direction * (length / 2.0)
    R = get_rotation_matrix_to_align_z_with(direction)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center

    cylinder.apply_transform(T)
    wood_meshes.append(cylinder)

    # 3. Handle foliar distribution along twig segments (at final levels)
    if depth >= max_depth - 1:
        distribute_leaves_spiral_canopy(start_pt, direction, length, leaf_meshes, num_leaves=6)

    # 4. Branch Split Recursion
    if depth < max_depth:
        # Orthonormal basis
        d = direction
        if abs(d[0]) < 0.9:
            r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            r = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = np.cross(d, r)
        u = u / np.linalg.norm(u)
        v = np.cross(d, u)

        if preset_cfg["apical_dominance"]:
            # CONIFER / PINE: Apical Dominance Mode
            # One dominant vertical leader + a whorl of lateral side branches
            
            # Leader: grows almost straight vertically with minimal decay
            leader_dir = np.cos(np.radians(3)) * d + np.sin(np.radians(3)) * (np.cos(0) * u + np.sin(0) * v)
            leader_dir = leader_dir / np.linalg.norm(leader_dir)
            
            generate_branch_recursive(
                start_pt=end_pt,
                direction=leader_dir,
                length=length * np.random.uniform(0.85, 0.90),
                radius=radius * np.random.uniform(0.80, 0.85),
                depth=depth + 1,
                max_depth=max_depth,
                wood_meshes=wood_meshes,
                leaf_meshes=leaf_meshes,
                preset_cfg=preset_cfg,
                gravity_vector=gravity_vector
            )

            # Lateral whorl: 3 side branches spreading wide outward
            num_laterals = 3
            for i in range(num_laterals):
                theta = np.random.uniform(preset_cfg["branching_angle_min"], preset_cfg["branching_angle_max"])
                phi = (2 * np.pi * i / num_laterals) + np.random.uniform(-0.15, 0.15)

                child_dir = np.cos(theta) * d + np.sin(theta) * np.cos(phi) * u + np.sin(theta) * np.sin(phi) * v
                
                # Apply Gravitropism (Downward droop on side branches)
                child_dir = child_dir + preset_cfg["gravity_pull"] * gravity_vector
                child_dir = child_dir / np.linalg.norm(child_dir)

                generate_branch_recursive(
                    start_pt=end_pt,
                    direction=child_dir,
                    length=length * np.random.uniform(preset_cfg["length_decay_min"], preset_cfg["length_decay_max"]),
                    radius=radius * np.random.uniform(preset_cfg["radius_decay_min"], preset_cfg["radius_decay_max"]),
                    depth=depth + 1,
                    max_depth=max_depth,
                    wood_meshes=wood_meshes,
                    leaf_meshes=leaf_meshes,
                    preset_cfg=preset_cfg,
                    gravity_vector=gravity_vector
                )
        else:
            # BROADLEAF / OAK / CYPRESS: Sympodial Bifurcation Mode (Splits into 2)
            theta1 = np.random.uniform(preset_cfg["branching_angle_min"], preset_cfg["branching_angle_max"])
            theta2 = np.random.uniform(preset_cfg["branching_angle_min"], preset_cfg["branching_angle_max"])

            phi1 = np.random.uniform(0, 2 * np.pi)
            phi2 = phi1 + np.pi + np.random.uniform(np.radians(-15), np.radians(15))

            child1_dir = np.cos(theta1) * d + np.sin(theta1) * np.cos(phi1) * u + np.sin(theta1) * np.sin(phi1) * v
            child2_dir = np.cos(theta2) * d + np.sin(theta2) * np.cos(phi2) * u + np.sin(theta2) * np.sin(phi2) * v

            # Apply Gravitropism
            child1_dir = child1_dir + preset_cfg["gravity_pull"] * gravity_vector
            child2_dir = child2_dir + preset_cfg["gravity_pull"] * gravity_vector

            child1_dir = child1_dir / np.linalg.norm(child1_dir)
            child2_dir = child2_dir / np.linalg.norm(child2_dir)

            length_decay = np.random.uniform(preset_cfg["length_decay_min"], preset_cfg["length_decay_max"])
            radius_decay = np.random.uniform(preset_cfg["radius_decay_min"], preset_cfg["radius_decay_max"])

            generate_branch_recursive(
                start_pt=end_pt,
                direction=child1_dir,
                length=length * length_decay,
                radius=radius * radius_decay,
                depth=depth + 1,
                max_depth=max_depth,
                wood_meshes=wood_meshes,
                leaf_meshes=leaf_meshes,
                preset_cfg=preset_cfg,
                gravity_vector=gravity_vector
            )
            generate_branch_recursive(
                start_pt=end_pt,
                direction=child2_dir,
                length=length * length_decay,
                radius=radius * radius_decay,
                depth=depth + 1,
                max_depth=max_depth,
                wood_meshes=wood_meshes,
                leaf_meshes=leaf_meshes,
                preset_cfg=preset_cfg,
                gravity_vector=gravity_vector
            )


def plot_point_cloud(points: np.ndarray, labels: np.ndarray) -> None:
    """
    Displays the generated 3D point cloud using Matplotlib 3D scatter plot.
    """
    print("Launching interactive 3D plot of the generated tree point cloud...")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed. Skipping display plot.")
        return

    wood_pts = points[labels == 1]
    leaf_pts = points[labels == 0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(wood_pts[:, 0], wood_pts[:, 1], wood_pts[:, 2], c='#8B4513', s=2, label='Wood/Trunk (Class 1)', alpha=0.6)
    ax.scatter(leaf_pts[:, 0], leaf_pts[:, 1], leaf_pts[:, 2], c='#228B22', s=0.8, label='Leaf/Canopy (Class 0)', alpha=0.5)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('Procedural Upgraded Flat Canopy Tree 3D Point Cloud')
    ax.legend(loc='upper right')

    max_range = np.array([points[:, 0].max() - points[:, 0].min(), 
                          points[:, 1].max() - points[:, 1].min(), 
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def main() -> None:
    PRESETS = {
        "oak": {
            "branching_angle_min": np.radians(25),
            "branching_angle_max": np.radians(35),
            "length_decay_min": 0.75,
            "length_decay_max": 0.85,
            "radius_decay_min": 0.65,
            "radius_decay_max": 0.75,
            "gravity_pull": 0.15,
            "apical_dominance": False
        },
        "pine": {
            "branching_angle_min": np.radians(55),
            "branching_angle_max": np.radians(70),
            "length_decay_min": 0.55,
            "length_decay_max": 0.65,
            "radius_decay_min": 0.45,
            "radius_decay_max": 0.55,
            "gravity_pull": 0.25, # droop down
            "apical_dominance": True
        },
        "cypress": {
            "branching_angle_min": np.radians(10),
            "branching_angle_max": np.radians(18),
            "length_decay_min": 0.80,
            "length_decay_max": 0.90,
            "radius_decay_min": 0.70,
            "radius_decay_max": 0.78,
            "gravity_pull": -0.05, # upward hugging
            "apical_dominance": False
        }
    }

    parser = argparse.ArgumentParser(description="Procedural L-System synthetic flat leaf tree generator.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--preset", type=str, default="oak", choices=["oak", "pine", "cypress"], help="Tree preset profile.")
    parser.add_argument("--max-depth", type=int, default=5, help="L-system recursion depth.")
    parser.add_argument("--trunk-length", type=float, default=4.0, help="Trunk length in meters.")
    parser.add_argument("--trunk-radius", type=float, default=0.25, help="Trunk radius in meters.")
    parser.add_argument("--axis", type=str, default="y", choices=["x", "y", "z"], help="Vertical growth axis.")
    parser.add_argument("--output", type=str, default=os.path.join("output", "realistic_canopy_tree.las"), help="Output LAS path.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.seed is None:
        args.seed = int(np.random.randint(0, 1000000))
        print(f"Using random seed: {args.seed}")
    else:
        print(f"Using manual seed: {args.seed}")

    np.random.seed(args.seed)

    # Set up growth axes and gravity vectors
    if args.axis == "x":
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        gravity_vector = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    elif args.axis == "y":
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        gravity_vector = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    else:
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        gravity_vector = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    wood_meshes: List[trimesh.Trimesh] = []
    leaf_meshes: List[trimesh.Trimesh] = []

    print(f"Generating organic flat-canopy 3D tree. Preset: {args.preset.upper()} along {args.axis.upper()}-axis...")
    generate_branch_recursive(
        start_pt=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        direction=direction,
        length=args.trunk_length,
        radius=args.trunk_radius,
        depth=1,
        max_depth=args.max_depth,
        wood_meshes=wood_meshes,
        leaf_meshes=leaf_meshes,
        preset_cfg=PRESETS[args.preset],
        gravity_vector=gravity_vector
    )

    print(f"Concatenating {len(wood_meshes)} wood meshes and {len(leaf_meshes)} flat canopy meshes...")
    wood_mesh = trimesh.util.concatenate(wood_meshes)
    leaf_mesh = trimesh.util.concatenate(leaf_meshes)

    # Colors for PLY
    wood_mesh.visual.vertex_colors = np.tile([139, 69, 19, 255], (len(wood_mesh.vertices), 1)).astype(np.uint8)
    leaf_mesh.visual.vertex_colors = np.tile([34, 139, 34, 255], (len(leaf_mesh.vertices), 1)).astype(np.uint8)

    ply_output = args.output.replace(".las", ".ply")
    print(f"Saving solid 3D representation to: {ply_output}...")
    combined_mesh = trimesh.util.concatenate([wood_mesh, leaf_mesh])
    combined_mesh.export(ply_output)

    # Point Surface Sampling
    print("Scattering points across mesh surfaces...")
    num_wood_samples = 10000
    num_leaf_samples = 30000 # dense leaf canopy

    wood_points, _ = trimesh.sample.sample_surface(wood_mesh, num_wood_samples)
    leaf_points, _ = trimesh.sample.sample_surface(leaf_mesh, num_leaf_samples)

    wood_labels = np.full(wood_points.shape[0], 1, dtype=np.int32)
    leaf_labels = np.full(leaf_points.shape[0], 0, dtype=np.int32)

    all_points = np.vstack([wood_points, leaf_points])
    all_labels = np.concatenate([wood_labels, leaf_labels])

    # Generate realistic LiDAR Intensity and Return statistics
    num_pts = all_points.shape[0]
    intensities = np.zeros(num_pts, dtype=np.int32)
    return_numbers = np.ones(num_pts, dtype=np.int32)
    number_of_returns = np.ones(num_pts, dtype=np.int32)

    # Wood points (Class 1) - Low NIR reflectivity (bark/wood), 100% single returns
    wood_mask = (all_labels == 1)
    num_wood = np.sum(wood_mask)
    if num_wood > 0:
        intensities[wood_mask] = np.clip(np.random.normal(60, 15, num_wood).astype(np.int32), 10, 120)
        return_numbers[wood_mask] = 1
        number_of_returns[wood_mask] = 1

    # Leaf points (Class 0) - High NIR reflectivity (chlorophyll), multiple returns simulated
    leaf_mask = (all_labels == 0)
    num_leaf = np.sum(leaf_mask)
    if num_leaf > 0:
        intensities[leaf_mask] = np.clip(np.random.normal(180, 25, num_leaf).astype(np.int32), 80, 255)
        
        # Simulating laser penetration: 70% single return, 20% first of two, 10% second of two
        leaf_rand = np.random.rand(num_leaf)
        leaf_returns = np.ones(num_leaf, dtype=np.int32)
        leaf_num_returns = np.ones(num_leaf, dtype=np.int32)
        
        mask_1_of_2 = (leaf_rand >= 0.7) & (leaf_rand < 0.9)
        leaf_returns[mask_1_of_2] = 1
        leaf_num_returns[mask_1_of_2] = 2
        
        mask_2_of_2 = (leaf_rand >= 0.9)
        leaf_returns[mask_2_of_2] = 2
        leaf_num_returns[mask_2_of_2] = 2
        
        return_numbers[leaf_mask] = leaf_returns
        number_of_returns[leaf_mask] = leaf_num_returns

    # Verify no empty/NaN values exist. If any NaN exists, drop that row completely.
    valid_mask = ~np.isnan(all_points).any(axis=1) & ~np.isnan(all_labels) & ~np.isnan(intensities) & ~np.isnan(return_numbers) & ~np.isnan(number_of_returns)
    all_points = all_points[valid_mask]
    all_labels = all_labels[valid_mask]
    intensities = intensities[valid_mask]
    return_numbers = return_numbers[valid_mask]
    number_of_returns = number_of_returns[valid_mask]

    # LAS Export using laspy
    print(f"Configuring LasHeader and exporting to: {args.output}...")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(all_points, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001]) # mm scale

    las = laspy.LasData(header)
    las.x = all_points[:, 0]
    las.y = all_points[:, 1]
    las.z = all_points[:, 2]
    las.classification = all_labels
    las.intensity = intensities
    las.return_number = return_numbers
    las.number_of_returns = number_of_returns
    las.write(args.output)

    # Also save as human-readable CSV for easy analysis (Excel, MATLAB, etc.)
    csv_output = args.output.replace(".las", "_points.csv")
    print(f"Saving human-readable CSV coordinates to: {csv_output}...")
    csv_data = np.column_stack((
        all_points[:, 0],
        all_points[:, 1],
        all_points[:, 2],
        all_labels,
        intensities,
        return_numbers,
        number_of_returns
    ))
    np.savetxt(csv_output, csv_data, delimiter=",", 
               header="x,y,z,classification,intensity,return_number,number_of_returns", 
               fmt=["%.5f", "%.5f", "%.5f", "%d", "%d", "%d", "%d"],
               comments="")

    # Save a sidecar JSON file with generation metadata
    import json
    json_output = args.output.replace(".las", ".json")
    v_idx = {"x": 0, "y": 1, "z": 2}[args.axis.lower()]
    tree_height = float(np.max(all_points[:, v_idx]) - np.min(all_points[:, v_idx]))
    metadata = {
        "File_Path": args.output,
        "Ground_Truth_DBH": float(2 * args.trunk_radius * 100.0), # in cm
        "Tree_Height": tree_height, # in meters
        "Total_Points": int(all_points.shape[0])
    }
    print(f"Saving sidecar JSON metadata to: {json_output}...")
    with open(json_output, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n--- Point Cloud Validation Statistics ---")
    print(f"File: {args.output}")
    print(f"Total Points: {all_points.shape[0]}")
    unique_classes, counts = np.unique(all_labels, return_counts=True)
    for class_id, count in zip(unique_classes, counts):
        class_name = "Leaf/Canopy (Class 0)" if class_id == 0 else "Wood/Trunk (Class 1)"
        print(f"  {class_name}: {count} points ({count/all_points.shape[0]*100:.1f}%)")
    print("----------------------------------------\n")

    plot_point_cloud(all_points, all_labels)


if __name__ == "__main__":
    main()
