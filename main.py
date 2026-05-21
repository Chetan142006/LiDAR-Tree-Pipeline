"""
Forestry LiDAR Analysis Pipeline - Unified Project Orchestrator.

This script coordinates:
1. Procedural L-system 3D tree generation (Oak, Pine, Cypress presets).
2. The Modular Forestry LiDAR Analysis Pipeline:
   - lidar_loader: Reads LAS point clouds.
   - stem_crown_segmentation: Classifies wood vs. leaves.
   - dbh_estimation: Calculates Diameter at Breast Height (DBH) at 1.3m height.
   - cylinder_fitting: Reconstructs tapering 3D stem cylinders.
   - canopy_analysis: Estimates volume and 2D canopy spread area.
   - branch_analysis: Measures bifurcation branching angles.
   - crown_base_height: Locates living canopy base (CBH).
   - visualization: Launches the 2x2 interactive 3D visual analysis dashboard.
"""

import sys
import os
import argparse
import subprocess
import numpy as np
from typing import Dict, Any

# Import modular pipeline components directly for high performance
from lidar_loader import load_las
from stem_crown_segmentation import segment_stem_and_crown
from dbh_estimation import estimate_dbh
from cylinder_fitting import fit_trunk_cylinders
from canopy_analysis import analyze_canopy
from branch_analysis import analyze_branches
from crown_base_height import estimate_crown_base_height
from visualization import render_forestry_dashboard


def run_generator_script(script_name: str, preset: str, axis: str, seed: int = None) -> None:
    """
    Runs a target tree generator script as a subprocess, forwarding biological presets.

    Parameters
    ----------
    script_name : str
        Target python file.
    preset : str
        Tree growth preset ("oak", "pine", "cypress").
    axis : str
        Tree growth orientation axis ("x", "y", "z").
    seed : int, optional
        Random seed.
    """
    if not os.path.exists(script_name):
        print(f"\n[ERROR] Target script '{script_name}' not found.")
        return

    print(f"\n" + "=" * 65)
    print(f"[GENERATING] Executing: {script_name}")
    print(f"             Preset: {preset.upper()} | Vertical Axis: {axis.upper()}")
    print("=" * 65 + "\n")

    cmd = [sys.executable, "-u", script_name, "--preset", preset, "--axis", axis]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    try:
        # Inherit standard streams so interactive plots display
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] '{script_name}' finished execution.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] '{script_name}' exited with error code {e.returncode}.")
    except KeyboardInterrupt:
        print(f"\n[INFO] Tree generation interrupted by user.")


def execute_analysis_pipeline(file_path: str, vertical_axis: str = "y") -> None:
    """
    Runs the entire modular analytical forestry pipeline on a .las point cloud file.

    Parameters
    ----------
    file_path : str
        Path to the target .las file to analyze.
    vertical_axis : str, optional
        The vertical coordinate axis of the point cloud. Default is "y".
    """
    if not os.path.exists(file_path):
        print(f"\n[ERROR] Point cloud file not found at: {file_path}")
        print("Please generate a tree first using Options [1] or [2]!")
        return

    print(f"\n" + "=" * 65)
    print(f"             RUNNING FORESTRY LIDAR ANALYSIS PIPELINE       ")
    print(f"  Target File: {file_path}")
    print(f"  Vertical Growth Axis: {vertical_axis.upper()}")
    print("=" * 65)

    try:
        # Step 1: LiDAR Data Loading
        data = load_las(file_path)
        points = data["points"]
        labels = data["classification"]

        # Step 2: Stem-Crown Point Cloud Segmentation
        stem_pts, crown_pts = segment_stem_and_crown(points, labels, vertical_axis)

        # Step 3: DBH Estimation at 1.3 meters
        dbh, dbh_center, dbh_inliers = estimate_dbh(stem_pts, vertical_axis)

        # Step 4: Reconstruct trunk tapering 3D cylinders
        cylinders = fit_trunk_cylinders(stem_pts, vertical_axis)

        # Step 5: Canopy Structure Analysis (Height, Area, Volume)
        canopy_res = analyze_canopy(crown_pts, vertical_axis)

        # Step 6: Branch Architecture Analysis
        branches = analyze_branches(stem_pts, vertical_axis)

        # Step 7: Crown Base Height (CBH) Estimation
        cbh = estimate_crown_base_height(points, labels, vertical_axis)

        # ----------------------------------------------------
        # Print Consolidated Ecological Forestry Report
        # ----------------------------------------------------
        print("\n" + "=" * 65)
        print("                 FORESTRY STRUCTURE SUMMARY REPORT          ")
        print("=" * 65)
        print(f"  Point Cloud File:         {os.path.basename(file_path)}")
        print(f"  Total LiDAR Point Count:  {points.shape[0]:,}")
        print(f"  Estimated DBH (Diameter): {dbh * 100:.1f} cm ({dbh:.3f} meters)")
        print(f"  Crown Base Height (CBH):  {cbh:.2f} meters")
        print(f"  Canopy Crown Height:      {canopy_res['canopy_height']:.2f} meters")
        print(f"  Canopy Spread (Width):    {canopy_res['canopy_spread_axis1']:.2f}m x {canopy_res['canopy_spread_axis2']:.2f}m")
        print(f"  Canopy Projection Area:   {canopy_res['projection_area_m2']:.2f} m²")
        print(f"  Canopy Crown Volume:      {canopy_res['canopy_volume_m3']:.2f} m³")
        print("-" * 65)
        print("  Reconstructed Cylinder Taper Profile:")
        if len(cylinders) > 0:
            for c in cylinders:
                print(f"    Elevation: {c['elevation']:4.1f}m -> Radius: {c['radius']:.3f}m (D={c['radius']*200:.1f}cm)")
        else:
            print("    No trunk cylinders successfully fitted.")
        print("-" * 65)
        print("  Structural Branching Bifurcation Angles:")
        if len(branches) > 0:
            for b in branches:
                print(f"    Branch [{b['quadrant']:10}]: Angle: {b['angle_deg']:.1f}°")
        else:
            print("    No major branch segments isolated.")
        print("=" * 65 + "\n")

        # Step 8: Render the Multi-Panel Interactive Visualization Dashboard
        render_forestry_dashboard(
            points=points,
            classification=labels,
            dbh_val=dbh,
            dbh_center=dbh_center,
            dbh_inliers=dbh_inliers,
            canopy_results=canopy_res,
            cylinders=cylinders,
            vertical_axis=vertical_axis
        )

    except Exception as e:
        print(f"\n[ERROR] Pipeline analysis encountered a critical error: {e}")
        import traceback
        traceback.print_exc()


def show_menu() -> None:
    """
    Displays an interactive console menu and executes user choices.
    """
    while True:
        print("\n" + "=" * 60)
        print("    FORESTRY LIDAR GENERATION & ANALYSIS SYSTEM MENU     ")
        print("=" * 60)
        print("  [1] Generate Realistic L-System Tree (Curved Leaves)")
        print("      - Saves: 'realistic_synthetic_tree.las'")
        print("  [2] Generate Canopy L-System Tree (Flat Box Leaves)")
        print("      - Saves: 'realistic_canopy_tree.las'")
        print("-" * 60)
        print("  [3] Run Modular Forestry Analysis Pipeline")
        print("      - Slices DBH, fits tapering cylinders, calculates")
        print("        canopy volume/spread, CBH, branching angles, and")
        print("        launches the 3D/2D comparative analysis dashboard.")
        print("-" * 60)
        print("  [4] Run Sequential Run (Gen Curved Tree + Pipeline)")
        print("-" * 60)
        print("  [5] Exit System")
        print("=" * 60)

        choice = input("Enter choice [1-5]: ").strip()
        if not choice:
            continue

        if choice in ["1", "2", "4"]:
            # Prompt for biological preset
            print("\nSelect Biological/Architectural Preset:")
            print("  [1] Oak (Decurrent/Broadleaf, wide angles, gravity droop) [Default]")
            print("  [2] Pine (Excurrent/Conifer, central leader, wide side branches)")
            print("  [3] Cypress (Columnar/Fastigiate, narrow branches, upward hugging)")
            preset_choice = input("Choice [1-3] (Default=1): ").strip()
            
            preset = "oak"
            if preset_choice == "2":
                preset = "pine"
            elif preset_choice == "3":
                preset = "cypress"

            # Prompt for Vertical Axis
            print("\nSelect Growth Vertical Axis:")
            print("  [1] Y-Axis (Default upright growth for standard coordinate frames)")
            print("  [2] Z-Axis (Upright vertical for standard GIS coordinates)")
            axis_choice = input("Choice [1-2] (Default=1): ").strip()
            axis = "y" if axis_choice != "2" else "z"

            seed_in = input("\nEnter custom seed (integer) for reproducibility [Optional, Press Enter for random]: ").strip()
            seed = int(seed_in) if seed_in.isdigit() else None

            if choice == "1":
                run_generator_script("generate_realistic_tree.py", preset, axis, seed)
            elif choice == "2":
                run_generator_script("generate_realistic_canopy.py", preset, axis, seed)
            elif choice == "4":
                run_generator_script("generate_realistic_tree.py", preset, axis, seed)
                # Analyze the generated tree
                target_file = "realistic_synthetic_tree.las"
                execute_analysis_pipeline(target_file, axis)

        elif choice == "3":
            # Prompt for file and axis
            print("\nSelect point cloud file to analyze:")
            print("  [1] realistic_synthetic_tree.las (Lanceolate leaf model) [Default]")
            print("  [2] realistic_canopy_tree.las (Flat canopy leaf model)")
            print("  [3] Enter custom .las file path...")
            file_choice = input("Choice [1-3] (Default=1): ").strip()
            
            target_file = "realistic_synthetic_tree.las"
            if file_choice == "2":
                target_file = "realistic_canopy_tree.las"
            elif file_choice == "3":
                custom_file = input("Enter full file path: ").strip()
                if custom_file:
                    target_file = custom_file

            print(f"\nSpecify vertical growth axis used to generate this tree:")
            axis_in = input("Axis (y/z) [Default=y]: ").strip().lower()
            axis = "y" if axis_in not in ["x", "y", "z"] else axis_in

            execute_analysis_pipeline(target_file, axis)

        elif choice == "5":
            print("\nExiting Forestry LiDAR Orchestrator. Thank you!")
            break
        else:
            print("\n[INVALID] Please choose an option from 1 to 5.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Forestry LiDAR Orchestrator and Modular Analysis Pipeline.")
    parser.add_argument(
        "--run", 
        type=str, 
        choices=["tree", "canopy", "analyze", "all"], 
        default=None, 
        help="Command execution bypass without interactive menu."
    )
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=["oak", "pine", "cypress"], 
        default="oak", 
        help="Biological preset for tree generator (oak, pine, cypress)."
    )
    parser.add_argument(
        "--axis", 
        type=str, 
        choices=["x", "y", "z"], 
        default="y", 
        help="Vertical growth axis (default is 'y')."
    )
    parser.add_argument(
        "--file", 
        type=str, 
        default="realistic_synthetic_tree.las", 
        help="Target LAS file for analysis."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Random seed for tree generation."
    )
    
    args = parser.parse_args()

    if args.run is not None:
        if args.run == "tree":
            run_generator_script("generate_realistic_tree.py", args.preset, args.axis, args.seed)
        elif args.run == "canopy":
            run_generator_script("generate_realistic_canopy.py", args.preset, args.axis, args.seed)
        elif args.run == "analyze":
            execute_analysis_pipeline(args.file, args.axis)
        elif args.run == "all":
            run_generator_script("generate_realistic_tree.py", args.preset, args.axis, args.seed)
            execute_analysis_pipeline("realistic_synthetic_tree.las", args.axis)
    else:
        # Default to interactive menu mode
        show_menu()


if __name__ == "__main__":
    main()
