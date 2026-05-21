import laspy
import numpy as np

# 1. Read the binary LAS file
las = laspy.read("realistic_synthetic_tree.las")

# 2. Print file header information
print("--- LAS File Header Information ---")
print(f"Total Points: {las.header.point_count}")
print(f"LAS Version:  {las.header.version}")
print(f"Point Format: {las.header.point_format.id}")
print(f"Min Bounds (X, Y, Z): {las.header.mins}")
print(f"Max Bounds (X, Y, Z): {las.header.maxs}")

# 3. Combine coordinates and classification labels into a NumPy array
# (Class 0 = Leaf/Canopy, Class 1 = Wood/Trunk)
pts = np.stack([las.x, las.y, las.z, las.classification], axis=1)

# 4. Print a preview of the first 10 points
print("\n--- First 10 Points in Dataset (X, Y, Z, Class ID) ---")
for i, pt in enumerate(pts[:10]):
    print(f"Point {i+1:2d}: X = {pt[0]:8.4f}, Y = {pt[1]:8.4f}, Z = {pt[2]:8.4f} | Class = {int(pt[3])}")
