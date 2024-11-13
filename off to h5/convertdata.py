import os
import h5py
import numpy as np
import trimesh
from tqdm import tqdm

def load_off(file_path):
    try:
        mesh = trimesh.load(file_path, file_type='off')
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None  # Return None if vertices or faces are empty
        return mesh
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

def generate_inside_outside(mesh, num_points=10000):
    # Add a margin around the bounding box to cover slightly outside points
    margin = 0.05
    min_bounds = mesh.bounds[0] - margin
    max_bounds = mesh.bounds[1] + margin

    # Randomly sample points within the bounding box
    points = np.random.uniform(low=min_bounds, high=max_bounds, size=(num_points, 3))

    # Use ray casting to determine if points are inside the mesh
    inside_outside = mesh.contains(points).astype(int).reshape(-1, 1)  # 1 for inside, 0 for outside

    return points, inside_outside

def convert_off_to_h5(input_dir, output_dir, points_per_shape=10000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    skipped_files = []  # List to store names of skipped files

    for file_name in tqdm(os.listdir(input_dir), desc="Converting .off to .h5"):
        if file_name.endswith('.off'):
            input_path = os.path.join(input_dir, file_name)
            mesh = load_off(input_path)
            if mesh is None:
                print(f"Skipping {file_name}: File {input_path} has empty vertices or faces")
                skipped_files.append(file_name)  # Log the skipped file
                continue

            points, inside_outside = generate_inside_outside(mesh, num_points=points_per_shape)

            output_file = os.path.join(output_dir, file_name.replace('.off', '.h5'))
            with h5py.File(output_file, 'w') as h5f:
                h5f.create_dataset('vertices', data=points)
                h5f.create_dataset('inside_outside', data=inside_outside)
    
    # Print the list of skipped files
    if skipped_files:
        print("\nSkipped files:")
        for file_name in skipped_files:
            print(file_name)

    print("Conversion completed! .h5 files saved in '{}'".format(output_dir))

# Example usage
input_dir = 'D:/Code/Python/ANK/IN NET/sano data/train'  # Path to directory with .off files
output_dir = 'D:/Code/Python/IM-NET/datasets/train'  # Output directory for .h5 files
convert_off_to_h5(input_dir, output_dir)
