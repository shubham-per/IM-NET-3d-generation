import torch
import numpy as np
from model import IMNet30000
from skimage.measure import marching_cubes
import trimesh

# Load trained model
model = IMNet30000()
model.load_state_dict(torch.load('imnet_model_30000.pth'))
model.eval()

# Generate shape (vertices)
with torch.no_grad():
    noise = torch.randn((1, 30000))  # Adjust noise size based on input dimensions
    vertices = model(noise).view(-1, 3).numpy()

# Step 1: Convert vertices into a voxel grid (or scalar field)
# Here, we assume a simple method to convert to voxel grid.
# This is a simplified version; you might need more sophisticated methods.
# We'll create a grid and fill it with 1s and 0s (binary voxel grid).

# Define the size of the voxel grid
grid_size = 64
voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

# Populate the voxel grid based on the vertices
# This is just an example; you'll need a better way to convert vertices to the scalar field
for v in vertices:
    x, y, z = np.clip(np.floor(v * grid_size).astype(int), 0, grid_size - 1)
    voxel_grid[x, y, z] = 1  # Set a voxel to 1 (inside the shape)

# Step 2: Apply Marching Cubes to extract the mesh from the voxel grid
# The algorithm returns vertices, faces, and normals.
verts, faces, _, _ = marching_cubes(voxel_grid, level=0.5)

# Step 3: Save the mesh as an OBJ file
def save_obj(vertices, faces, filename="generated_shape1.obj"):
    with open(filename, 'w') as file:
        # Write vertices
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write faces
        for f in faces:
            # OBJ indices start at 1, so we add 1 to each face index
            file.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")
    
    print(f"3D model with mesh saved as {filename}")

# Save the mesh
save_obj(verts, faces)
