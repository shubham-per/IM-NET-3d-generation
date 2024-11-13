import os
import h5py
import trimesh
import numpy as np

# Directory containing .off files
off_dir = 'D:/Code/Python/ANK/IN NET/sano data/test'
# Output HDF5 file
h5_file = 'D:/Code/Python/ANK/IN NET/sano data/preprocessed/processed_data_test.h5'

# Initialize HDF5 file
with h5py.File(h5_file, 'w') as h5f:
    # Iterate through all .off files in the directory
    for i, filename in enumerate(os.listdir(off_dir)):
        if filename.endswith('.off'):
            file_path = os.path.join(off_dir, filename)
            # Load the mesh
            mesh = trimesh.load_mesh(file_path)
            
            # Convert mesh data to numpy arrays
            vertices = np.array(mesh.vertices, dtype=np.float32)
            faces = np.array(mesh.faces, dtype=np.int32)
            
            # Group name based on the index or filename
            group_name = f'model_{i}'
            group = h5f.create_group(group_name)
            
            # Store vertices and faces
            group.create_dataset('vertices', data=vertices, compression='gzip')
            group.create_dataset('faces', data=faces, compression='gzip')
            
            print(f'Processed {filename} and saved as {group_name} in HDF5 file.')

print("All .off files have been processed and saved in HDF5 format.")

