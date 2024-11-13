import torch
from model import IMNet30000
import numpy as np

# Load trained model
model = IMNet30000()
model.load_state_dict(torch.load('imnet_model_30000.pth'))

# Generate shape with noise matching the expected input size
noise = torch.randn((1, 30000))  # Adjusted input size
model.eval()
with torch.no_grad():
    vertices = model(noise).view(-1, 3).numpy()

print("Generated shape vertices:", vertices)

# Save as an .obj file
def save_obj(vertices, filename="generated_shape.obj"):
    with open(filename, 'w') as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
    print(f"3D model saved as {filename}")

# Call the function to save as .obj
save_obj(vertices)
