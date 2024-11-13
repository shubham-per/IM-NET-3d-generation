import torch
import torch.optim as optim
import torch.nn as nn
from dataset import ShapeDataset
from model import IMNet30000
from torch.utils.data import DataLoader

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
data_dir = 'D:/Code/Python/IM-NET/datasets/train'
train_dataset = ShapeDataset(data_dir)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss, and optimizer
model = IMNet30000().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, vertices in enumerate(train_loader):
        vertices = vertices.to(device)
        
        # Ensure input shape matches (batch_size, 30000)
        assert vertices.shape[1] == 30000, f"Unexpected input shape {vertices.shape}, expected (batch_size, 30000)"
        
        # Forward pass
        outputs = model(vertices)
        loss = criterion(outputs, vertices)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete.")
torch.save(model.state_dict(), 'imnet_model_300001.pth')
