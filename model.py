import torch
import torch.nn as nn

class IMNet30000(nn.Module):
    def __init__(self):
        super(IMNet30000, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30000, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 30000)  # Match original input dimension
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
