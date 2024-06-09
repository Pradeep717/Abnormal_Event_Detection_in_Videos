import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

processed_data_dir = config['data']['processed_dir']
model_path = config['training']['anomaly_model_path']
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
epochs = config['training']['epochs']

class AnomalyDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(128, 128)
        self.logvar_layer = nn.Linear(128, 128)
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        decoded = self.decoder(z)
        return decoded, mu, logvar

def train_anomaly_detector():
    train_data_path = os.path.join(processed_data_dir, 'training.npy')
    train_dataset = AnomalyDataset(train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get the input dimension from the first batch
    train_features = next(iter(train_loader))
    input_dim = train_features.view(train_features.size(0), -1).shape[1]

    # Save the input_dim in the config file
    config['training']['input_dim'] = input_dim
    with open('config/config.yaml', 'w') as f:
        yaml.dump(config, f)

    model = VAE(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.view(inputs.size(0), -1)
            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)
            recon_loss = criterion(outputs, inputs)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), model_path)
    print("Training completed. Model saved at", model_path)

if __name__ == "__main__":
    train_anomaly_detector()
