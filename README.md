import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from timm import create_model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import cv2  # Import cv2 for image resizing

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

processed_data_dir = config['data']['processed_dir']
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
learning_rate = config['training']['learning_rate']
model_path = config['training']['transformer_model_path']

# Custom Dataset class
class VideoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Load preprocessed training data
training_data = np.load(os.path.join(processed_data_dir, 'training.npy')).astype(np.float32)  # Convert to float32
print(f"Training data shape: {training_data.shape}")

# The training data shape is (height, width, frames)
# Let's assume 10-frame sequences (as in the initial example)
num_frames = 10
height, width, total_frames = training_data.shape
num_samples = total_frames // num_frames

# Reshape the data to (num_samples, num_frames, height, width)
training_data = training_data[:, :, :num_samples * num_frames]  # Ensure total_frames is a multiple of num_frames
training_data = training_data.reshape(num_samples, num_frames, height, width)

# Convert to (num_samples, num_frames, channels, height, width)
# Since the images are grayscale, we only have 1 channel
training_data = np.expand_dims(training_data, axis=2)

# Resize frames to 224x224 to match Swin Transformer input and convert to RGB
training_data_resized = np.zeros((num_samples, num_frames, 3, 224, 224), dtype=np.float32)
for i in range(num_samples):
    for j in range(num_frames):
        resized_frame = cv2.resize(training_data[i, j, 0], (224, 224))
        training_data_resized[i, j, 0] = resized_frame
        training_data_resized[i, j, 1] = resized_frame
        training_data_resized[i, j, 2] = resized_frame

# Train-validation split
train_data, val_data = train_test_split(training_data_resized, test_size=0.2, random_state=42)
train_dataset = VideoDataset(train_data)
val_dataset = VideoDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the feature extractor (Swin Transformer)
swin_model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)  # No classification head
swin_model = swin_model.eval()  # We don't need to train the Swin Transformer

# Define the temporal model (LSTM)
class TemporalModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super(TemporalModel, self).__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # We only need the output of the last time step
        return out

# Parameters for LSTM
feature_dim = 1024  # The dimension of features extracted by Swin Transformer
hidden_dim = 512
num_layers = 2
temporal_model = TemporalModel(feature_dim, hidden_dim, num_layers)
temporal_model = temporal_model.train()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(temporal_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    temporal_model.train()
    train_loss = 0.0
    for inputs in tqdm(train_loader):
        inputs = inputs.float()  # Convert inputs to float32
        batch_size, num_frames, channels, height, width = inputs.size()
        inputs = inputs.view(-1, channels, height, width)  # Merge batch and frames for Swin Transformer

        with torch.no_grad():
            features = swin_model(inputs)  # Extract features with Swin Transformer

        features = features.view(batch_size, num_frames, -1)  # Reshape back to (batch_size, num_frames, feature_dim)
        features = features.float()

        optimizer.zero_grad()
        outputs = temporal_model(features)
        loss = criterion(outputs, features[:, -1, :])  # Compare with the last frame's features
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)

    # Validation loop
    temporal_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.float()  # Convert inputs to float32
            batch_size, num_frames, channels, height, width = inputs.size()
            inputs = inputs.view(-1, channels, height, width)  # Merge batch and frames for Swin Transformer

            with torch.no_grad():
                features = swin_model(inputs)  # Extract features with Swin Transformer

            features = features.view(batch_size, num_frames, -1)  # Reshape back to (batch_size, num_frames, feature_dim)
            features = features.float()

            outputs = temporal_model(features)
            loss = criterion(outputs, features[:, -1, :])  # Compare with the last frame's features
            val_loss += loss.item() * inputs.size(0)
    
    val_loss = val_loss / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model checkpoint
    torch.save(temporal_model.state_dict(), model_path)

print("Training complete. Model saved.")




//Anormaly model 

import torch
import torch.nn as nn
import yaml

# Define the anomaly detection model
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get the input dimension from the configuration
input_dim = config['training']['input_dim']

# Instantiate the model
model = AnomalyDetector(input_dim)

# Load the state dictionary
model_path = config['training']['anomaly_model_path']
model.load_state_dict(torch.load(model_path))

# Set the model to evaluation mode
model.eval()

# Print the model structure
print(model)

# Print the input dimension
print("Input dimension: ", input_dim)

(venv) PS D:\model_creations\vad> python scripts/test2.py
AnomalyDetector(
  (encoder): Sequential(
    (0): Linear(in_features=163213, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=128, bias=True)
    (3): ReLU()
  )
  (decoder): Sequential(
    (0): Linear(in_features=128, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=163213, bias=True)
    (3): Sigmoid()
  )
)
Input dimension:  163213
(venv) PS D:\model_creations\vad> 