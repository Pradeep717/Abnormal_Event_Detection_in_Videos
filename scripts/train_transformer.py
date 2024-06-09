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
import cv2

# Custom Dataset class
class VideoDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define the temporal model (LSTM)
class TemporalModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super(TemporalModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_model():
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_data_dir = config['data']['processed_dir']
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    model_path = config['training']['transformer_model_path']

    # Load preprocessed training data
    training_data = np.load(os.path.join(processed_data_dir, 'training.npy')).astype(np.float32)
    print(f"Training data shape: {training_data.shape}")

    num_frames = 10
    height, width, total_frames = training_data.shape
    num_samples = total_frames // num_frames

    # Reshape the data
    training_data = training_data[:, :, :num_samples * num_frames]
    training_data = training_data.reshape(num_samples, num_frames, height, width)
    training_data = np.expand_dims(training_data, axis=2)

    # Resize frames
    training_data_resized = np.zeros((num_samples, num_frames, 3, 224, 224), dtype=np.float32)
    for i in range(num_samples):
        for j in range(num_frames):
            resized_frame = cv2.resize(training_data[i, j, 0], (224, 224))
            training_data_resized[i, j, 0] = resized_frame
            training_data_resized[i, j, 1] = resized_frame
            training_data_resized[i, j, 2] = resized_frame

    train_data, val_data = train_test_split(training_data_resized, test_size=0.2, random_state=42)
    train_dataset = VideoDataset(train_data)
    val_dataset = VideoDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    swin_model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0).to(device)
    swin_model.eval()

    feature_dim = 1024
    hidden_dim = 512
    num_layers = 2
    temporal_model = TemporalModel(feature_dim, hidden_dim, num_layers).to(device)
    temporal_model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(temporal_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        temporal_model.train()
        train_loss = 0.0
        for inputs in tqdm(train_loader):
            inputs = inputs.to(device).float()
            batch_size, num_frames, channels, height, width = inputs.size()
            inputs = inputs.view(-1, channels, height, width)

            with torch.no_grad():
                features = swin_model(inputs)

            features = features.view(batch_size, num_frames, -1)
            features = features.to(device).float()

            optimizer.zero_grad()
            outputs = temporal_model(features)
            loss = criterion(outputs, features[:, -1, :])
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        temporal_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                inputs = inputs.to(device).float()
                batch_size, num_frames, channels, height, width = inputs.size()
                inputs = inputs.view(-1, channels, height, width)

                features = swin_model(inputs)
                features = features.view(batch_size, num_frames, -1)
                features = features.to(device).float()

                outputs = temporal_model(features)
                loss = criterion(outputs, features[:, -1, :])
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        torch.save(temporal_model.state_dict(), model_path)

    print("Training complete. Model saved.")

if __name__ == "__main__":
    train_model()