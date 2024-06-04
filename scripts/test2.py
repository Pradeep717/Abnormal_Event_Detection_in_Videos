import cv2
import numpy as np
import torch
import yaml
from train_anomaly import AnomalyDetector

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_path = config['training']['anomaly_model_path']
anomaly_threshold = config['data']['anomaly_threshold']
test_video_path = config['data']['test_video_path']
input_dim = config['training']['input_dim']

# Define and load the anomaly detection model
model = AnomalyDetector(input_dim)
model.load_state_dict(torch.load(model_path))
model.eval()

cap = cv2.VideoCapture(test_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize and convert frame to grayscale
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
    gray = (gray - gray.mean()) / gray.std()
    gray = np.clip(gray, 0, 1)
    
    # Flatten the gray frame and add to sequence
    gray_flattened = gray.flatten()
    frame_sequence.append(gray_flattened)

    # Ensure the sequence contains enough frames to match the input dimension
    if len(frame_sequence) * gray_flattened.size >= input_dim:
        # Concatenate frames to form the input
        input_data = np.concatenate(frame_sequence, axis=0)[:input_dim]
        frame_sequence.pop(0)  # Remove the oldest frame to maintain sequence length

        if input_data.shape[0] != input_dim:
            print(f"Error: Expected input dimension {input_dim}, but got {input_data.shape[0]}")
            continue

        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).view(-1, input_dim)
            output_tensor = model(input_tensor)
            loss = np.mean((output_tensor.numpy() - input_tensor.numpy())**2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if loss > anomaly_threshold:
            print('Abnormal Event Detected', loss)
            cv2.putText(frame, "Abnormal Event", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print('Normal', loss)
            cv2.putText(frame, "Normal Event", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video", frame)

cap.release()
cv2.destroyAllWindows()
