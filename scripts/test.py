import cv2
import numpy as np
import torch
from timm import create_model
from train_transformer import TemporalModel
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_path = config['training']['transformer_model_path']
anomaly_threshold = config['data']['anomaly_threshold']
test_video_path = config['data']['test_video_path']

swin_model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
swin_model.eval()

feature_dim = 1024
hidden_dim = 512
num_layers = 2
temporal_model = TemporalModel(feature_dim, hidden_dim, num_layers)
temporal_model.load_state_dict(torch.load(model_path))
temporal_model.eval()

cap = cv2.VideoCapture(test_video_path)
print(cap.isOpened())

while cap.isOpened():
    imagedump = []
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        imagedump.append(gray)

    if len(imagedump) < 10:
        break

    imagedump = np.array(imagedump)
    imagedump = np.expand_dims(imagedump, axis=0)
    imagedump = np.expand_dims(imagedump, axis=2)
    imagedump = np.repeat(imagedump, 3, axis=2)

    with torch.no_grad():
        inputs = torch.from_numpy(imagedump).float()
        inputs = inputs.view(-1, 3, 224, 224)
        features = swin_model(inputs)
        features = features.view(1, 10, -1)
        outputs = temporal_model(features)

    loss = np.mean((outputs.detach().numpy() - features[:, -1, :].detach().numpy())**2)

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
