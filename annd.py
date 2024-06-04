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

# Define the feature extractor (Swin Transformer)
swin_model = create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)  # No classification head
swin_model = swin_model.eval()  # We don't need to train the Swin Transformer

# Parameters for LSTM
feature_dim = 1024  # The dimension of features extracted by Swin Transformer
hidden_dim = 512
num_layers = 2
temporal_model = TemporalModel(feature_dim, hidden_dim, num_layers)
temporal_model.load_state_dict(torch.load(model_path))
temporal_model = temporal_model.eval()

cap = cv2.VideoCapture("test_video.avi")
print(cap.isOpened())

while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()

    for i in range(10):
        ret,frame=cap.read()
        frame=cv2.resize(frame, (227,227), interpolation = cv2.INTER_AREA)
        gray=0.2989*frame[:,:,0]+0.5870*frame[:,:,1]+0.1140*frame[:,:,2]
        gray=(gray-gray.mean())/gray.std()
        gray=np.clip(gray,0,1)
        imagedump.append(gray)

    imagedump=np.array(imagedump)
    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=3)

    with torch.no_grad():
        features = swin_model(torch.from_numpy(imagedump).float())
        features = features.view(1, 10, -1)  # Reshape to (batch_size, num_frames, feature_dim)
        outputs = temporal_model(features)

    loss = np.mean((outputs.detach().numpy() - features[:, -1, :].detach().numpy())**2)

    if frame.any()==None:
        print("none")

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
    if loss>anomaly_threshold:
        print('Abnormal Event Detected')
        cv2.putText(frame,"Abnormal Event",(220,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
    elif loss<anomaly_threshold:
        print('Normal')
        print(loss)
        cv2.putText(frame,"Normal Event",(220,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

    cv2.imshow("video",frame)

cap.release()
cv2.destroyAllWindows()
