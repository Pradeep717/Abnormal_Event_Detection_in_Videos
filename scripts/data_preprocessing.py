import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

raw_data_dir = config['data']['raw_dir']
processed_data_dir = config['data']['processed_dir']
frame_height = config['data']['frame_height']
frame_width = config['data']['frame_width']
fps = config['data']['fps']

store_image = []
train_videos = os.listdir(raw_data_dir)

def store_in_array(image_path):
    image = load_img(image_path)
    image = img_to_array(image)
    image = cv2.resize(image, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    store_image.append(gray)

# Create processed data directory if not exists
os.makedirs(processed_data_dir, exist_ok=True)

for video in train_videos:
    video_path = os.path.join(raw_data_dir, video)
    video_name = os.path.splitext(video)[0]
    train_images_path = os.path.join(processed_data_dir, 'frames', video_name)
    os.makedirs(train_images_path, exist_ok=True)
    
    # Extract frames using ffmpeg
    os.system(f'ffmpeg -i {video_path} -r {fps} {train_images_path}/%03d.jpg')
    
    images = os.listdir(train_images_path)
    for image in images:
        image_path = os.path.join(train_images_path, image)
        store_in_array(image_path)

store_image = np.array(store_image)
a, b, c = store_image.shape
store_image.resize(b, c, a)
store_image = (store_image - store_image.mean()) / (store_image.std())
store_image = np.clip(store_image, 0, 1)
np.save(os.path.join(processed_data_dir, 'training.npy'), store_image)
