import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        for image_name in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, image_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(0 if label == "non-wanted" else 1)
    images = np.array(images) / 255.0
    labels = np.array(labels)
    return train_test_split(images, labels, test_size=0.2, random_state=42)
