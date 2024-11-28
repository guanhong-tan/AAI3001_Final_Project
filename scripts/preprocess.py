import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
IMAGE_SIZE = (128, 128)
DATASET_PATH = 'C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/data'
CATEGORIES = ['Glass', 'Clothing','Plastic', 'Paper', 'Metal']

def load_and_preprocess_data():
    data = []
    labels = []

    for idx, category in enumerate(CATEGORIES):
        category_path = os.path.join(DATASET_PATH, category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            try:
                img = cv2.imread(image_path)
                img = cv2.resize(img, IMAGE_SIZE)
                data.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
    
    data = np.array(data) / 255.0  # Normalize
    labels = np.array(labels)

    return train_test_split(data, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    output_dir = 'C:/Users/bryan/OneDrive/Desktop/Project/AAI3001_Final_Project/model'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving files to: {os.path.abspath(output_dir)}")  
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print("Preprocessing completed and data saved.")
