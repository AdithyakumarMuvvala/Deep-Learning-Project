import os
import cv2
import numpy as np
import random
from glob import glob
from sklearn.model_selection import train_test_split

IMG_SIZE = (128, 128)
CLASSES = ['clean', 'rust', 'broken', 'dusty']

def generate_synthetic_data(data_dir, num_samples=200):
    """
    Generates synthetic industrial surface images.
    """
    print(f"Generating {num_samples} synthetic images per class in {data_dir}...")
    
    for label in CLASSES:
        os.makedirs(os.path.join(data_dir, label), exist_ok=True)
        
        for i in range(num_samples):
            # 1. Base texture (industrial metal look, RGB)
            # Create a gray base
            base = np.random.normal(128, 20, IMG_SIZE).astype(np.uint8)
            img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            if label == 'rust':
                # Add orange/brown noise patches
                rust_layer = np.zeros_like(img)
                rust_layer[:, :] = (30, 70, 160) # BGR for Orange-ish Brown
                
                mask = np.random.rand(*IMG_SIZE) > 0.7  # Random patches
                mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
                mask = np.dstack([mask]*3)
                
                # Blend rust onto metal
                img = (img * (1 - mask) + rust_layer * mask).astype(np.uint8)
                # Add some granular noise for texture
                noise = np.random.randint(0, 50, img.shape, dtype='uint8')
                img = cv2.add(img, noise)

            elif label == 'broken':
                # Add large jagged cracks (black)
                num_cracks = random.randint(1, 3)
                for _ in range(num_cracks):
                    pts = []
                    start_x = random.randint(0, IMG_SIZE[1])
                    start_y = random.randint(0, IMG_SIZE[0])
                    pts.append([start_x, start_y])
                    
                    segments = random.randint(3, 8)
                    for _ in range(segments):
                        start_x += random.randint(-20, 20)
                        start_y += random.randint(-20, 20)
                        pts.append([start_x, start_y])
                    
                    pts = np.array(pts, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], False, (10, 10, 10), thickness=random.randint(2, 5))

            elif label == 'dusty':
                # Add white/gray noise overlay and reduce contrast
                noise = np.random.normal(0, 20, img.shape)
                img = img.astype(np.float32) + noise
                img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Whiten (add constant value)
                overlay = np.full(img.shape, 200, dtype=np.uint8)
                img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
            
            # Save
            filepath = os.path.join(data_dir, label, f"{label}_{i}.jpg")
            cv2.imwrite(filepath, img)
            
    print("Data generation complete.")

def load_data(data_dir, split=0.2):
    """
    Loads images and labels from folders.
    Returns: (X_train, X_test, y_train, y_test, class_names)
    """
    images = []
    labels = []
    
    # Reload classes from dir to ensuring matching indices
    class_names = sorted(CLASSES)
    class_map = {cls: i for i, cls in enumerate(class_names)}
    
    for cls in class_names:
        path = os.path.join(data_dir, cls, "*")
        files = glob(path)
        for f in files:
            try:
                img = cv2.imread(f) # Default reads as BGR
                if img is None: continue
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(class_map[cls])
            except Exception as e:
                print(f"Error loading {f}: {e}")
                
    X = np.array(images)
    y = np.array(labels)
    
    # Shuffle and split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=42)
    return X_train, X_test, y_train, y_test, class_names
