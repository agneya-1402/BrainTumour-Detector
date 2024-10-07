import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt

def load_and_preprocess_images(folder_path):

    images = []
    labels = []
    original_images = []  

    for label, subfolder in enumerate(['no', 'yes']):
        path = os.path.join(folder_path, subfolder)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                original_images.append(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                images.append(resized.flatten())
                labels.append(label)
    return np.array(images), np.array(labels), original_images

# Canny Edge Detector
def extract_features(images):
    features = []
    for img in images:
        img = img.reshape(64, 64)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(img, 100, 200)
        features.append(edges.flatten())
    return np.array(features)

# SVM Classifier on extracted features
def train_model(X_train, y_train):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    return model

# accuracy
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# output
def visualize_detection(original_image, processed_image, prediction, probability):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Processed image with detection
    ax2.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    if prediction == 1:  # If tumor detected
        h, w = original_image.shape[:2]
        cv2.rectangle(original_image, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (0, 0, 255), 2)
        ax2.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax2.set_title(f"Tumor Detected (Probability: {probability:.2f})")
    else:
        ax2.set_title(f"No Tumor Detected (Probability: {1-probability:.2f})")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    dataset_path = 'BrainTumour_Detector/dataset/brain_tumor_dataset'
    
    # proprocess images
    images, labels, original_images = load_and_preprocess_images(dataset_path)
    
    # Extract features
    features = extract_features(images)
    
    # Splitting data into train nd test
    X_train, X_test, y_train, y_test, orig_train, orig_test = train_test_split(
        features, labels, original_images, test_size=0.2, random_state=42)
    
    # Training model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Visualize some predictions
    for i in range(min(5, len(X_test))):  # Visualize 5 test images
        prediction = model.predict([X_test[i]])[0]
        probability = model.predict_proba([X_test[i]])[0][1]  # Probability
        visualize_detection(orig_test[i], X_test[i].reshape(64, 64), prediction, probability)

if __name__ == "__main__":
    main()