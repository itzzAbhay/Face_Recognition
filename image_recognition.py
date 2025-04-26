import os
import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

app = Flask(__name__)

def load_dataset(folder_path):
    images, labels = [], []
    label_map = {}

    for i, person_name in enumerate(sorted(os.listdir(folder_path))):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        label_map[i] = person_name
        for file in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file)
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
                labels.append(i)
    return images, labels, label_map

def extract_encodings(images):
    encodings = []
    for img in images:
        try:
            # DeepFace expects RGB format
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Extract face embedding (512D vector)
            embedding = DeepFace.represent(
                rgb_img,
                model_name="Facenet",  # or "VGG-Face", "OpenFace", etc.
                enforce_detection=False,  # Skip if no face detected
                detector_backend="opencv",  # Faster than default
            )
            if embedding:
                encodings.append(embedding[0]["embedding"])
            else:
                encodings.append(None)
        except Exception as e:
            print(f"Error processing image: {e}")
            encodings.append(None)
    return encodings

def filter_valid_encodings(encodings, labels):
    filtered_encodings, filtered_labels = [], []
    for enc, label in zip(encodings, labels):
        if enc is not None:
            filtered_encodings.append(enc)
            filtered_labels.append(label)
    return filtered_encodings, filtered_labels

def recognize_faces():
    output = []

    output.append("Loading the training data...")
    train_images, train_labels, label_map = load_dataset("dataset/train")
    train_encodings = extract_encodings(train_images)
    train_encodings, train_labels = filter_valid_encodings(train_encodings, train_labels)
    output.append(f"{len(train_encodings)} valid face encodings found for training.")

    if not train_encodings:
        return "No valid face encodings found in training set."

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_encodings, train_labels)
    output.append("Training KNN classifier...")

    output.append("Loading test data...")
    test_images, test_labels, _ = load_dataset("dataset/test")
    test_encodings = extract_encodings(test_images)
    test_encodings, test_labels = filter_valid_encodings(test_encodings, test_labels)
    output.append(f"{len(test_encodings)} valid face encodings found for testing.")

    if not test_encodings:
        return "No valid face encodings found in test set."

    output.append("Predicting and evaluating...")
    predictions = knn.predict(test_encodings)
    probabilities = knn.predict_proba(test_encodings)

    output.append("\nMatched Results with Confidence:")
    for idx, (pred_label, probs) in enumerate(zip(predictions, probabilities)):
        try:
            class_index = list(knn.classes_).index(pred_label)
            confidence = probs[class_index] * 100
            matched_name = label_map[pred_label]
            output.append(f"Test Image {idx + 1}: Matched with {matched_name} ({confidence:.2f}% confidence)")
        except ValueError:
            output.append(f"Test Image {idx + 1}: Unknown person (No matching class found)")

    output.append("\nThe Classification Report:")
    labels_present = unique_labels(test_labels, predictions)
    report = classification_report(
        test_labels, predictions,
        labels=labels_present,
        target_names=[label_map[i] for i in labels_present],
        zero_division=0
    )
    output.append(report)
    output.append(f"Accuracy: {accuracy_score(test_labels, predictions) * 100:.2f}%")

    return "\n".join(output)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        result = recognize_faces()
    return render_template('index5.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)