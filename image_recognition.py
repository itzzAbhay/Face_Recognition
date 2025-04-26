import os
import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
MAX_IMAGE_SIZE = (320, 240)
MODEL_NAME = "Facenet"

def load_dataset(folder_path):
    images, labels = [], []
    label_map = {}
    
    try:
        for i, person_name in enumerate(sorted(os.listdir(folder_path))):
            person_folder = os.path.join(folder_path, person_name)
            if not os.path.isdir(person_folder):
                continue
                
            label_map[i] = person_name
            for file in os.listdir(person_folder):
                file_path = os.path.join(person_folder, file)
                try:
                    image = cv2.imread(file_path)
                    if image is not None:
                        image = cv2.resize(image, MAX_IMAGE_SIZE)
                        images.append(image)
                        labels.append(i)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        
    return images, labels, label_map

def extract_encodings(images):
    encodings = []
    
    for img in images:
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            embedding = DeepFace.represent(
                rgb_img,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend="opencv",
            )
            
            if embedding:
                encodings.append(embedding[0]["embedding"])
            else:
                encodings.append(None)
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            encodings.append(None)
            
    return encodings

@app.route('/recognize', methods=['POST'])
def recognize_faces():
    try:
        # Training phase
        train_images, train_labels, label_map = load_dataset("dataset/train")
        train_encodings = extract_encodings(train_images)
        train_encodings = [e for e in train_encodings if e is not None]
        train_labels = [l for e, l in zip(train_encodings, train_labels) if e is not None]
        
        if not train_encodings:
            return jsonify({"error": "No valid face encodings in training set"}), 400

        # Train model
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(train_encodings, train_labels)
        
        # Testing phase
        test_images, test_labels, _ = load_dataset("dataset/test")
        test_encodings = extract_encodings(test_images)
        test_encodings = [e for e in test_encodings if e is not None]
        test_labels = [l for e, l in zip(test_encodings, test_labels) if e is not None]
        
        if not test_encodings:
            return jsonify({"error": "No valid face encodings in test set"}), 400

        # Predictions
        predictions = knn.predict(test_encodings)
        probabilities = knn.predict_proba(test_encodings)

        # Format results
        results = []
        for idx, (pred_label, probs) in enumerate(zip(predictions, probabilities)):
            confidence = probs[pred_label] * 100
            matched_name = label_map[pred_label]
            results.append({
                "test_id": idx+1,
                "matched_name": matched_name,
                "confidence": round(confidence, 2)
            })

        # Evaluation metrics
        labels_present = unique_labels(test_labels, predictions)
        report = classification_report(
            test_labels, predictions,
            labels=labels_present,
            target_names=[label_map[i] for i in labels_present],
            zero_division=0,
            output_dict=True
        )

        return jsonify({
            "results": results,
            "accuracy": round(accuracy_score(test_labels, predictions)*100, 2),
            "report": report
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
