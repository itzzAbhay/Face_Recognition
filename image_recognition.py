import os
import cv2
import numpy as np
from deepface import DeepFace
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Limit image size to prevent memory issues
MAX_IMAGE_SIZE = (640, 480)  

def load_dataset(folder_path):
    """Load images and labels from dataset folder"""
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
                        # Resize to conserve memory
                        image = cv2.resize(image, MAX_IMAGE_SIZE)
                        images.append(image)
                        labels.append(i)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        
    return images, labels, label_map

def extract_encodings(images):
    """Extract face embeddings using DeepFace"""
    encodings = []
    
    for img in images:
        try:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use lighter model for production
            embedding = DeepFace.represent(
                rgb_img,
                model_name="Facenet",  # or "OpenFace" for lighter model
                enforce_detection=False,
                detector_backend="opencv",
            )
            
            if embedding:
                encodings.append(embedding[0]["embedding"])
            else:
                encodings.append(None)
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            encodings.append(None)
            
    return encodings

# ... (rest of your existing functions remain the same) ...

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            result = recognize_faces()
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            result = f"An error occurred: {str(e)}"
            
    return render_template('index5.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
