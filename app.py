
'''cred = credentials.Certificate('/etc/secrets/ServiceAccountKey.json')
firebase_admin.initialize_app(cred)
store = firestore.client()

app = Flask(__name__)
COLLECTION_NAME = "academy:register"'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import firebase_admin
from firebase_admin import credentials, firestore
from insightface.app import FaceAnalysis
import os
import base64

# Init Flask
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains (or restrict using origins=[])

# Load Firebase credentials from Render ENV file

cred = credentials.Certificate('/etc/secrets/ServiceAccountKey.json')
firebase_admin.initialize_app(cred)
store = firestore.client()
COLLECTION_NAME = "academy:register"
# InsightFace setup
faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

@app.route('/collect-face', methods=['POST'])
def collect_face():
    data = request.json
    person_name = data.get('name')
    role = data.get('role')

    if not person_name or not role:
        return jsonify({'error': 'Missing name or role'}), 400

    key = f"{person_name}@{role.upper()}"
    cap = cv2.VideoCapture(0)

    sample = 0
    face_embeddings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = faceapp.get(frame, max_num=1)
        for res in results:
            sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            embeddings = res['embedding']
            face_embeddings.append(embeddings)
            if sample >= 700:
                break

        if sample >= 700:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(face_embeddings) > 0:
        facial_features = np.asarray(face_embeddings).mean(axis=0)
        facial_features_bytes = facial_features.tobytes()

        doc_ref = store.collection(COLLECTION_NAME).document("facial_features")
        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            if key not in data:
                doc_ref.set({key: facial_features_bytes}, merge=True)
                return jsonify({"message": "Face data added"}), 200
            else:
                return jsonify({"message": "Face data already exists"}), 200
        else:
            doc_ref.set({key: facial_features_bytes})
            return jsonify({"message": "Document created with first face data"}), 200
    else:
        return jsonify({"error": "No face detected"}), 400

@app.route('/')
def home():
    return "Backend is running"

if __name__ == '__main__':
    app.run(debug=True)
