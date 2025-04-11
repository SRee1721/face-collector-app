
from flask import Flask, request, jsonify, render_template
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
from insightface.app import FaceAnalysis

app = Flask(__name__)
cred = credentials.Certificate("ServiceAccountKey.json")
firebase_admin.initialize_app(cred)
store = firestore.client()
COLLECTION_NAME = "academy:register"

faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=[
                       'CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# Shared session state
received_embeddings = []
sample_limit = 400
current_name_role = None


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/start-face-collection', methods=['POST'])
def start_face_collection():
    global received_embeddings, current_name_role
    data = request.get_json()
    name = data['name']
    role = data['role']
    current_name_role = f"{name}@{role}"
    received_embeddings = []
    return jsonify({"message": "Collection started"})


@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    global received_embeddings, current_name_role

    if not current_name_role:
        return jsonify({"error": "Collection not started"}), 400

    file = request.files['frame']
    img = Image.open(BytesIO(file.read()))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    results = faceapp.get(frame, max_num=1)

    if results:
        embedding = results[0]['embedding']
        if len(received_embeddings) < sample_limit:
            received_embeddings.append(embedding)
            print(f"Sample {len(received_embeddings)}/{sample_limit}")

    if len(received_embeddings) == sample_limit:
        final_embedding = np.mean(received_embeddings, axis=0)
        received_embeddings = []

        doc_ref = store.collection(COLLECTION_NAME).document("facial_features")
        doc = doc_ref.get()
        embedding_bytes = final_embedding.tobytes()

        if doc.exists:
            existing = doc.to_dict()
            if current_name_role in existing:
                print("User already exists.")
            else:
                doc_ref.set({current_name_role: embedding_bytes}, merge=True)
        else:
            doc_ref.set({current_name_role: embedding_bytes})

        print(" Face data saved for:", current_name_role)
        current_name_role = None
        return jsonify({"done": True})

    return jsonify({"done": False})


if __name__ == '__main__':
    app.run(debug=True)
