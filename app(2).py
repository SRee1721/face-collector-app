from flask import Flask, render_template, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from face_utils import collect_face_embeddings

cred = credentials.Certificate('ServiceAccountKey.json')
firebase_admin.initialize_app(cred)
store = firestore.client()

app = Flask(__name__)
COLLECTION_NAME = "academy:register"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start-face-collection', methods=['POST'])
def start_face_collection():
    data = request.get_json()
    name = data['name']
    role = data['role']

    name_role, facial_features = collect_face_embeddings(name, role)

    if facial_features.size == 0:
        return jsonify({"message": "No faces detected."})

    doc_ref = store.collection(COLLECTION_NAME).document('facial_features')
    doc = doc_ref.get()

    facial_features_bytes = facial_features.tobytes()

    if doc.exists:
        existing_data = doc.to_dict()
        if name_role in existing_data:
            return jsonify({"message": "User already exists."})
        else:
            doc_ref.set({name_role: facial_features_bytes}, merge=True)
            return jsonify({"message": "Face data added."})
    else:
        doc_ref.set({name_role: facial_features_bytes})
        return jsonify({"message": "Document created with first face data."})


if __name__ == '__main__':
    app.run(debug=True)
