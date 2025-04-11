import cv2
import numpy as np
from insightface.app import FaceAnalysis


def collect_face_embeddings(person_name, role):
    faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=[
                           'CPUExecutionProvider'])
    faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

    name_role = f"{person_name}@{role}"
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
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q') or sample >= 700:
            break

    cap.release()
    cv2.destroyAllWindows()

    return name_role, np.asarray(face_embeddings).mean(axis=0) if face_embeddings else np.array([])
