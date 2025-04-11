import cv2
from insightface.app import FaceAnalysis


def get_face_embedding(frame):
    faceapp = FaceAnalysis(name='buffalo_sc', root='insightface_model', providers=[
                           'CPUExecutionProvider'])
    faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    results = faceapp.get(frame, max_num=1)
    return results[0]['embedding'] if results else None
