from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2

app = FastAPI()

# Load HaarCascade (comes bundled with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def read_image(uploaded: UploadFile):
    img_bytes = uploaded.file.read()
    img = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    return face, [int(x), int(y), int(x+w), int(y+h)]

def get_features(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def compute_similarity(des1, des2):
    if des1 is None or des2 is None:
        return 0

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return len(matches)

@app.post("/verify")
async def verify(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    img1 = read_image(image1)
    img2 = read_image(image2)

    face1, bbox1 = detect_face(img1)
    face2, bbox2 = detect_face(img2)

    if face1 is None or face2 is None:
        return JSONResponse({"error": "Face not detected in one or both images"})

    k1, d1 = get_features(face1)
    k2, d2 = get_features(face2)

    similarity = compute_similarity(d1, d2)

    # Simple threshold (tune this)
    result = "same person" if similarity > 75 else "different person"

    return {
        "verification_result": result,
        "similarity_score": similarity,
        "face1_bbox": bbox1,
        "face2_bbox": bbox2
    }
