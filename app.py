from flask import Flask, request, jsonify, send_from_directory
import base64
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)
face_db = []

def capture_face(prompt="Press SPACE to capture"):
    cap = cv2.VideoCapture(0)
    print(prompt)

    while True:
        ret, frame = cap.read()
        cv2.imshow(prompt, frame)
        key = cv2.waitKey(1)
        if key == 32:  # SPACE
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise Exception("Capture cancelled.")

def get_embedding(img: np.ndarray):
    result = DeepFace.represent(img_path=img, enforce_detection=True)[0]
    return np.array(result["embedding"])

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")

@app.route("/store", methods=["POST"])
def store_face_and_file():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    try:
        frame = capture_face("Store Mode - Press SPACE")
        embedding = get_embedding(frame)

        file = request.files["file"]
        file_b64 = base64.b64encode(file.read()).decode("utf-8")

        face_db.append({"embedding": embedding, "file": file_b64})
        return jsonify({"status": "Face and file stored"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/match", methods=["GET"])
def match_face():
    try:
        frame = capture_face("Match Mode - Press SPACE")
        query_embedding = get_embedding(frame)

        for record in face_db:
            score = cosine_similarity(query_embedding, record["embedding"])
            if score > 0.95:
                return jsonify({
                    "status": "Match found",
                    "file_base64": record["file"]
                }), 200

        return jsonify({"status": "No match found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
