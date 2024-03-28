from flask import Flask, render_template, request
import face_recognition
import cv2
import numpy as np
import os


app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello_world():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def face_recog():

    imagefile = request.form["imagePath"].strip()
    userId = request.form["id"].upper()
    image_path = "../../photos/" + imagefile

    trained_model = np.load("model/trained_model.npz")
    known_face_encodings = trained_model["known_face_encodings"]
    known_face_names = trained_model["known_face_names"]

    # Load the image
    frame = cv2.imread(image_path)

    # Resize frame of the image to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    # Find all the faces and face encodings in the current frame of the image
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        is_matched = False

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            name = os.path.splitext(os.path.basename(name))[0]
            if name == userId:
                is_matched = True

    # return face_names
    if is_matched:
        response = {"status": 200, "identification": name}

    else:
        response = {"status": 404, "identification": ""}

    return response


if __name__ == "__main__":
    app.run(port=5510, debug=True)
