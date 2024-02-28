import face_recognition
from flask import Flask, render_template, request
from os.path import join, dirname, abspath
import numpy as np
import cv2

filename = join(dirname(__file__), "model", "predictor.pkl")
current_dir = dirname(abspath(__file__))

model = np.load(open(filename, "rb"))
known_face_encodings = model['known_face_encodings']
known_face_names = model['known_face_names']

app = Flask(__name__)


@app.route("/")
def hello_word():
    return {"ok" : "initialised"}


@app.route("/submit", methods=["POST"])
def recognise_face():

    photo_url = str(request.form['photoURL'])
    file_path = join(current_dir, '..', 'backend', 'photos', photo_url)
    frame = cv2.imread(file_path)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    
    if len(face_names) > 0:
        return {"status" : 200, "identification": face_names}
    return {"status" : 404, "identification" : []}

if __name__ == "__main__":
    app.run(debug=True)