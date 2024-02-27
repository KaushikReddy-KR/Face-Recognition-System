import face_recognition
import numpy as np
import os
import glob

train_folder = "data_known/"

known_face_encodings = []
known_face_names = []   

list_of_files = [f for f in glob.glob(train_folder + '*.jpg')]
number_files = len(list_of_files)

names = list_of_files.copy()

for i in range(number_files):
    image = face_recognition.load_image_file(list_of_files[i])
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)

    names[i] = names[i].replace("data_known/", "")
    known_face_names.append(names[i])

np.savez("trained_model.npz", known_face_encodings=known_face_encodings, known_face_names=known_face_names)
