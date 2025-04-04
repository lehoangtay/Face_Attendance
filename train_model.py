import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pickle

# Chuẩn bị dữ liệu huấn luyện
data_path = "dataset/"
onlyfolders = [join(data_path, d, f) for d in listdir(data_path) 
               for f in listdir(join(data_path, d)) if not isfile(join(data_path, d, f))]

training_data, labels = [], []
label_dict = {}
current_id = 0

for folder in onlyfolders:
    label = folder.split("dataset/")[1]  # Lấy ClassX/student_id_student_name
    if label not in label_dict.values():
        label_dict[current_id] = label
        current_id += 1
    folder_path = folder
    for file in listdir(folder_path):
        if file.endswith(".jpg"):
            image_path = join(folder_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            training_data.append(image)
            labels.append(list(label_dict.keys())[list(label_dict.values()).index(label)])

labels = np.array(labels)

# Huấn luyện mô hình
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(training_data, labels)
recognizer.save("trained_model/trained_faces.yml")

# Lưu label_dict
with open("trained_model/labels.pkl", "wb") as f:
    pickle.dump(label_dict, f)

print("Huấn luyện hoàn tất!")