from flask import Flask, render_template, Response, jsonify, send_file, request
import cv2
import numpy as np
import sqlite3
import pickle
from datetime import datetime
import pandas as pd
import os
import shutil
from os import listdir
from os.path import join, isfile

app = Flask(__name__)

# Biến toàn cục cho mô hình và nhãn
recognizer = cv2.face.LBPHFaceRecognizer_create()
label_dict = {}

# Load mô hình và nhãn khi khởi động
def load_model():
    global recognizer, label_dict
    if os.path.exists("trained_model/trained_faces.yml"):
        recognizer.read("trained_model/trained_faces.yml")
    if os.path.exists("trained_model/labels.pkl"):
        with open("trained_model/labels.pkl", "rb") as f:
            label_dict = pickle.load(f)
    print("Loaded label_dict:", label_dict)  # Debug khi khởi động

load_model()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Khởi tạo database
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance 
                 (id INTEGER PRIMARY KEY, student_id TEXT, student_name TEXT, class_name TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Hàm kiểm tra điểm danh hôm nay
def check_attendance_today(student_id, class_name):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT timestamp FROM attendance WHERE student_id = ? AND class_name = ? AND timestamp LIKE ?", 
              (student_id, class_name, f"{today}%"))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

# Hàm lấy danh sách sinh viên
def get_students(class_name):
    class_path = f"dataset/{class_name}/"
    students = []
    if os.path.exists(class_path):
        for folder in os.listdir(class_path):
            if os.path.isdir(os.path.join(class_path, folder)):
                student_id, student_name = folder.split("_", 1)
                students.append({"student_id": student_id, "student_name": student_name})
    return students

# Hàm kiểm tra sinh viên thuộc lớp
def is_student_in_class(student_id, student_name, class_name):
    folder_name = f"{student_id}_{student_name}"
    class_path = f"dataset/{class_name}/{folder_name}"
    exists = os.path.exists(class_path)
    print(f"Checking if {folder_name} belongs to {class_name}: {exists} (Path: {class_path})")  # Debug
    return exists

# Thu thập ảnh khuôn mặt qua web
def collect_faces(class_name, student_id, student_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield jsonify({"error": "Cannot open webcam / Không thể mở webcam"})
        return

    dataset_dir = f"dataset/{class_name}/{student_id}_{student_name}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if count < 200:
                count += 1
                face = gray[y:y+h, x:x+w]
                file_name = f"{dataset_dir}/{count}.jpg"
                cv2.imwrite(file_name, face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Collecting: {count}/200", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Collected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if count >= 200:
            break

    cap.release()

# Huấn luyện mô hình
def train_model():
    global recognizer, label_dict
    data_path = "dataset/"
    if not os.path.exists(data_path):
        return "The directory dataset/ does not exist!"

    try:
        onlyfolders = [join(data_path, d, f) for d in listdir(data_path) 
                       for f in listdir(join(data_path, d)) if not isfile(join(data_path, d, f))]
    except Exception as e:
        return f"Error accessing dataset/: {str(e)}"

    if not onlyfolders:
        return "No data to train in dataset/!"

    training_data, labels = [], []
    label_dict = {}  # Reset label_dict
    current_id = 0

    for folder in onlyfolders:
        label = folder.split("dataset/")[1] 
        if label not in label_dict.values():
            label_dict[current_id] = label
            current_id += 1
        folder_path = folder
        for file in listdir(folder_path):
            if file.endswith(".jpg"):
                image_path = join(folder_path, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    training_data.append(image)
                    labels.append(list(label_dict.keys())[list(label_dict.values()).index(label)])

    if not training_data:
        return "No valid images for training!"

    try:
        labels = np.array(labels)
        recognizer.train(training_data, labels)
        recognizer.save("trained_model/trained_faces.yml")
        with open("trained_model/labels.pkl", "wb") as f:
            pickle.dump(label_dict, f)
        print("Label dict after training:", label_dict)  # Debug
        load_model()  # Tải lại mô hình ngay sau khi huấn luyện
        return "Training complete!"
    except Exception as e:
        return f"Error while training the model: {str(e)}"

# Video điểm danh
def gen_frames(class_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nError: Cannot open webcam\r\n'
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")  # Debug
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            print(f"Predicted label: {label}, Confidence: {confidence}, Label dict: {label_dict}")  # Debug
            if confidence < 100 and label in label_dict:
                # Tách student_info tương thích với cả "/" và "\"
                student_info = label_dict[label].replace("\\", "/").split("/", 1)[1]  # Chuyển \ thành / rồi tách
                student_id, student_name = student_info.split("_", 1)
                print(f"Recognized: {student_id}_{student_name} in class {class_name}")  # Debug
                
                if not is_student_in_class(student_id, student_name, class_name):
                    text = "Warning"
                    color = (0, 0, 255)
                elif check_attendance_today(student_id, class_name):
                    text = f"{student_name} - Attended"
                    color = (0, 255, 255)
                else:
                    text = f"{student_name} ({confidence:.2f})"
                    color = (0, 255, 0)
                    conn = sqlite3.connect("attendance.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO attendance (student_id, student_name, class_name, timestamp) VALUES (?, ?, ?, ?)",
                              (student_id, student_name, class_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    conn.close()
            else:
                text = "Unknown" if confidence >= 100 else "Invalid label"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame")  # Debug
            break
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/class/<class_name>')
def class_page(class_name):
    students = get_students(class_name)
    return render_template('class.html', class_name=class_name, students=students)

@app.route('/video_feed/<class_name>')
def video_feed(class_name):
    return Response(gen_frames(class_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/collect_faces/<class_name>/<student_id>/<student_name>')
def collect_faces_route(class_name, student_id, student_name):
    return Response(collect_faces(class_name, student_id, student_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train_model', methods=['GET'])
def train_model_route():
    message = train_model()
    return jsonify({"message": message})

@app.route('/attendance_status')
def get_attendance_status():
    class_name = request.args.get('class')
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT student_id, timestamp FROM attendance WHERE class_name = ? AND timestamp LIKE ?", 
              (class_name, f"{today}%"))
    attendance_data = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    students = get_students(class_name)
    for student in students:
        timestamp = attendance_data.get(student["student_id"])
        student["status"] = "Attended" if timestamp else "Off"
        student["timestamp"] = timestamp if timestamp else "N/A"
    return jsonify(students)

@app.route('/export_excel')
def export_excel():
    class_name = request.args.get('class')
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("SELECT student_id, timestamp FROM attendance WHERE class_name = ? AND timestamp LIKE ?", 
              (class_name, f"{today}%"))
    attendance_data = {row[0]: row[1] for row in c.fetchall()}
    
    students = get_students(class_name)
    for student in students:
        timestamp = attendance_data.get(student["student_id"])
        student["status"] = "Attended" if timestamp else "Off"
        student["timestamp"] = timestamp if timestamp else "N/A"

    df = pd.DataFrame(students, columns=["student_id", "student_name", "status", "timestamp"])
    df.columns = ["Student ID", "Student Name", "Status", "Time"]
    excel_file = f"attendance_{class_name}_{today}.xlsx"
    df.to_excel(excel_file, index=False)

    c.execute("DELETE FROM attendance WHERE class_name = ? AND timestamp LIKE ?", 
              (class_name, f"{today}%"))
    conn.commit()
    conn.close()

    return send_file(excel_file, as_attachment=True)

@app.route('/delete_student', methods=['POST'])
def delete_student():
    try:
        class_name = request.form.get('class')
        student_id = request.form.get('student_id')
        student_name = request.form.get('student_name')

        if not class_name or not student_id or not student_name:
            return jsonify({"message": "Missing required fields! / Thiếu các trường bắt buộc!"}), 400

        folder_name = f"{student_id}_{student_name}"
        student_path = f"dataset/{class_name}/{folder_name}"
        
        if os.path.exists(student_path):
            shutil.rmtree(student_path)  # Delete student's folder / Xóa thư mục của sinh viên
            # Remove student attendance records / Xóa bản ghi điểm danh của sinh viên
            conn = sqlite3.connect("attendance.db")
            c = conn.cursor()
            c.execute("DELETE FROM attendance WHERE student_id = ? AND class_name = ?", (student_id, class_name))
            conn.commit()
            conn.close()
            return jsonify({"message": f"Student {student_name} deleted successfully! Please retrain the model. / Sinh viên {student_name} đã được xóa thành công! Vui lòng huấn luyện lại mô hình."}), 200
        else:
            return jsonify({"message": "Student not found! / Sinh viên không tồn tại!"}), 404
    except Exception as e:
        return jsonify({"message": f"Error deleting student: {str(e)} / Lỗi khi xóa sinh viên: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)