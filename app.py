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
import face_recognition  # Sử dụng face_recognition

app = Flask(__name__)

# Biến toàn cục để lưu trữ encoding của các sinh viên
encoding_dict = {}

# Load encoding khi khởi động
def load_encodings():
    global encoding_dict
    if os.path.exists("trained_model/encodings.pkl"):
        with open("trained_model/encodings.pkl", "rb") as f:
            encoding_dict = pickle.load(f)
    print("Loaded encoding_dict:", encoding_dict)  # Debug khi khởi động

load_encodings()

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

        # Sử dụng face_recognition để phát hiện khuôn mặt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
            if count < 50:  # Thu thập 50 ảnh
                count += 1
                face = frame[top:bottom, left:right]
                file_name = f"{dataset_dir}/{count}.jpg"
                cv2.imwrite(file_name, face)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Collecting: {count}/50", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Collected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if count >= 50:  # Dừng khi đủ 50 ảnh
            break

    cap.release()

# Huấn luyện mô hình (tạo encoding bằng face_recognition)
def train_model():
    global encoding_dict
    data_path = "dataset/"
    if not os.path.exists(data_path):
        return "The directory dataset/ does not exist!"

    try:
        # Lấy danh sách thư mục con (mỗi thư mục là một lớp)
        class_folders = [d for d in listdir(data_path) if not isfile(join(data_path, d))]
        if not class_folders:
            return "No classes found in dataset/!"

        encoding_dict.clear()  # Reset encoding_dict

        # Duyệt qua từng lớp
        for class_folder in class_folders:
            class_path = join(data_path, class_folder)
            # Lấy danh sách thư mục sinh viên trong lớp
            student_folders = [f for f in listdir(class_path) if not isfile(join(class_path, f))]
            for student_folder in student_folders:
                # Đường dẫn đầy đủ đến thư mục sinh viên
                student_path = join(class_path, student_folder)
                student_key = f"{class_folder}/{student_folder}"
                print(f"Processing student: {student_key}")  # Debug

                # Tính encoding cho sinh viên
                encodings = []
                for file in listdir(student_path):
                    if file.endswith(".jpg"):
                        image_path = join(student_path, file)
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:  # Đảm bảo tìm thấy khuôn mặt
                            encodings.append(face_encodings[0])
                        else:
                            print(f"No face found in image {image_path}")  # Debug

                if encodings:
                    # Lưu encoding trung bình hoặc encoding đầu tiên
                    encoding_dict[student_key] = encodings[0]  # Lấy encoding đầu tiên
                    print(f"Encoding calculated for {student_key}")  # Debug
                else:
                    print(f"No valid encodings for {student_key}")  # Debug

    except Exception as e:
        return f"Error accessing dataset/: {str(e)}"

    if not encoding_dict:
        return "No valid encodings generated!"

    # Tạo thư mục trained_model nếu chưa tồn tại
    if not os.path.exists("trained_model"):
        os.makedirs("trained_model")

    # Lưu encoding_dict
    try:
        with open("trained_model/encodings.pkl", "wb") as f:
            pickle.dump(encoding_dict, f)
        print("Encoding dict after training:", encoding_dict)  # Debug
        load_encodings()  # Tải lại encoding ngay sau khi huấn luyện
        return "Training complete!"
    except Exception as e:
        return f"Error while saving encodings: {str(e)}"

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

        # Chuyển đổi khung hình sang RGB để sử dụng với face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Tính khoảng cách đến tất cả encoding đã lưu
            distances = face_recognition.face_distance(list(encoding_dict.values()), face_encoding)
            tolerance = 0.55  # Giảm tolerance để tăng độ chính xác (chỉ nhận diện khi rất giống)

            # Tìm khoảng cách nhỏ nhất và kiểm tra xem có nằm trong ngưỡng không
            min_distance = float('inf')
            best_match_index = -1
            for i, distance in enumerate(distances):
                if distance < min_distance and distance <= tolerance:
                    min_distance = distance
                    best_match_index = i

            if best_match_index != -1:
                student_key = list(encoding_dict.keys())[best_match_index]
                class_folder, student_folder = student_key.split("/", 1)
                student_id, student_name = student_folder.split("_", 1)
                print(f"Recognized: {student_id}_{student_name} in class {class_name}, Distance: {min_distance:.2f}")  # Debug
                
                if class_folder != class_name or not is_student_in_class(student_id, student_name, class_name):
                    text = "Warning: Student not in this class!"
                    color = (0, 0, 255)
                elif check_attendance_today(student_id, class_name):
                    text = f"{student_name} - Attended"
                    color = (0, 255, 255)
                else:
                    text = f"{student_name} (Distance: {min_distance:.2f})"
                    color = (0, 255, 0)
                    # Ghi nhận điểm danh
                    conn = sqlite3.connect("attendance.db")
                    c = conn.cursor()
                    c.execute("INSERT INTO attendance (student_id, student_name, class_name, timestamp) VALUES (?, ?, ?, ?)",
                              (student_id, student_name, class_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    conn.close()
                    print(f"Attendance recorded for {student_name} in class {class_name}")  # Debug
            else:
                text = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
    app.run(host="0.0.0.0", port=5000, debug=False)