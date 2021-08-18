import cv2
import streamlit as st

def app():
    st.title("Webcam Eye Detection")
    camera_index = 0

    eye_cascade = cv2.CascadeClassifier(r'C:\Users\komp18\Streamlit\data\haarcascade_eye.xml')
    cap = cv2.VideoCapture(camera_index)
    FRAME_WINDOW = st.image([])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(":(")
            break
        boxes = eye_cascade.detectMultiScale(frame)
        for box in boxes:
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        FRAME_WINDOW.image(frame)
    