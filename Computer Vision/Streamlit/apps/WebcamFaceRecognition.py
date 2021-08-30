import cv2
import streamlit as st

def app():
    st.title("Webcam Face Recognition")


    camera_index = 0

    vc = cv2.VideoCapture(camera_index)
    detector = cv2.CascadeClassifier(r"C:\Users\skhot\Streamlit\data\haarcascade_frontalface_default.xml")

    FRAME_WINDOW = st.image([])

    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print(":(")
            break
        boxes = detector.detectMultiScale(frame)
        for box in boxes:
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        FRAME_WINDOW.image(frame)
