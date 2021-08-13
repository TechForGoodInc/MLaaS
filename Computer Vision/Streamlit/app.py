import streamlit as st
from multiapp import MultiApp
from apps import home, canny, cartoonApp, imageannotation,imageResize, WebcamFaceRecognition #, FaceDetection, eyes , imageclassification # import your app modules here
app = MultiApp()





st.markdown("""
# Tech For Good Inc
Making the world a beter place - one line of code at a time


""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Canny", canny.app)
app.add_app("Cartoon", cartoonApp.app)
app.add_app("Image Annotation", imageannotation.app)
app.add_app("Image Resize", imageResize.app)
app.add_app("Webcam Face Recognition", WebcamFaceRecognition.app)
#app.add_app("Eyes Detection", eyes.app)
#app.add_app("Face Detection", FaceDetection.app)
#app.add_app("Image Classification", imageclassification.app)

# The main app
app.run()


# Created by : Siti Khotijah, 07/29/2021
