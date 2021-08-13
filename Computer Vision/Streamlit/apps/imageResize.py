import streamlit as st
from PIL import Image
import numpy as np
import cv2

def app():
    st.title('Image Resizing with OpenCV')

    img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.image(image, caption=f"Original Image",use_column_width= True)

    useWH = st.checkbox('Resize using a Custom Height and Width')

    useScaling = st.checkbox('Resize using a Scaling Factor')

    if useWH:
        st.subheader('Input a new Width and Height')

        width = int(st.number_input('Input a new a Width',value = 720))
        height = int(st.number_input('Input a new a Height',value = 720))

        points = (width, height)
    
        resized_image = process_image(image , points)

        st.image(
        resized_image, caption=f"Resized image", use_column_width=False)
    

    if useScaling:
        st.subheader('Drag the Slider to change the Image Size')
    
        scaling_factor = st.slider('Reszie the image using scaling factor',min_value = 0.1 , max_value = 5.0 ,
                               value = 1.0, step = 0.5)
    
        resized1_image = process_scaled_image(image,scaling_factor)
    
        st.image(
        resized1_image, caption=f"Resized image using Scaling factor", use_column_width=False)
    






DEMO_IMAGE = ''

@st.cache
def process_image(image,points):
    
    resized_img = cv2.resize(image , points , interpolation = cv2.INTER_LINEAR)
    
    return resized_img

@st.cache
def process_scaled_image(image,scaling_factor):
    
    resized_img = cv2.resize(image , None,fx= scaling_factor, fy= scaling_factor, interpolation = cv2.INTER_LINEAR)
    
    return resized_img






    
    
    
    
    
    
    
    
    
    





