import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Setting up the page title and header
st.title('Object Detector')
st.header(
    'Upload an image and the model will predict the class of the main object.')

# Endpoint URL of your FastAPI app
url = 'http://127.0.0.1:8000/detect/'

print(url)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...",
                                 type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Preparing the image for sending to FastAPI
    img_bytes = uploaded_file.getvalue()
    files = {'file': ('image.jpg', img_bytes, 'multipart/form-data')}

    # POST request to the FastAPI server
    response = requests.post(url, files=files)

    # Display the response
    if response.status_code == 200:
        result = response.json()
        for detection, probability in result['detections'].items():
            st.write(f"**{detection.capitalize()}**: {probability:.2%}")
        # st.write('Detected class:', result)
    else:
        st.error('Failed to process image.')
