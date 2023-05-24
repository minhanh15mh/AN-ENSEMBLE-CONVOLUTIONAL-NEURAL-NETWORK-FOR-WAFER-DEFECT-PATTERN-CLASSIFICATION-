import streamlit as st 
import pandas as pd
import numpy as np
from PIL import Image
from model_for_web import *

st.title('Wafer defects classification')
st.write('Please upload image')
st.markdown("""[Example Wafer Map image (download files here)](https://drive.google.com/drive/folders/1jBtw3ena1MsmZfMD7iEkX9pi0Ac4MXoA?usp=drive_link)""")
input_image = st.file_uploader('Input Wafer Map image here')

if input_image is not None:
   open_image = np.array(Image.open(input_image))
   st.image(open_image)
else:
   pass

clasify_button = st.button('Classify')

if clasify_button:
   model = load_model()
   output = predict_WM(model,open_image)
   st.write('The type of defect: ', output)
