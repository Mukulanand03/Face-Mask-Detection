# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:48:07 2023

@author: mukul
"""

import streamlit as st
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
#from cv2 import cv2_imshow

model = load_model('facemask.h5')

#input_image_path = input('Path of the image to be predicted: ')
def Mask_prediction(input_image):
    
   # input_image = cv2.imread(input_image_path)
    
    #cv2_imshow(input_image)
    
    input_image_resized = cv2.resize(input_image, (128,128))
    
    input_image_scaled = input_image_resized/255
    
    input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
    
    input_prediction = model.predict(input_image_reshaped)
    
    print(input_prediction)
    
    
    input_pred_label = np.argmax(input_prediction)
    
    print(input_pred_label)
    
    
    if input_pred_label == 1:
    
      return 'The person in the image is wearing a mask'
    
    else:
    
      return 'The person in the image is not wearing a mask'
  
def main():
    
    
    # giving a title
    st.title('Face Mask Prediction Web App')
    
    
    # getting the input data from the user
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Button to trigger prediction
    if st.button("Predict"):
        prediction_result = Mask_prediction(image)
        st.write(prediction_result)



if __name__ == '__main__':
    main()