import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
model=tf.keras.models.load_model('/content/save.h5')
st.markdown("<h1 style='text-align: center;'>Malaria detection by CNN model</h1>", unsafe_allow_html=True)
st.subheader('Input will be the cell snapshots of suspected person')
st.set_option('deprecation.showfileUploaderEncoding', False)
img=st.file_uploader('Drop or upload cell images here',types=['jpeg','png','jpg'])
st.markdown("<br><br>",unsafe_allow_html=True)
if (st.button('SUBMIT')) & (img is not None):
  img=Image.open(img)
  st.markdown("<br>",unsafe_allow_html=True)
  st.image(img,caption='Uploaded image')
  image = tf.keras.preprocessing.image.img_to_array(img)
  img=np.resize(image,(1,90,90,3))
  #image = tf.keras.preprocessing.image.load_img(image, target_size=(90,90,3))
  #image = tf.keras.preprocessing.image.img_to_array(image)
  if model.predict(img)==0:
    st.markdown('POSITIVE')
    st.header('Model implying that the image contains malaria')
  else:
    st.balloons()
    st.markdown('Negative')
    st.header('Model implying that the image does not contain malaria')

    
