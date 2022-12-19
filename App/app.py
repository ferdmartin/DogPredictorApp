#!streamlit/bin/python
import streamlit as st
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import json
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

'''
path = "C:/Users/madhu/AppData/Local/Programs/Python/Python38/NeuralNetScratch/"
dataset = os.path.join(path,"cats-and-dogs/")
print(dataset)
'''

from zipfile import ZipFile

from GDownload import download_file_from_google_drive

@st.cache(allow_output_mutation=True)
def load_model():
  model_location = '1jPO4OLmiEn3rFpF8wUv6Q7jkJzk9wK7n'
  save_dest = Path('saved_model')
  save_dest.mkdir(exist_ok=True)
  saved_model = Path("saved_model/FerNetEfficientNetB2.zip")
  if not saved_model.exists():
    download_file_from_google_drive(model_location, saved_model)
  
  with ZipFile(saved_model, 'r') as zObject:
    zObject.extractall(path="saved_model/FerNetEfficientNetB2")
    
  os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

  '''
  path = "saved_model"
  dataset = os.path.join(path,"FerNetEfficientNetB2")
  print(dataset)
  '''
  model_to_load = tf.keras.models.load_model("")
  return model_to_load

@st.cache
def load_classes():
    with open(str(Path().parent.absolute())+'/App/classes_dict.json') as classes:
        class_names = json.load(classes)
    return class_names

def load_and_prep_image(filename, img_shape=260):
  #img = tf.io.read_file(filename)
  img = np.array(filename)#tf.io.decode_image(filename, channels=3)
  # Resize our image
  img = tf.image.resize(img, [img_shape,img_shape])
  # Scale
  return img # don't need to resclae images for EfficientNet models in Tensorflow

if __name__ == '__main__':

  saved_model = load_model()
  class_names = load_classes()

  st.header("Dog Breeds Predictor")
  st.write("Choose any dog image and get the corresponding breed:")

  uploaded_image = st.file_uploader("Choose an image...")
    
  if uploaded_image:
    uploaded_image = Image.open(uploaded_image)
    image_for_the_model = load_and_prep_image(uploaded_image)
    prediction = saved_model.predict(tf.expand_dims(image_for_the_model, axis=0), verbose=0)
    print(tf.argmax(prediction, axis=1).numpy())
    predicted_breed = class_names[str(tf.argmax(prediction, axis=1).numpy()[0])]
    predicted_breed = ' '.join(predicted_breed.split('_'))
    predicted_breed = predicted_breed.title()
    st.title(f'This dog is a {predicted_breed}')
    st.image(uploaded_image)
