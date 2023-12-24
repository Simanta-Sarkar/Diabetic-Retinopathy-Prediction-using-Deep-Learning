import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import PIL
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from jupyterthemes import jtplot
from IPython.display import display
import tkinter as tk
from tkinter import filedialog

# Function to open file dialog and get the path of the selected file
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path

# Function to preprocess an image
def preprocess_image(image_path):
    img = PIL.Image.open(image_path)
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to resize and preprocess an image
def resize_and_preprocess_image(image_path, target_size=(256, 256)):
    img = PIL.Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions on an image
def predict_class(model, image_path, labels):
    img_array = resize_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class

# Create a simple GUI with a button to browse for an image
root = tk.Tk()
root.withdraw()  # Hide the main window

# Browse for an image
uploaded_image_path = browse_image()

if uploaded_image_path:
    # Load the trained model
    model = load_model("retina_weights.hdf5")

    # Define labels mapping
    labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}

    # Get prediction for the uploaded image
    prediction = predict_class(model, uploaded_image_path, labels)
    print(f"Predicted Class: {prediction}")

    # Display the image and prediction (you can customize this part)
    img = PIL.Image.open(uploaded_image_path)
    plt.imshow(img)
    plt.title(f"Predicted Class: {prediction}")
    plt.show()

root.mainloop()
