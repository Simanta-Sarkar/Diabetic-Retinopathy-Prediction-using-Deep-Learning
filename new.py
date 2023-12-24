import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
from PIL import Image
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

# Define labels
labels = {i: label for i, label in enumerate(["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"])}

# Function to open file dialog and get the path of the selected file
def browse_image():
    root = tk.Tk()
    root.withdraw()
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            raise FileNotFoundError
    finally:
        root.destroy()
    return file_path

# Function to preprocess an image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions on an image
def predict_class(model, image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class, prediction

# Create a simple GUI with a button to browse for an image
root = tk.Tk()
root.geometry("300x100")
root.title("DR Image Prediction")

browse_button = tk.Button(root, text="Browse Image", command=lambda: update_prediction())
browse_button.pack(pady=10)

image_label = tk.Label(root, text="No image selected", font=("Arial", 12))
image_label.pack(pady=5)

prediction_label = tk.Label(root, text="", font=("Arial", 12))
prediction_label.pack(pady=5)

def update_prediction():
    try:
        uploaded_image_path = browse_image()
        if uploaded_image_path:
            # Load the trained model
            model = load_model("retina_weights.hdf5")

            # Get prediction and confidence scores
            predicted_class, prediction_scores = predict_class(model, uploaded_image_path)

            # Update GUI labels
            image_label.config(text=f"Image: {os.path.basename(uploaded_image_path)}")
            prediction_label.config(text=f"Predicted Class: {predicted_class}\nConfidence scores: {prediction_scores}")

            # Display the image
            img = Image.open(uploaded_image_path)
            plt.imshow(img)
            plt.title(f"Predicted Class: {predicted_class}")
            plt.show()
    except FileNotFoundError:
        print("Error: No file selected")

root.mainloop()
