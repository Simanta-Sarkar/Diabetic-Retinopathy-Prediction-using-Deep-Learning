import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess and predict the uploaded image
def predict_uploaded_image(uploaded_image):
    # Load the trained model
    model = load_model("retina_weights.hdf5")

    # Define labels mapping
    labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3: 'Proliferate_DR', 4: 'Severe'}

    # Preprocess and predict
    img_array = resize_and_preprocess_image(uploaded_image)
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]
    

    return predicted_class

# Function to resize and preprocess an image
def resize_and_preprocess_image(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit web app
def main():
    st.title("Diabetic Retinopathy using Deep Learning")
    st.subheader("By:\n1.Jishantu Kripal Bordoloi\n\n2.Simanta Sarkar\n\n3.Amlan Jyoti Dutta\n\n4.Hriseekesh Kalita")
    st.subheader("Under the guidance of: Dr. Sanyukta Chetia\nAssistant Professor & HOD\n\nDepartment of Electronics and Telecommunication Engineering")

    st.write("\n")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Make predictions and display result
        if st.button("Predict"):
            prediction = predict_uploaded_image(uploaded_image)
            st.success(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
