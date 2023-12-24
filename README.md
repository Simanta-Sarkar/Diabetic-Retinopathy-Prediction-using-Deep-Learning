# Diabetic-Retinopathy-Prediction-using-Deep-Learning
Introduction:

Diabetic Retinopathy (DR) is a diabetic eye disease that affects the retina, leading to vision impairment and, if left untreated, blindness. Early detection and classification of DR severity are crucial for timely medical intervention. This project focuses on the development of a Deep Learning-based system for the automated classification of fundus images to predict the severity of Diabetic Retinopathy.


Technology Stack:

Deep Learning Framework: TensorFlow and Keras
Web Framework: Streamlit (for the web application version)
Data Processing: PIL (Python Imaging Library) for image processing
Model Architecture: Utilizes a pre-trained model (ResNet50)

Project Components:

1. Model Training:
A deep neural network model is trained on a labeled dataset of fundus images.
The model is designed to classify images into different severity levels: Mild, Moderate, No DR, Proliferate_DR, and Severe.
2. Web Application (Streamlit):
The Streamlit web application allows users to upload fundus images.
Uploaded images are processed by the trained model to predict the severity of Diabetic Retinopathy.
The application displays the uploaded image, the predicted severity level, and the associated probability.
3. Image Preprocessing:
Input images undergo preprocessing to ensure consistency and compatibility with the model.
Image resizing, array conversion, and preprocessing steps are applied to prepare the image for prediction.

How to Use:

Access the Web Application:


Run the Streamlit app using the provided script (streamlit_app.py).
Open the provided local URL in a web browser.

Upload Fundus Images:


Use the file uploader in the web application to select and upload fundus images in JPG, JPEG, or PNG format.

View Predictions:


The application will display the uploaded image along with the predicted severity level and probability.

Conclusion:

This project contributes to the development of an accessible and user-friendly tool for Diabetic Retinopathy prediction. The integration of deep learning techniques with a web application allows for quick and efficient analysis of fundus images, aiding in early diagnosis and intervention to prevent vision loss in diabetic patients.
