# Pkgs
import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os


# Cascades
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

# Detection Features


def detect_faces(our_image):
    new_img = np.array(our_image.convert("RGB"))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect Facce
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw Rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, faces


def detect_eyes(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return img


def detect_smiles(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect Smiles
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the Smiles
    for (x, y, w, h) in smiles:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img


def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Edges
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # Color
    color = cv2.bilateralFilter(img, 9, 300, 300)
    # Cartoon
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon


def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny


def main():
    """Face Detection App"""

    st.title("Face Dectection App")
    st.write("**Build with Streamit and OpenCV**")

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activities", activities)

    if choice == "Detection":
        st.subheader("Face Detection")

        image_file = st.file_uploader(
            "Upload Image", type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image, width=400)

        enhance_type = st.sidebar.radio(
            "Enhance Type", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])

        if enhance_type == "Gray-Scale":
            new_img = np.array(our_image.convert("RGB"))
            gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            st.image(gray, width=400)

        if enhance_type == "Contrast":
            c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output, width=400)

        if enhance_type == "Brightness":
            b_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(b_rate)
            st.image(img_output, width=400)

        if enhance_type == "Blurring":
            new_img = np.array(our_image.convert("RGB"))
            blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
            blur = cv2.GaussianBlur(new_img, (11, 11), blur_rate)
            st.image(blur, width=400)

        # else:
        #     st.image(our_image, width=300)

        # Face Detection
        task = ["Faces", "Smiles", "Eyes", "Cannize", "Cartonize"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):

            if feature_choice == "Faces":
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img, width=400)
                st.success(f"Found {len(result_faces)} faces")

            elif feature_choice == 'Smiles':
                result_img = detect_smiles(our_image)
                st.image(result_img, width=400)

            elif feature_choice == 'Eyes':
                result_img = detect_eyes(our_image)
                st.image(result_img, width=400)

            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(our_image)
                st.image(result_img, width=400)

            elif feature_choice == 'Cannize':
                result_canny = cannize_image(our_image)
                st.image(result_canny, width=400)

    elif choice == "About":
        st.subheader("About")
        st.markdown("This is a simple face detection app made with OpenCV and Streamlit and deployed on Heroku. This is just a basic app it's not perfect yet.")
        st.text("Autor: Sattwik Raj (SR)")
        st.write(
                '''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos. 
		The algorithm has four stages:
			1. Haar Feature Selection 
			2. Creating  Integral Images
			3. Adaboost Training
			4. Cascading Classifiers
Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')
        st.write(
            "Source Code: https://github.com/Sattwikb/Streamlit-face_detection.git")
        st.text("Instagram: @sattwik_raj_7")
        st.text("Team Instgram: @kni8angle")
        st.text("Twitter: @SattwikRaj")


if __name__ == "__main__":
    main()
