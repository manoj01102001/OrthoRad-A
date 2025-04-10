import streamlit as st
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt

@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    try:
        img = image.resize((64, 64)).convert('RGB')
        img_array = np.array(img)[:, :, ::-1]
        img_flattened = img_array.flatten().reshape(1, -1)
        return img_flattened
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def home():
    st.title("CVMI Classification App")
    st.write("Welcome to the CVMI Classification App. This application allows you to classify images using a trained machine learning model.")
    st.write("Navigate through the sidebar to upload an image and view the classification results.")

    if st.button("Go to Upload Image"):
        st.session_state.page = "Upload Image"
        st.rerun()

def input_image():
    st.title("Upload Image")
    st.write("Upload an image for classification using our trained ML model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
        st.success("Image uploaded successfully!")

        if st.button("Go to Model Classification"):
            st.session_state.page = "Model Classification"
            st.rerun()

def model_classification():
    st.title("Model Classification")

    if 'uploaded_image' not in st.session_state:
        st.warning("Please upload an image first.")
        return

    model_path = 'cvmi_classification_2025_04_05.pkl'
    model = load_model(model_path)

    if model is None:
        return

    class_names = ['CVMI 4', 'CVMI 6', 'CVMI 5', 'CVMI 3', 'CVMI 2']

    try:
        image = Image.open(st.session_state.uploaded_image)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        processed_image = preprocess_image(image)

        if processed_image is not None:
            prediction = model.predict(processed_image)

            st.subheader("Prediction")
            st.write(f"Predicted class: {class_names[prediction[0]]}")

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_image)
                st.write("Class probabilities:")

                # Plotting the probabilities with dark background
                fig, ax = plt.subplots()
                fig.patch.set_facecolor('#000033')  # Dark blue background
                ax.set_facecolor('#000033')         # Dark blue axes background

                bars = ax.bar(class_names, probabilities[0], color='skyblue')

                # Customize text colors
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom',
                            color='white')

                ax.set_ylabel('Probability')
                ax.set_title('Class Probabilities')
                st.pyplot(fig)
    except Exception as e:
        st.error(f"Error processing file: {e}")

    if st.button("Go to Home"):
        st.session_state.page = "Home"
        st.rerun()

    if st.button("Go to Upload Image"):
        st.session_state.page = "Upload Image"
        st.rerun()

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Upload Image":
        input_image()
    elif st.session_state.page == "Model Classification":
        model_classification()

if __name__ == '__main__':
    main()
