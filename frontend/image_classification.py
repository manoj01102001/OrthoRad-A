# classification.py
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from Utils.constants import AppConstants

@st.cache_resource
def load_classification_model():
    """Load and cache the classification model"""
    try:
        print(f"Trying to open {AppConstants.CLASSIFICATION_MODEL_PATH} for classification")
        with open(AppConstants.CLASSIFICATION_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {str(e)}")
        return None

def preprocess_classification_image(image):
    """Preprocess image for classification model"""
    try:
        img = image.resize((64, 64)).convert('RGB')
        img_array = np.array(img)[:, :, ::-1]  # Convert RGB to BGR
        return img_array.flatten().reshape(1, -1)
    except Exception as e:
        st.error(f"Image preprocessing failed: {str(e)}")
        return None

def show_classification_results(prediction, probabilities):
    """Display classification results with visualization"""
    st.subheader(f"Prediction: {AppConstants.CLASS_NAMES[prediction[0]]}")
    
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#000033')
    ax.set_facecolor('#000033')
    
    bars = ax.bar(
        AppConstants.CLASS_NAMES,
        probabilities,
        color='skyblue'
    )
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            color='white'
        )
    
    ax.tick_params(axis='x', labelrotation=45, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_ylabel('Probability', color='white')
    ax.set_title('Class Probabilities', color='white')
    st.pyplot(fig)

def classification_page():
    """Main classification analysis page"""

    _, col2,_ = st.columns([0.2, 0.6,0.26])

    with col2:
        st.markdown("<h1 style='text-align: center;'> CVMI Classification </h1>", unsafe_allow_html=True)
        
        if 'uploaded_image' not in st.session_state or not st.session_state.uploaded_image:
            st.warning("Please upload an image first!")
            st.session_state.page = "upload"
            st.rerun()
            return
        
        model = load_classification_model()
        if model is None:
            return
        
        col1, col2 = st.columns([0.6 , 0.4])
        
        with col1:
            st.image(
                st.session_state.uploaded_image,
                caption="Uploaded Image",
                use_container_width=True
            )
        
        with col2:
            with st.spinner("Analyzing image..."):
                try:
                    processed_img = preprocess_classification_image(st.session_state.uploaded_image)
                    if processed_img is not None:
                        prediction = model.predict(processed_img)
                        probabilities = model.predict_proba(processed_img)[0]
                        show_classification_results(prediction, probabilities)
                except Exception as e:
                    st.error(f"Classification failed: {str(e)}")
        
        st.markdown("---")
        col1, col2, col3= st.columns(3)
        with col1:
            if st.button("🏠 Return to Home"):
                st.session_state.page = "home"
                st.session_state.analysis_type = None
                st.rerun()
        with col2:
            if st.button("📤 Upload New Image"):
                st.session_state.uploaded_image = None
                st.session_state.page = "upload"
                st.rerun()

        with col3:
            if st.button("Try Segmentation for Image"):
                st.session_state.page = "segmentation"
                st.session_state.analysis_type = "segmentation"
                st.rerun()


