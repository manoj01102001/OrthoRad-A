# classification.py
import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from Utils.constants import AppConstants
from Utils.footer import add_footer_with_logo
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

    st.markdown(
        """
        <style>
          .stApp {
            background-color: #000000;
          }s
  /* Base slider track */
  .stSlider input[type="range"] {
    -webkit-appearance: none;
       -moz-appearance: none;
            appearance: none;
    width: 100%;
    height: 6px;
    background: #1a1a1a;      /* dark unfilled track */
    border-radius: 3px;
    margin: 1rem 0;
  }

  /* Chrome/Safari unfilled track */
  .stSlider input[type="range"]::-webkit-slider-runnable-track {
    background: #1a1a1a;
    height: 6px;
    border-radius: 3px;
  }
  /* Chrome/Safari filled portion */
  .stSlider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    background: #3299A2;      /* aqua thumb */
    border: 2px solid #ffffff;/* white border */
    border-radius: 50%;
    margin-top: -5px;          /* center on track */
    cursor: pointer;
  }

  /* Firefox track & filled portion */
  .stSlider input[type="range"]::-moz-range-track {
    background: #1a1a1a;
    height: 6px;
    border-radius: 3px;
  }
  .stSlider input[type="range"]::-moz-range-progress {
    background: #3299A2;
    height: 6px;
    border-radius: 3px;
  }
  .stSlider input[type="range"]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    background: #3299A2;
    border: 2px solid #ffffff;
    border-radius: 50%;
    cursor: pointer;
  }

  /* IE/Edge */
  .stSlider input[type="range"]::-ms-track {
    background: transparent;
    border-color: transparent;
    color: transparent;
    height: 6px;
  }
  .stSlider input[type="range"]::-ms-fill-lower {
    background: #3299A2;
    border-radius: 3px;
  }
  .stSlider input[type="range"]::-ms-fill-upper {
    background: #1a1a1a;
    border-radius: 3px;
  }
  .stSlider input[type="range"]::-ms-thumb {
    width: 16px;
    height: 16px;
    background: #3299A2;
    border: 2px solid #ffffff;
    border-radius: 50%;
    cursor: pointer;
    margin-top: 0;
  }

  /* Slider label text */
  .stSlider label {
    color: white !important;
  }



        /* Slider track */
        .stSlider input[type="range"] {
            color: #3299A2 !important;
            accent-color: #3299A2; 
            width: 100%;
            height: 6px;
            background: #3299A2;              /* Dark track background */
            border-radius: 3px;
            outline: none;
            margin: 1rem 0;
            
        }

        /* Slider label text */
        .stSlider label {
            color: white !important;
        }


          .stTextInput input, .stNumberInput input, .stSelectbox select {
            color: white !important;
            background-color: #1a1a1a !important;
            border: 1px solid #3299A2 !important;
            border-radius: 4px !important;
            padding: 8px !important;
          }

          .stSelectbox [role="listbox"] {
            background-color: #1a1a1a !important;
            color: white !important;
          }

          .stSelectbox [role="option"] {
            color: white !important;
            background-color: #1a1a1a !important;
          }

          /* Radio group container */
            .stRadio [role="radiogroup"] {
                background-color: #3299A2 !important;
                border: 1px solid #3299A2 !important;
                padding: 10px 15px !important;
                border-radius: 6px !important;
            }

            /* Force every bit of text inside the radio group to white */
            .stRadio [role="radiogroup"] label,
            .stRadio [role="radiogroup"] label * {
                color: white !important;
            }

          /* Fix button layout */
            .stButton>button {
            color: white !important;
            background-color: #3299A2 !important;
            white-space: nowrap !important;
            text-align: center !important;
            width: 100% !important;
            }

            /* You can adjust width to a fixed value if needed */
            .stButton>button {
            min-width: 160px !important;
            padding: 0.75rem 1.5rem !important;
            }
          .stButton>button:hover {
            background-color: #267880 !important;
            transform: scale(1.05);
          }

          h1, h2, h3, h4 {
            color: #3299A2 !important;
          }

          label {
            color: white !important;
          }

          .upload-preview {
            color: grey !important;
            font-style: italic;
            text-align: center;
            margin-top: 5px;
          }
        </style>
        """,
        unsafe_allow_html=True
    )


    _, col2,_ = st.columns([0.2, 0.6,0.2])

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

    add_footer_with_logo("images/company_logo.jfif")


