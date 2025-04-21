# segmentation.py
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from Utils.constants import AppConstants
from Utils.footer import add_footer_with_logo


@st.cache_resource
def load_segmentation_model():
    """Load and cache the segmentation model"""
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            AppConstants.SEGMENTATION_MODEL_PATH
        ).to(AppConstants.DEVICE)
        
        processor = SegformerFeatureExtractor.from_pretrained(
            AppConstants.SEGMENTATION_MODEL_PATH
        )
        return model, processor
    except Exception as e:
        st.error(f"Error loading segmentation model: {str(e)}")
        return None, None

def segment_image(image, model, processor, threshold):
    """Perform segmentation on the input image"""
    try:
        inputs = processor(images=image, return_tensors="pt").to(AppConstants.DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode='bilinear',
            align_corners=False
        )
        
        probabilities = torch.softmax(upsampled_logits, dim=1)
        prob_values, pred_mask = torch.max(probabilities, dim=1)
        
        prob_map = prob_values.squeeze().cpu().numpy()
        final_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
        final_mask[prob_map <= threshold] = 0
        
        return final_mask, prob_map
    except Exception as e:
        st.error(f"Segmentation failed: {str(e)}")
        return None, None

def segmentation_page():
    """Main segmentation analysis page"""

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

    _, col2,_ = st.columns([0.25, 0.5,0.25])

    with col2:

        st.markdown("<h1 style='text-align: center;'> Segmentation of cervical spine vertebrae </h1>", unsafe_allow_html=True)
        
        if 'uploaded_image' not in st.session_state or not st.session_state.uploaded_image:
            st.warning("Please upload an image first!")
            st.session_state.page = "upload"
            st.rerun()
            return
        
        model, processor = load_segmentation_model()
        if model is None:
            return
        _, col2,_ = st.columns([0.2, 0.6,0.2])

        with col2:
            threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0,
                value=AppConstants.SEGMENTATION_THRESHOLD,
                step=0.01,
                help="Adjust the minimum confidence level for segmentation"
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(
                st.session_state.uploaded_image,
                caption="Original Image",
                use_container_width=True
            )
        
        try:
            mask, prob_map = segment_image(
                st.session_state.uploaded_image,
                model,
                processor,
                threshold
            )
            
            if mask is not None and prob_map is not None:
                with col2:
                    fig, ax = plt.subplots()
                    ax.imshow(mask, cmap=AppConstants.SEGMENTATION_COLORMAP)
                    ax.axis('off')
                    st.pyplot(fig)
                    st.caption("Segmentation Mask")
                
                with col3:
                    fig, ax = plt.subplots()
                    im = ax.imshow(prob_map, cmap=AppConstants.PROBABILITY_COLORMAP)
                    plt.colorbar(im, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
                    st.caption("Probability Map")
                
                st.markdown("---")
                st.subheader("Quantitative Analysis")
                
                col1, col2, col3 = st.columns(3)
                max_prob = np.max(prob_map)
                avg_prob = np.mean(prob_map[mask > 0]) if np.any(mask) else 0
                
                col2.metric("Maximum Confidence", f"{max_prob:.2f}")
                col3.metric("Average Confidence", f"{avg_prob:.2f}")
        
        except Exception as e:
            st.error(f"Error during segmentation: {str(e)}")
        
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
            if st.button("Try CVMI classification"):
                st.session_state.page = "classification"
                st.session_state.analysis_type = "classification"
                st.rerun()
    add_footer_with_logo("images/company_logo.jfif")