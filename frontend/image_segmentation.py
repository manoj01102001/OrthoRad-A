# segmentation.py
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from Utils.constants import AppConstants

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
    st.title("Medical Image Segmentation")
    
    if 'uploaded_image' not in st.session_state or not st.session_state.uploaded_image:
        st.warning("Please upload an image first!")
        st.session_state.page = "upload"
        st.rerun()
        return
    
    model, processor = load_segmentation_model()
    if model is None:
        return
    
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
        if st.button("Try CVMI classification for Image"):
            st.session_state.page = "classification"
            st.session_state.analysis_type = "classification"
            st.rerun()