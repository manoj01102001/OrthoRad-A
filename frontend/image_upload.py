# upload_page.py
import streamlit as st
from PIL import Image
from Utils.constants import AppConstants

def upload_page():
    """Dedicated page for image upload and preprocessing"""
    _ , col2, _ = st.columns([0.25,0.5, 0.25])
    with col2:

        st.title("Image Upload Center")
        
        # Initialize session state variables
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Choose a medical image file",
            type=["jpg", "jpeg", "png", "dcm"],
            help="Supported formats: JPEG, PNG, DICOM",
            key="main_uploader"
        )
        
        # Handle new file upload
        if uploaded_file and uploaded_file != st.session_state.get('current_file'):
            try:
                st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.current_file = uploaded_file
                st.success("Image uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.session_state.uploaded_image = None
        
        # Display preview and analysis options
        if st.session_state.uploaded_image:

            if "analysis_type" not in  st.session_state or not st.session_state.analysis_type:
                st.session_state.analysis_type = None
                st.session_state.page = "post_image_upload"
                st.rerun()
            else:
                st.session_state.page = st.session_state.analysis_type
                st.rerun()
  
        # Navigation footer
        st.markdown("---")
        if st.button("← Return to Home"):
            st.session_state.page = "home"
            st.rerun()