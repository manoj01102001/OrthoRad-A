# upload_page.py
import streamlit as st
from PIL import Image
from Utils.constants import AppConstants

def upload_page():
    """Dedicated page for image upload and preprocessing"""
    st.markdown(
        """
        <style>
          .stApp {
            background-color: #000000;
          }
          .stButton>button {
            background-color: #3299A2 !important;
            color: white !important;
            border: 2px solid white !important;
            border-radius: 5px !important;
            padding: 0.5rem 2rem !important;
            transition: all 0.3s ease;
          }
          .stButton>button:hover {
            background-color: #267880 !important;
            transform: scale(1.05);
          }
          .stSuccess {
            color: #3299A2 !important;
            border-color: #3299A2 !important;
          }
          .stError {
            border-color: #3299A2 !important;
          }
          .upload-title {
            color: #3299A2 !important;
            text-align: center;
            margin-bottom: 2rem;
          }
          .upload-instructions {
            color: white !important;
            text-align: center;
            margin-bottom: 2rem;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    _, col2, _ = st.columns([0.25, 0.5, 0.25])
    with col2:
        # Page title
        st.markdown("<h1 class='upload-title'>Image Upload Center</h1>", unsafe_allow_html=True)
        
        # Upload instructions
        st.markdown("""
            <div class='upload-instructions'>
                <p>Upload lateral cephalogram images for CVMI analysis</p>
                <p>Supported formats: JPEG, PNG, DICOM</p>
            </div>
        """, unsafe_allow_html=True)

        # Initialize session state variables
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        
        # File upload section
        uploaded_file = st.file_uploader(
            " ",
            type=["jpg", "jpeg", "png", "dcm"],
            help="Select medical image for analysis",
            key="main_uploader"
        )
        
        # Handle file upload
        if uploaded_file and uploaded_file != st.session_state.get('current_file'):
            try:
                st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.session_state.current_file = uploaded_file
                st.success("Image uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.session_state.uploaded_image = None
        
        # Process uploaded image
        if st.session_state.uploaded_image:
            if "analysis_type" not in st.session_state or not st.session_state.analysis_type:
                st.session_state.analysis_type = None
                st.session_state.page = "post_image_upload"
                st.rerun()
            else:
                st.session_state.page = st.session_state.analysis_type
                st.rerun()
  
        # Navigation footer
        st.markdown("---")
        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
        with col2:
            if st.button("← Return to Home", key="return_home"):
                st.session_state.page = "home"
                st.rerun()