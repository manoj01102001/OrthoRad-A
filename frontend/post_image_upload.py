import streamlit as st
from PIL import Image
from Utils.constants import AppConstants

def post_image_upload_page():
    col1, col2 = st.columns([0.5, 0.5])
        
    with col1:
        st.image(
            st.session_state.uploaded_image,
            caption="Uploaded Image Preview",
            use_container_width=True,
            output_format="JPEG"
        )
        
    with col2:
        st.subheader("Analysis Options")
        st.markdown("""
        **Select the type of analysis you want to perform:**
        """)
        
        analysis_type = st.radio(
            "Analysis Type",
            ["CVMI Classification", "Segmentation"],
            horizontal=True
        )
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Analysis", type="primary"):
                if analysis_type == "CVMI Classification":
                    st.session_state.analysis_type = "classification"
                    st.session_state.page = "classification"
                else:
                    st.session_state.analysis_type = "segmentation"
                    st.session_state.page = "segmentation"
                st.rerun()
        with col2:
            if st.button("Upload New Image"):
                st.session_state.uploaded_image = None
                st.session_state.current_file = None
                st.session_state.analysis_type = None
                st.rerun()