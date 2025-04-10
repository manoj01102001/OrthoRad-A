# home.py
import streamlit as st

def home_page():
    """Main landing page with navigation"""
    st.title("Medical Imaging Suite")
    
    st.markdown("""
    ## This is demo App for dev team and QA Team to test
    
    **Get started by uploading your medical image for analysis:**
    """)
    
    if st.button("📤 Upload Image", type="primary"):
        st.session_state.page = "upload"
        st.rerun()

    st.subheader("Analysis Types")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🩺 CVMI Classification")
        st.markdown("""
        - Cervical Vertebral Maturation Index
        - Automated staging
        - Probability visualization
        """)
    
    with col2:
        st.markdown("### 🎯 Segmentation of C2, C3 and C4")
        st.markdown("""
        - Anatomical structure segmentation
        - Quantitative measurements
        """)
    
    st.markdown("---")
    st.info("""
    **Supported formats:** JPEG, PNG, DICOM  
    **Maximum file size:** 20MB  
    **Resolution recommendations:** 512x512 to 2048x2048 pixels
    """)