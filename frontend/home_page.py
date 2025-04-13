# home.py
import streamlit as st
import base64
from pathlib import Path
from Utils.constants import AppConstants

def home_page():
    """Main landing page with navigation"""

    _, col2, _ = st.columns([0.25, 0.5, 0.25])

    with col2:
        # Load and display logo

        _,col1 ,_= st.columns(3)

        with col1:

            if Path(AppConstants.LOGO_LOCATION).is_file():
                st.image(AppConstants.LOGO_LOCATION, width=150)  # adjust width as needed
            else:
                st.warning("Logo not found!")

        

        # Subtitle
        st.markdown("""<h2 style='text-align: center;'>This is a demo App for the dev team and QA team to test</h2>""", unsafe_allow_html=True)

        # Description
        st.markdown("""<h4 style='text-align: center;'>Get started by uploading your medical image for analysis:</h4>""", unsafe_allow_html=True)


        st.markdown("----")
        # Analysis Types
        st.markdown("<h2 style='text-align: center;'>Analysis Types</h2>", unsafe_allow_html=True)
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

        _, center_col, _ = st.columns(3)

        with center_col:
            if st.button("📤 Login in", type="primary"):
                st.session_state.page = "login_in"
                st.rerun()
