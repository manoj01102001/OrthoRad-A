import streamlit as st
from pathlib import Path
from Utils.constants import AppConstants

def home_page():
    """Main landing page with navigation"""
    st.markdown(
        """
        <style>
          .stApp {
            background-color: #000000;
          }
          .custom-button {
            background-color: #3299A2 !important;
            color: white !important;
            border: 1px solid #3299A2 !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )
    print("Manoj annan")

    _, col2, _ = st.columns([0.25, 0.5, 0.25])

    with col2:
        _, col1, _ = st.columns(3)
        with col1:
            if Path(AppConstants.LOGO_LOCATION).is_file():
                st.image(AppConstants.LOGO_LOCATION, width=50000)
            else:
                st.warning("Logo not found!")

        # Title and descriptions
        st.markdown("""<h2 style='color: #3299A2; text-align: center;'>Welcome to Orthorad-A</h2>""", 
                   unsafe_allow_html=True)
        st.markdown("""<h4 style='color: #3299A2; text-align: center;'>
                       AI-Powered CVMI Classification Made Simple</h4>""", 
                   unsafe_allow_html=True)
        st.markdown("""<p style='color: white; text-align: center;'>
                       Upload your lateral cephalogram and let Orthorad-A do the rest. 
                       Our advanced algorithm accurately classifies Cervical Vertebral 
                       Maturation Index (CVMI), assisting you in orthodontic growth 
                       assessment with precision and ease.</p>""", 
                   unsafe_allow_html=True)
        
        st.markdown("----")
        
        # Analysis Types Section
        st.markdown("<h3 style='color: #3299A2; text-align: center;'>Analysis Types</h3>", 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4 style='color: #3299A2;'>App Objective</h4>", 
                       unsafe_allow_html=True)
            st.markdown("""
                <div style='color: white; text-align: justify;'>
                This app uses Vision Transformer (ViT) models to classify masked cervical 
                vertebrae images (C2, C3, C4) into predefined CVMI stages:
                </div>
                """, unsafe_allow_html=True)
            
            # Stages display
            for stage_number in range(2, 7):
                st.markdown(
                    f"<p style='color: #3299A2; text-align: center; margin: 0.5rem 0;'>"
                    f"Stage {stage_number}</p>",
                    unsafe_allow_html=True
                )

        with col2:
            st.markdown("<h4 style='color: #3299A2;'>Key Features</h4>", 
                       unsafe_allow_html=True)
            st.markdown("""
                <ul style='color: white;'>
                  <li>Anatomical structure segmentation</li>
                  <li>Quantitative measurements</li>
                  <li>Automated stage classification</li>
                  <li>Clinical report generation</li>
                </ul>
                """, unsafe_allow_html=True)

        # Login button
        _, center_col, _ = st.columns(3)
        with center_col:
            st.markdown("""
                <style>
                    .stButton>button {
                        background-color: #3299A2 !important;
                        color: white !important;
                        border: 2px solid white !important;
                        border-radius: 5px !important;
                        padding: 0.5rem 2rem !important;
                    }
                </style>
                """, unsafe_allow_html=True)
            
            if st.button("🔐 Login", key="login_btn"):
                st.session_state.page = "login_in"
                st.rerun()