import streamlit as st
from PIL import Image
from Utils.constants import AppConstants
from Utils.footer import add_footer_with_logo

def post_image_upload_page():
    st.markdown(
        """
        <style>
          .stApp {
            background-color: #000000;
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

    _, col_middle, _ = st.columns([0.25, 0.5, 0.25])
    with col_middle:
        col1, col2 = st.columns([0.6, 0.4])
            
        with col1:
            st.image(
                st.session_state.uploaded_image,
                caption="Uploaded Image Preview",
                use_container_width=True,
                output_format="JPEG"
            )

        with col2:
            st.subheader("Analysis Options")
            st.markdown(
                "<p style='color: white;'>Select the type of analysis you want to perform:</p>", 
                unsafe_allow_html=True
            )

            analysis_type = st.radio(
                "Analysis Type",
                ["CVMI Classification", "Segmentation"],
                horizontal=False,
                key="analysis_type_radio"
            )

        # Centered buttons
        st.markdown("<br>", unsafe_allow_html=True)
        _,col1, _,  col2,_ = st.columns(5)

        with col1:
            if st.button("Start Analysis", key="start_analysis"):
                st.session_state.analysis_type = "classification" if analysis_type == "CVMI Classification" else "segmentation"
                st.session_state.page = st.session_state.analysis_type
                st.rerun()

        with col2:
            if st.button("Upload New Image", key="new_upload"):
                st.session_state.page = "upload"
                st.session_state.uploaded_image = None
                st.session_state.current_file = None
                st.session_state.analysis_type = None
                st.rerun()
    add_footer_with_logo("images/company_logo.jfif")