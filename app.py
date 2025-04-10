# app.py
import streamlit as st
from frontend.home_page import home_page
from frontend.image_upload import upload_page
from frontend.post_image_upload import post_image_upload_page
from frontend.image_classification import classification_page
from frontend.image_segmentation import segmentation_page

def main():
    st.set_page_config(
        page_title="OrthoRad-A",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    # Page routing
    pages = {
        "home": home_page,
        "upload": upload_page,
        "post_image_upload": post_image_upload_page,
        "classification": classification_page,
        "segmentation": segmentation_page
    }
    
    # Show current page
    pages[st.session_state.page]()

if __name__ == "__main__":
    main()