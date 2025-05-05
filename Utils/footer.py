import streamlit as st
import base64
import os

def get_base64_image(image_path):
    """
    Reads an image file and returns its Base64 representation.
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def add_footer_with_logo(logo_filename=None):
    """
    Adds a footer with a Base64 encoded logo to the Streamlit application.
    """
    if logo_filename is None:
        
        return # Don't add footer if logo couldn't be encoded
    logo_base64_string = get_base64_image(logo_filename)
    
    footer_html = f"""
    <style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1; /* Light gray background */
        color: #303030; /* Dark gray text */
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #e1e1e1; /* Light gray border at the top */
        display: flex; /* Use flexbox to align items */
        justify-content: center; /* Center items horizontally */
        align-items: center; /* Center items vertically */
    }}
    .footer img {{
        margin-right: 10px; /* Space between logo and text */
        vertical-align: middle; /* Align image nicely with text */
    }}
    .footer a {{
        color: #007bff; /* Blue link color */
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    </style>
    <div class="footer">
        <img src="data:image/png;base64,{logo_base64_string}" alt="Company Logo" height="30"> <p style="margin: 0;">Developed for Jairus | &copy; 2025 JS Tech</p>
        <p style="margin: 0 0 0 20px;"><a href="http://jairus.in/" target="_blank">Visit Our Website</a> | <a href="tel:+91-9176337062">Contact Us</a></p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
