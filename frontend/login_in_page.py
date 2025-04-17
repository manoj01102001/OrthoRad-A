import streamlit as st
import base64
from pathlib import Path
import streamlit.components.v1 as components

def popup_warning(msg: str):
    # injects a tiny <script> that fires off a JS alert()
    components.html(
        f"""
        <script>
          alert("{msg}");
        </script>
        """,
        height=0,  # no extra vertical space
    )

def load_image_as_base64(image_path: str) -> str:
    p = Path(image_path)
    if not p.is_file():
        st.error(f"Image file not found: {image_path}")
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")

def login_in_page():
    st.markdown(
        """
        <style>
          .stApp {
            background-color: #000000;
          }
          /* Input fields */
          .stTextInput input, .stNumberInput input, .stSelectbox select {
            color: white !important;
            background-color: #3299A2 !important;
            border: 1px solid #3299A2 !important;
            border-radius: 4px !important;
            padding: 8px !important;
          }
          
          /* Dropdown menu */
          .stSelectbox [role="listbox"] {
            background-color: #3299A2 !important;
            color: #3299A2 !important;
          }
          
          /* Dropdown options */
          .stSelectbox [role="option"] {
            color: #3299A2 !important;
            background-color: #3299A2 !important;
          }
          
          /* Focus states */
          .stTextInput input:focus, 
          .stNumberInput input:focus,
          .stSelectbox select:focus {
            box-shadow: 0 0 0 2px #3299A2 !important;
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
          
          /* Form submit button */
          div[data-testid="stFormSubmitButton"] > button {
            width: 100% !important;
          }
          
          /* Section headers */
          h1, h2, h3, h4 {
            color: #3299A2 !important;
          }
          
          /* Labels */
          label {
            color: white !important;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    _, col2, _ = st.columns([0.2, 0.6, 0.2])
    with col2:
        st.markdown("<h1 style='text-align: center;'>📝 Patient Information</h1>", unsafe_allow_html=True)
        
        # Personal Info Form
        
        name = st.text_input("Full Name:")
        
        age = st.number_input("Age:", min_value=0, max_value=120, step=1)
        gender = st.selectbox(
            "Gender:",
            ("Male", "Female", "Non-Binary", "Prefer not to say")
        )
        height = st.number_input("Height (cm):", min_value=50, max_value=250, step=1)
        weight = st.number_input("Weight (kg):", min_value=10, max_value=300, step=1)
        
        st.markdown("<h3 style='text-align: center;'>Body Type Selection</h3>", unsafe_allow_html=True)
        
        # Body type selection
        body_types_files = {
            "Mesomorphic": "images/mesomorph.png",
            "Endomorphic": "images/endomorph_37328.webp",
            "Ectomorphic": "images/ectomorph.jfif"
        }
        body_types = {b: load_image_as_base64(path) for b, path in body_types_files.items()}

        cols = st.columns(3)
        for idx, (btype, img_data) in enumerate(body_types.items()):
            with cols[idx]:
                border_color = "#3299A2" if st.session_state.get('selected_body_type') == btype else "#444"
                html = f"""
                <div style="text-align: center; margin: 10px 0;">
                    <img src="data:image/png;base64,{img_data}" 
                            style="width:100%; 
                                border-radius:10px; 
                                border: 3px solid {border_color};
                                cursor: pointer;
                                transition: all 0.3s ease;">
                </div>
                """
                # Render the HTML
                st.markdown(html, unsafe_allow_html=True)
                if st.button(btype, key=f"btn_{btype}"):
                    st.session_state.selected_body_type = btype
                    st.rerun()

        st.markdown("---")
        col1,_ ,col2 = st.columns(3)

        with col1:
            if st.button("← Return to Home"):
                st.session_state.page = "home"
                st.rerun()

        with col2:
            if st.button("Upload cephalogram Image→"):     
                if (name=="") or (age==0) or (st.session_state.selected_body_type == None):
                    popup_warning("Please enter all the required fields")
                else:
                    st.session_state.page = "upload"
                    st.rerun()
        