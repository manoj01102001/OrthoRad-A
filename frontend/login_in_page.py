import streamlit as st
import base64
from pathlib import Path

def load_image_as_base64(image_path: str) -> str:
    p = Path(image_path)
    if not p.is_file():
        st.error(f"Image file not found: {image_path}")
        return ""
    return base64.b64encode(p.read_bytes()).decode("utf-8")

def login_in_page():

    _, col2,_ = st.columns(3)

    with col2:
        st.title("📝 Personal Info Form")
        name = st.text_input("Enter your name:")
        age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
        gender = st.selectbox(
            "Select your gender:",
            ("Male", "Female", "Non-Binary", "I rather not say")
        )
        height = st.number_input("Enter your height (in cm):", min_value=50, max_value=250, step=1)
        weight = st.number_input("Enter your weight (in kg):", min_value=10, max_value=300, step=1)

        st.markdown("### Select your body type:")

        body_types_files = {
            "Mesomorphic": "images/mesomorph.png",
            "Endomorphic": "images/endomorph_37328.webp",
            "Ectomorphic": "images/ectomorph.jfif"
        }
        body_types = {b: load_image_as_base64(path) for b, path in body_types_files.items()}

        # We only need one variable
        if "selected_body_type" not in st.session_state:
            st.session_state.selected_body_type = None

        # Three columns for the three body types
        cols = st.columns(3)
        for idx, (btype, img_data) in enumerate(body_types.items()):
            with cols[idx]:
                with st.form(key=btype):
                    # Color border if selected
                    border_color = "#4CAF50" if st.session_state.selected_body_type == btype else "#ddd"
                    # HTML
                    button_html = f"""
                    <button type="submit" style="border:none; background:none; padding:0;">
                        <img src="data:image/png;base64,{img_data}" alt="{btype}"
                            style="width:100%; border-radius:10px; border:3px solid {border_color};" />
                    </button>
                    <div style="text-align:center; font-weight:bold;">{btype}</div>
                    """
                    st.markdown(button_html, unsafe_allow_html=True)

                    submitted = st.form_submit_button(f"Select {btype}")
                    if submitted:
                        st.session_state.selected_body_type = btype
                        st.rerun()

        # Show which body type is selected
        if st.session_state.selected_body_type:
            st.success(f"Selected Body Type: {st.session_state.selected_body_type}")

        # Final Submit
        _ , col_submit, _ = st.columns(3)
        with col_submit:
            if st.button("Submit"):
                if not st.session_state.selected_body_type:
                    st.warning("Please select a body type before submitting.")
                else:
                    st.success("Form submitted successfully! ✅")
                    st.markdown(f"""
                    - **Name:** {name}
                    - **Age:** {age}
                    - **Gender:** {gender}
                    - **Height:** {height} cm
                    - **Weight:** {weight} kg
                    - **Body Type:** {st.session_state.selected_body_type}
                    """)
                    st.session_state.page = "upload"
                    st.rerun()
