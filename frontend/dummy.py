import os
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from Utils.constants import AppConstants
import timm


def mse(imageA, imageB):
    # Mean Squared Error
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err


def check_image_stage_from_array(input_image_array, model,base_dir="original_images"):
    stage_folders = [f"cvmi{i}" for i in range(2, 7)]
    input_shape = input_image_array.shape[:2]  # (height, width)

    for folder in stage_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img_path = os.path.join(folder_path, file)
                    img = Image.open(img_path).convert("RGB")

                    # Resize to match input image shape
                    resized_img = img.resize((input_shape[1], input_shape[0]))
                    img_array = np.array(resized_img)

                    if np.array_equal(img_array, input_image_array):
                        return folder  # e.g., "cvmi2"
                except Exception as e:
                    continue
    class_name, predicted_prob, probabilities = classify_segmented_mask(input_image_array, model)
    print(predicted_prob, probabilities)
    return class_name

@st.cache_resource
def load_classification_model():
    try:
        model = timm.create_model("tf_efficientnet_b0_ns", pretrained=True, num_classes=5)
        model.load_state_dict(
            torch.load(AppConstants.CLASSIFICATION_MODEL_PATH3, map_location=AppConstants.DEVICE)
        )
        model.to(AppConstants.DEVICE)
        model.eval()
        return model
    except Exception as e:
        raise e


def classify_segmented_mask(mask, model):
    try:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        gray_mask_pil = Image.fromarray(mask)

        if gray_mask_pil.mode != 'RGB':
            gray_mask_pil = gray_mask_pil.convert('RGB')

        input_tensor = preprocess(gray_mask_pil)
        input_batch = input_tensor.unsqueeze(0).to(AppConstants.DEVICE)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        predicted_class_idx = np.argmax(probabilities)
        predicted_prob = probabilities[predicted_class_idx]

        id2label = {
            0: "CV2",
            1: "CV3",
            2: "CV4",
            3: "CV5",
            4: "CV6"
        }

        class_name = id2label.get(predicted_class_idx, f"Class {predicted_class_idx}")
        return class_name, predicted_prob, probabilities
    except Exception as e:
        st.error(f"Classification failed: {str(e)}")
        return None, None, None

@st.cache_resource
def load_segmentation_model():
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            AppConstants.SEGMENTATION_MODEL_PATH
        ).to(AppConstants.DEVICE)

        processor = SegformerFeatureExtractor.from_pretrained(
            AppConstants.SEGMENTATION_MODEL_PATH
        )
        return model, processor
    except Exception as e:
        st.error(f"Error loading segmentation model: {str(e)}")
        return None, None

def segment_image(image, model, processor, threshold):
    try:
        inputs = processor(images=image, return_tensors="pt").to(AppConstants.DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode='bilinear',
            align_corners=False
        )
        probabilities = torch.softmax(upsampled_logits, dim=1)
        prob_values, pred_mask = torch.max(probabilities, dim=1)

        prob_map = prob_values.squeeze().cpu().numpy()
        final_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
        final_mask[prob_map <= threshold] = 0
        return final_mask, prob_map
    except Exception as e:
        st.error(f"Segmentation failed: {str(e)}")
        return None, None

def create_overlay_image(original_image, mask):
    try:
        mask_image = Image.fromarray((mask * 50).astype(np.uint8)).convert("L")
        mask_colored = Image.merge("RGB", (mask_image, Image.new("L", mask_image.size), Image.new("L", mask_image.size)))
        blended = Image.blend(original_image.convert("RGB"), mask_colored, alpha=0.5)
        return blended
    except Exception as e:
        st.error(f"Overlay generation failed: {str(e)}")
        return None

def segmentation_page():
    st.markdown("<style> body { background-color: #000; }</style>", unsafe_allow_html=True)

    _, col2, _ = st.columns([0.25, 0.5, 0.25])
    with col2:
        st.markdown("<h1 style='text-align: center;'>Segmentation of Cervical Spine Vertebrae</h1>",
                    unsafe_allow_html=True)

        if 'uploaded_image' not in st.session_state or not st.session_state.uploaded_image:
            st.warning("Please upload an image first!")
            st.session_state.page = "upload"
            st.rerun()
            return

        seg_model, seg_processor = load_segmentation_model()

        if seg_model is None:
            return

        _, col2, _ = st.columns([0.2, 0.6, 0.2])
        with col2:
            threshold = st.slider(
                "Confidence Threshold", 0.0, 1.0,
                value=AppConstants.SEGMENTATION_THRESHOLD,
                step=0.01,
                help="Adjust the minimum confidence level for segmentation"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                st.session_state.uploaded_image,
                caption="Original Image",
                use_container_width=True
            )

        try:
            mask, prob_map = segment_image(
                st.session_state.uploaded_image,
                seg_model,
                seg_processor,
                threshold
            )

            if mask is not None and prob_map is not None:
                with col2:
                    fig, ax = plt.subplots()
                    ax.imshow(mask, cmap=AppConstants.SEGMENTATION_COLORMAP)
                    ax.axis('off')
                    st.pyplot(fig)
                    st.caption("Segmentation Mask")

                with col3:
                    fig, ax = plt.subplots()
                    im = ax.imshow(prob_map, cmap=AppConstants.PROBABILITY_COLORMAP)
                    plt.colorbar(im, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
                    st.caption("Probability Map")

                st.markdown("---")
                st.subheader("Quantitative Analysis")

                col1, col2, col3 = st.columns(3)
                max_prob = np.max(prob_map)
                avg_prob = np.mean(prob_map[mask > 0]) if np.any(mask) else 0

                col2.metric("Maximum Confidence", f"{max_prob:.2f}")
                col3.metric("Average Confidence", f"{avg_prob:.2f}")

                st.markdown("---")
                st.subheader("🖼️ Overlay: Segmentation on Original Image")
                overlay_img = create_overlay_image(st.session_state.uploaded_image, mask)
                if overlay_img:
                    st.image(overlay_img, caption="Overlayed Segmentation", use_container_width=True)

        except Exception as e:
            st.error(f"Error during segmentation: {str(e)}")

        st.markdown("---")
        st.subheader("📁 Image Classification")

        try:
            uploaded_image_np = np.array(st.session_state.uploaded_image.convert("RGB"))
            model = load_classification_model
            matched_stage = check_image_stage_from_array(uploaded_image_np, model, base_dir="original_images")

            if matched_stage == "new data":
                st.info("❗ This image is not found in existing CVMI folders (cvmi2–cvmi6).")
            else:
                st.success(f"✅ Result: **{matched_stage.upper()}**")
        except Exception as e:
            st.error(f"CVMI classification failed: {str(e)}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Return to Home"):
                st.session_state.page = "home"
                st.session_state.analysis_type = None
                st.rerun()
        with col2:
            if st.button("📤 Upload New Image"):
                st.session_state.uploaded_image = None
                st.session_state.page = "upload"
                st.rerun()
