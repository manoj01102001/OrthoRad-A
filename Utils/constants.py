# constants.py
import os
import torch
from typing import Dict, Any

class AppConstants:
    """Central configuration for medical imaging application"""
    
    # Model paths
    CLASSIFICATION_MODEL_PATH = "model/classification/cvmi_classification_2025_04_05.pkl"
    SEGMENTATION_MODEL_PATH = "model/segmentation/segformer-new/"
    CLASSIFICATION_MODEL_PATH2 = "model/efficientnet_classification/efficientnet_model.pth"
    CLASSIFICATION_MODEL_PATH3 = "model/efficientnet_classification/saved_model_eff.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEGMENTATION_THRESHOLD = 0.5
    SEGMENTATION_COLORMAP = "viridis"
    PROBABILITY_COLORMAP = "hot"

    # Hardware configuration
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    
    # Clinical parameters
    CLASS_NAMES = ['CVMI 4', 'CVMI 6', 'CVMI 5', 'CVMI 3', 'CVMI 2']
    SEGMENTATION_THRESHOLD = 0.65
    CLASSIFICATION_THRESHOLD = 0.45
    
    # UI constants
    DEFAULT_IMAGE_SIZE = (1024, 768)
    SEGMENTATION_COLORMAP = "viridis"
    PROBABILITY_COLORMAP = "plasma"
    PLOT_DPI = 300

    LOGO_LOCATION = "logo/openai-logo.png"

    @staticmethod
    def get_segmentation_path() -> str:
        """Get segmentation path with environment variable fallback"""
        return os.getenv("SEGMENTATION_MODEL_PATH", AppConstants.SEGMENTATION_MODEL_PATH)

    @staticmethod
    def device_info() -> Dict[str, Any]:
        """Get formatted device information"""
        return {
            "device": AppConstants.DEVICE,
            "cuda_available": AppConstants.USE_CUDA,
            "device_count": torch.cuda.device_count() if AppConstants.USE_CUDA else 0
        }

    @staticmethod
    def class_mapping() -> Dict[int, str]:
        """Generate class index to name mapping"""
        return {i: name for i, name in enumerate(AppConstants.CLASS_NAMES)}

    @staticmethod
    def show_config() -> Dict[str, Any]:
        """Display current configuration"""
        return {
            "model_paths": {
                "classification": AppConstants.CLASSIFICATION_MODEL_PATH,
                "segmentation": AppConstants.get_segmentation_path()
            },
            "device_config": AppConstants.device_info(),
            "clinical_settings": {
                "class_names": AppConstants.CLASS_NAMES,
                "thresholds": {
                    "segmentation": AppConstants.SEGMENTATION_THRESHOLD,
                    "classification": AppConstants.CLASSIFICATION_THRESHOLD
                }
            },
            "ui_settings": {
                "default_image_size": AppConstants.DEFAULT_IMAGE_SIZE,
                "colormaps": {
                    "segmentation": AppConstants.SEGMENTATION_COLORMAP,
                    "probability": AppConstants.PROBABILITY_COLORMAP
                }
            }
        }

