# GAIA-MIL
This repository is an official implementation of using Gated Attention Instance Agregation in Multiple Instance Learning Model to detect Gallbladder Cancer (GBC) from Ultrasound Imaging.

Rationale beind levraging MIL paradigm is to mimmic the clical workflow of processing a "bag" of ultrasound images from single patient to produce a diagnosis.

## Pipeline
1. **Quality Control**: `quality.py` filters out non-diagnostic images based on channel variance and pixel intensity standard deviation.
2. **ROI Extraction**: `crop.py` identifies and crops the ultrasound cone using largest contour detection, removing UI elements and text.
3. **Model Training**: `efficientNet.py` implements a learnable attention mechanism to assign importance weights to every image in a patient's bag along with EfficientNet-B4 backbone for instance-level feature extraction. It also has built-in generation of Grad-CAM heatmaps and attention score visualization to explain model decisions.

### Training Stratergy
- Training loop initially freezes the backbone, then systematically unfreezes layers from the top down to prevent catastrophic forgetting.
- Handling dynamic bag sizes with sampling strategies for training (random subsampling) and validation.

### Usage
- **Preprocessing**: Run `quality.py` to filter data, followed by `crop.py` to standardize inputs.
- **Training**: Run `efficient_net.py` to start the 5-fold cross-validation with progressive unfreezing.


## Dataset
This framework was developed using the AURORA-GB (Advanced Ultrasound Repository for Oncological Research and Analysis - Gall Baldder)dataset, the largest multi-center public repository for gallbladder ultrasound, comprising 11,012 images from 1,151 patients.