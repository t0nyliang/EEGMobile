# EEGMobile

Accepted HCII 2024: PENDING

Abstract

Electroencephalography (EEG) analysis is an important domain in the realm of Brain-Computer Interface (BCI) research. To ensure BCI devices are capable of providing practical applications in the real world, brain signal processing techniques must be fast, accurate, and resource-conscious to deliver low-latency neural analytics. This study presents a model that leverages a pre-trained MobileViT alongside Knowledge Distillation (KD) for EEG regression tasks. Our results showcase that this model performs at a level comparable to the previous State-of-the-Art (SOTA) on the EEGEyeNet Absolute Position Task, achieving a Root Mean Squared Error (RMSE) of 53.6, a 3\% reduction in accuracy, while being 33\% faster and 60\% smaller. Our research presents a cost-effective model applicable to resource-constrained devices and contributes to expanding future research on lightweight, mobile-friendly models for EEG regression.

# Overview
EEGMobile incorporates a pre-trained MobileViT network first presented by Mehta & Rastegari in: ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
"](https://arxiv.org/abs/2110.02178) and further expanded in ["Separable Self-attention for Mobile Vision Transformers"](https://arxiv.org/abs/2206.02680). Furthermore, this model utilized Knowledge Distillation in the training procedure, based on the work of Hinton et al. in ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531). 

This repository includes the original EEGViT models which can be found [here](https://github.com/ruiqiRichard/EEGViT), the EEGViT-TCNet (teacher model) which can be found [here](https://github.com/ModeEric/EEGViT-TCNet), and our EEGMobile model. Weights for pre-trained models were loaded from [huggingface.co](https://huggingface.co/).

# Dataset
Data for the EEGEyeNet Absolute Position Task can be downloaded with
```
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
More information on this dataset and others can be found in: ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100)

# Requirements
Basic requirements can be installed with
```
pip install -r general_requirements.txt
```

# Basic Usage
Default training of the teacher model (EEGViT-TCNet) or others can be done with
```
python run.py
```

Once the teacher model's weights have been saved, they can be loaded to train EEGMobile with
```
python distillation_run.py
```

You can load and run a speed test on any saved model using
```
python inference_test.py
```

Be sure you have selected the right model when loading saved weights.
