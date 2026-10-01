# Soybean Seed Classification with Explainable AI

A deep learning pipeline for classifying soybean seed quality (broken, immature, 
intact, skin-damaged, spotted), with LIME-based explainability 
to make predictions interpretable — intended to give farmers actionable, visual feedback 
rather than a black-box label.

## Overview
- **Task:** 5-class image classification of soybean seed condition
- **Models:** ResNet50 and ResNet18 (transfer learning, frozen backbone + custom head)
- **Explainability:** LIME (Local Interpretable Model-agnostic Explanations) to visualize 
  which regions of a seed image drove the prediction

## Results

| Model     | Epochs | Test Accuracy |
|-----------|--------|----------------|
| ResNet50  | 10     | 81.40%         |
| ResNet18  | 20     | 76.21%         |

ResNet18 trades some accuracy for a lighter model and better regularization 
(augmentation + dropout), useful in lower-resource deployment settings.

## Explainability Example
LIME highlights the seed regions most influential to the predicted class, 
making misclassifications and model reasoning inspectable rather than opaque.

## Tech Stack
PyTorch · torchvision · scikit-learn · LIME · matplotlib
