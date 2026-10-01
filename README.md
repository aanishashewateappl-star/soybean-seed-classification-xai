# Soybean Seed Classification with Explainable AI

Transfer-learning classifiers (ResNet50 and ResNet18) for soybean seed condition, with LIME explanations that show which parts of a seed image influenced each prediction.

## Overview
- **Task:** 5-class image classification: broken, immature, intact, skin-damaged, spotted
- **Models:** ImageNet-pretrained ResNet50 and ResNet18 with a frozen backbone and a new classification layer
- **Explainability:** LIME (Local Interpretable Model-agnostic Explanations) highlights the image regions that drove a prediction, so errors and model behavior can be inspected rather than treated as a black box
- **Motivation:** seed quality grading is a visual task, and visual explanations make a model's decisions easier to check

## Results
| Model | Test Accuracy |
|---|---|---|
| ResNet50 | 81.40% |
| ResNet18 | 76.21% |

LIME segments the image into superpixels and shows which ones increased or decreased the score for the predicted class.

