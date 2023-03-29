# Tomato Disease Diagnosis Computer Vision Model
## Introduction
This project aims to diagnose tomato diseases using computer vision techniques. The model was developed using PyTorch and experiment management was done with WandB.

## Dataset
The dataset used for this project is [Mendely Data For Plant diseases](https://data.mendeley.com/datasets/tywbtsjrjv/1). It contains about [18,000 number] images of tomato plants with different diseases.

## Model Architecture
The model architecture used for this project is resnet18. It was finetuned on 7 epochs and achieved a validation accuracy of about 99%.

## Usage
To use the model, follow these steps:

1. Clone the repository
1. Install the required libraries listed in requirements.txt
1. Download the dataset and place it in the data directory
1. Run python eval.py --image_path [path to image] to predict the disease in a single image.

## Results
The model gives predictions in form of accuracy

## Conclusion
In conclusion, this project demonstrates the potential of computer vision techniques in diagnosing tomato diseases. The model achieved high accuracy and can be used to diagnose tomato diseases in real-world scenarios.
 
## Note:
Only sessions01, 02 and 03 were used for the project. Other repos init and training were used while testing some issues.



