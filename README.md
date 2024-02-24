# Chest X-ray Pneumonia Detection using Python and Keras

## Introduction

The goal of this project is to develop a model for detecting pneumonia from chest X-ray images using Python and Keras. The model leverages deep learning techniques to analyze medical images and make predictions based on learned patterns.

## Folder Structure
- **data/**: Contains image files downloaded from Kaggle.
- **models/**: Stores pre-trained models.
- **plots/**: Holds plots generated during exploratory data analysis (EDA).
- **src/**: Source code directory.
  - **api.py**: Module for reading and processing data.
  - **model.py**: Responsible for image data preprocessing, model setup, and training.
  - **plot.py**: Contains routines for generating plots for exploratory data analysis.
- **config**: Configuration and constants file.
- **main.py**: Main routine to execute the code.

## Getting Started

### Virtual Environment
Create a virtual environment:

```console
foo@bar:~$ python -m venv .venv 
```

Activate the environment:

```console
foo@bar:~$ source .venv/Scripts/activate 
```

Install required packages:

```console
foo@bar:~$ pip install -r requirements.txt
```

Ensure you have the following Python packages installed:

- pandas
- tensorflow
- scikit-learn
- seaborn
- matplotlib
- opencv-python
- numpy
- plotly

### Folder Structure

Make sure the project folder has the following structure:

```plaintext
xChest/
|-- data/
|-- models/
|-- plots/
|-- src/
|   |-- api.py
|   |-- model.py
|   |-- plot.py
|-- config
|-- main.py
```

### Running the Model

To run the model, execute the following command in the terminal:

```console
foo@bar:~$ python xChest/main.py
```



## Model Summary

The model is an image classification model built using TensorFlow/Keras. It consists of a pre-trained EfficientNetB3 base model followed by additional layers for fine-tuning and classification.

### Architecture:
- Base Model: EfficientNetB3 (pre-trained on ImageNet)
- Additional Layers:
    - Batch Normalization
    - Dense (fully connected) layer with ReLU activation
    - Dropout layer for regularization
    - Output layer with softmax activation for multi-class classification

### Training Configuration:
- Optimizer: Adamax
- Learning Rate: Defined by the HyperPars class
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy

The model is compiled and ready for training with the specified hyperparameters. Overall, the model architecture and training configuration are designed to leverage a pre-trained base model, incorporate additional layers for fine-tuning, and optimize the model's parameters using Adamax optimization with Categorical Crossentropy loss and accuracy as evaluation metrics. This setup aims to create an effective and efficient image classification model capable of accurately predicting the classes of input images.


### Layer Shape 
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ efficientnetb3 (Functional)          │ (None, 1536)                │      10,783,535 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 1536)                │           6,144 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 256)                 │         393,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 2)                   │             514 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 33,370,247 (127.30 MB)
 Trainable params: 11,093,290 (42.32 MB)
 Non-trainable params: 90,375 (353.03 KB)