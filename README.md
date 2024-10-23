# COVID-19 Image Classification

This project implements a convolutional neural network (CNN) to classify chest X-ray images as either COVID-19 positive or negative. It uses TensorFlow and Keras for model training and prediction.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Results](#results)

## Introduction

The aim of this project is to develop an automated system for detecting COVID-19 from chest X-ray images. By using deep learning techniques, the model can aid in the rapid diagnosis of the disease.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/covid19-image-classification.git
   cd covid19-image-classification
   ```

2. Install the required packages:
   ```bash
   pip install tensorflow numpy pandas matplotlib
   ```

3. Ensure you have the dataset organized in the following structure:
   ```
   DataSet/
       ├── train/
       │   ├── COVID19/
       │   └── NORMAL/
       └── test/
           ├── COVID19/
           └── NORMAL/
   ```

## Dataset

The dataset consists of chest X-ray images categorized into two folders:
- **COVID19**: Contains images of patients diagnosed with COVID-19.
- **NORMAL**: Contains images of healthy patients.

Ensure you have a balanced dataset for optimal model performance.

## Model Architecture

The CNN model architecture is as follows:

1. **Convolutional Layer**: 32 filters, kernel size of (5, 5), ReLU activation.
2. **Max Pooling Layer**: Pool size of (2, 2).
3. **Dropout Layer**: Dropout rate of 0.5.
4. **Convolutional Layer**: 64 filters, kernel size of (5, 5), ReLU activation.
5. **Max Pooling Layer**: Pool size of (2, 2).
6. **Flattening Layer**: Flattens the input.
7. **Dense Layer**: 256 units, ReLU activation.
8. **Dropout Layer**: Dropout rate of 0.5.
9. **Output Layer**: 1 unit with sigmoid activation for binary classification.

## Usage

1. Train the model by running the following command in your terminal:
   ```bash
   python train_model.py
   ```

2. To predict whether a new chest X-ray image is COVID-19 positive or negative, run the following script:
   ```bash
   python predict.py
   ```

   This will open a file dialog to select an image. The output will indicate if the report is COVID-19 positive or negative based on the model’s prediction.

## Results

The model is evaluated on a test dataset, and the accuracy and loss metrics are displayed. The prediction threshold is set at 0.6890, where values greater than this indicate a negative result.
