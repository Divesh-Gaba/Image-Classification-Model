This project implements a Convolutional Neural Network (CNN) for multi-class image classification using the CIFAR-10 dataset and TensorFlow.

It demonstrates a complete deep learning workflow — from data loading and preprocessing to model training and evaluation — marking my transition from Data Analysis to Deep Learning and Artificial Neural Networks (ANNs).

**Project Objective**

To build and train a CNN capable of classifying images into 10 different object categories using hierarchical feature learning.

**Dataset**

CIFAR-10 consists of:

60,000 color images

Image size: 32 × 32 × 3

10 object classes

50,000 training images

10,000 test images

**Classes include**:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

**Tech Stack**

Python

TensorFlow / Keras

TensorFlow Datasets

Matplotlib (visualization)

**Model Architecture**

The CNN consists of:

🔹 Convolution Block 1

Conv2D (32 filters, 3×3, ReLU)

MaxPooling (2×2)

🔹 Convolution Block 2

Conv2D (64 filters, 3×3, ReLU)

MaxPooling (2×2)

🔹 Convolution Block 3

Conv2D (128 filters, 3×3, ReLU)

🔹 Fully Connected Layers

Flatten

Dense (128 neurons, ReLU)

Dense (10 neurons, Softmax)

Filter progression: 32 → 64 → 128

This allows hierarchical feature extraction:
Edges → Shapes → Object Parts → Object Classes

**Data Pipeline**

Efficient tf.data pipeline implementation:

Normalization (pixel scaling 0–255 → 0–1)

Mapping using map()

Shuffling

Batching (64)

Prefetching (AUTOTUNE)

This ensures optimal GPU utilization and scalable training.

**Training Configuration
**
Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Metrics: Accuracy

Epochs: 10

Validation: Test dataset

**Model Performance**

Achieved strong multi-class classification performance on unseen test data.

(Replace with your actual accuracy if needed.)

**Key Learnings**

Understanding convolution operations and feature maps

Why filter depth increases across layers

Importance of MaxPooling and translation invariance

Efficient data pipelines using tf.data

Backpropagation in CNNs

Transitioning from statistical modeling to representation learning

**Future Improvements**

Add Batch Normalization

Add Dropout for regularization

Implement Data Augmentation

Hyperparameter tuning

Deploy model as REST API (FastAPI)

Convert model to TensorFlow Lite

**Motivation**

This project represents my transition from Data Analysis to Deep Learning — moving from analyzing patterns in data to building systems that automatically learn patterns using ANN architectures.
