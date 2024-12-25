# Spam Detection Model

This repository contains the implementation of a **Spam Detection Model** designed to classify messages as spam or non-spam using machine learning techniques. The system focuses on high accuracy, scalability, and practical usability.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Techniques](#modeling-techniques)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Spam Detection Model provides:

1. An effective way to filter out unwanted spam messages.
2. A machine learning pipeline for data preprocessing, feature extraction, model training, and evaluation.
3. Scalable architecture for integration into real-world applications.

## Features

- Preprocessing pipeline for text data, including tokenization, stopword removal, and stemming/lemmatization.
- Feature extraction using methods such as:
  - Bag of Words (BoW).
  - Term Frequency-Inverse Document Frequency (TF-IDF).
- Multiple machine learning models implemented for comparison.
- Easy integration into messaging platforms and email systems.

## Dataset

The dataset includes labeled messages as spam or non-spam, commonly sourced from publicly available spam datasets such as the [SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset).

**Note:** The dataset is not included in this repository. Please use a publicly available dataset or your dataset for training and testing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/youssefelzahar/spam_detection_model.git
   ```

2. Navigate to the project directory:
   ```bash
   cd spam_detection_model
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook
   ```

## Usage

1. Load the dataset and preprocess it using the provided scripts.
2. Extract features using BoW or TF-IDF methods.
3. Train and evaluate machine learning models for spam detection.
4. Integrate the model into your application for real-time classification.

## Modeling Techniques

The project implements and compares several models:

1. **Logistic Regression**: A baseline classifier for text classification tasks.
2. **Random Forest**: For handling non-linear relationships and feature importance analysis.
3. **Naive Bayes**: Specifically suited for text data classification.
4. **Support Vector Machines (SVM)**: For high-dimensional data classification.
5. **Gradient Boosting**: Advanced ensemble learning for better accuracy.

## Results

- **Evaluation Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-Score

- **Best Model:**
  The **Gradient Boosting Model** provided the best performance based on F1-Score, achieving:
  - Accuracy: `0.99`
  - Precision: `class 0-> 0.99 class 1-> 0.99`
  - Recall: `class 0-> 0.99 class 1-> 0.93`
  - F1-Score: `class 0-> 0.99 class 1-> 0.95`

For detailed evaluation metrics and visualizations, refer to the `model.ipynb` notebook.

## Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
