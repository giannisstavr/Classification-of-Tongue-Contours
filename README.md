# Neural Networks for Understanding Tongue Shapes - Voiced Alveolar Approximant Classification

## Project Overview

This project focuses on developing a machine learning model, specifically a Convolutional Neural Network (CNN), to classify Ultrasound Tongue Imaging (UTI) data related to the voiced alveolar approximant (/r/ sound). The goal is to assist speech therapists in the diagnosis and treatment of speech impairments by automating the classification of tongue shapes during speech production.

Manual interpretation of UTI images is a labor-intensive and time-consuming process. This project addresses this challenge by harnessing the power of deep learning to automate the classification of UTI images into distinct tongue shapes, thereby minimizing the time required for this lengthy process and enabling useful medical resources to be utilized elsewhere.

The project explores various CNN architectures and techniques to handle challenges such as limited dataset size and class imbalance, ultimately achieving a classification accuracy of **77%** using a DenseNet121 adaptation.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Background](#background)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contact](#contact)
- [License](#license)

## Key Features

* **Automated UTI Classification:** Develops a CNN model to classify ultrasound tongue images, specifically for the voiced alveolar approximant.
* **Four Tongue Shape Classes:** Classifies images into 'tip_up', 'mid_bunched', 'front_up', and 'front_bunched' tongue shapes.
* **Deep Learning with CNNs:** Utilizes Convolutional Neural Networks, leveraging their strengths in image pattern recognition.
* **Transfer Learning:** Employs pre-trained models (Inception V3, DenseNet121, ResNet50, InceptionResNet V2) to overcome limitations of small datasets.
* **Class Imbalance Handling:** Investigates and implements techniques like Class Weights, Data Augmentation, and Synthetic Minority Oversampling Technique (SMOTE) to address dataset imbalance.
* **Optimized Performance:** Achieves a notable accuracy of 77% with the DenseNet121 model combined with SMOTE.

## Background

Speech therapy heavily relies on Ultrasound Tongue Imaging (UTI) to visualize tongue movements during speech production, aiding in diagnosis and targeted therapy. The voiced alveolar approximant (/r/ sound), particularly significant in Scottish English due to the language being rhotic, presents a common challenge in pronunciation due to its intricate articulation and the multiple tongue shapes it can involve (e.g., "bunched" and "retroflex"). Automating the analysis of these tongue shapes can significantly aid therapists. This project builds upon the principles of machine learning, neural networks, and specifically CNNs, which excel in image classification tasks by learning hierarchical patterns from data.

## Dataset

The project utilized a labelled UTI dataset composed of ultrasound images of the voiced alveolar approximant (/r/). The dataset was initially structured with a `.csv` file linking image filenames to their corresponding `tongue_shape` labels. Due to initial challenges, the dataset structure was refined to separate images into four distinct folders, each representing one of the four tongue shape classes:

* `tip_up`
* `mid_bunched`
* `front_up`
* `front_bunched`

The dataset, initially around 400 images, was expanded to approximately 500 images for the refined 4-class CNN model. Addressing class imbalance within this dataset was a significant part of the project's methodology.

## Methodology

The project's methodology involved an iterative process of model development, refinement, and optimization.

1.  **Initial 4-Class CNN:**
    * An initial CNN model was constructed based on a binary model guide from GitHub.
    * Data was prepared by filtering and converting an Excel document to a `.csv` file, linking image filenames to `tongue_shape` labels.
    * The dataset was split into train (80%), validation (16%), and test (4%) sets.
    * Images were loaded and preprocessed (resized to 256x256 and normalized).
    * Transfer learning was applied using a pre-trained Inception V3 model with weights from RadImageNet.
    * Additional layers (pooling, dense with ReLU, dropout, and a final softmax dense layer) were added.
    * Adam optimizer with `LearningRateScheduler` and `EarlyStopping` was used for training.

2.  **Binary Base Model (Attempted):**
    * An attempt was made to create an "in-house" pre-trained model using the RadImageNet dataset, focusing on thyroid ultrasound images.
    * Challenges related to large data volume, Google Drive synchronization, and persistent `InvalidArgumentError` during training led to discontinuing this approach.

3.  **Refinement and Update of the 4-Class Model:**
    * The dataset structure was revamped, organizing images into class-specific subfolders to minimize data leaks.
    * `ImageDataGenerator` was used for creating data pipelines.
    * **Class Imbalance Handling:**
        * **Class Weights:** Implemented `compute_class_weight` to apply multipliers to the loss function, focusing on minority classes.
        * **Data Augmentation:** A custom script was developed to balance classes by generating augmented images (rotations, shifts, flips, zoom) for minority classes, applied only to the training dataset.
        * **SMOTE (Synthetic Minority Oversampling Technique):** This technique was applied to the training dataset to generate new synthetic samples for minority classes by interpolating between existing samples.

## Results

The most successful approach for handling class imbalance and achieving optimal classification performance was the implementation of **SMOTE**.

| Pre-Trained Model | SMOTE | Class Weights | Augmentation |
| :---------------- | :---- | :------------ | :----------- |
| IRV2 | 69.69% | 35.19% | 35.19% |
| DenseNet121 | **76.77%** | 16.66% | 20.37% |
| ResNet50 | 71.71% | 46.30% | 37.04% |
| Inception V3 | **76.77%** | 40.74% | 11.11% |

* **Optimal Accuracy:** The **DenseNet121** model, when combined with SMOTE, achieved the highest accuracy of **76.77%**. Inception V3 with SMOTE also yielded the same accuracy.
* **SMOTE's Superiority:** SMOTE consistently outperformed Class Weights and Data Augmentation across all tested pre-trained models, often by over 100% in terms of accuracy.
* **Loss Reduction:** The SMOTE-implemented model with DenseNet121 achieved a final loss of 0.5947.

Further analysis of regularization techniques (L1, L2, Dropout) in combination with SMOTE showed that adding these did not further improve the model's performance beyond the initial SMOTE implementation.

## Setup and Installation

To set up this project locally, you will need Python and Git installed.

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
    ```
    (Note: Specific versions of libraries might be required based on your original environment. You might need to add other libraries based on your code, e.g., `Pillow` for `load_img`)

3.  **Dataset Preparation:**
    * You will need to organize your ultrasound tongue imaging data into separate folders for each class (e.g., `data/train/tip_up`, `data/train/mid_bunched`, etc.).
    * The `filter_file_names.py` script (refer to `7.1 Filter File Names` in thesis) can be used to process your initial CSV and ensure image consistency.
    * The scripts for creating train/validation/test splits and handling class separation (refer to `7.3 Binary Base Model` and `7.4 Refined 4-Class Model` in thesis) will be crucial for preparing your dataset as per the refined model's requirements.

## Usage

The primary code for this project is likely structured into Python scripts (e.g., for data loading, model definition, training, and evaluation).

**To train the model (assuming your data is prepared in class-specific subfolders):**

1.  **Modify paths:** Update the data directory paths in your Python scripts (e.g., `data_dir` variables) to point to your local dataset.
2.  **Run the main training script:**
    This would typically involve running the script that defines the model, loads data, applies SMOTE, and initiates training. Based on your thesis, this would involve adapting the code used for the "Refined 4-Class Model SMOTE" implementation.

    ```bash
    python train_model.py # Or whatever your main script is named
    ```
    (Replace `train_model.py` with the actual name of your training script.)

**Key parameters to consider and adjust:**

* `batch_size`
* `epochs`
* `min_lr` and `factor` for `ReduceLROnPlateau`
* `learning_rate` for the Adam optimizer
* `patience`, `start_epoch`, `min_delta` for `EarlyStopping`
* The choice of pre-trained model (DenseNet121, Inception V3, etc.)

## Future Work

Based on the findings and challenges encountered during this project, the following areas are recommended for future work:

* **SMOTE and Under-sampling Combination:** Explore combining SMOTE with an under-sampling technique to potentially further increase model accuracy. It has been shown that there is the possibility to combine both techniques.
* **Advanced Learning Rate Scheduling:** Investigate the combined use of `ReduceLROnPlateau` along with the `LearningRateScheduler` to provide the optimal learning rate for specific iterations. Their combination could provide the optimal learning rate for the specific iteration.
* **Custom Model Architecture:** Design and implement a custom CNN architecture from scratch, specifically tailored for this classification task. This could lead to better accuracy and loss metrics, thus a better overall model, as the model will be personally designed for the specific classification. A similar study proved that a custom-made model architecture could be more beneficial than transfer learning.
* **Larger and More Balanced Dataset:** Acquire a significantly larger and more balanced dataset of UTI images to improve model generalization and performance.
* **Binary Classification Model:** Repurpose the existing dataset for a binary classification task (as suggested by your initial excel file's column for two classes), potentially simplifying the problem and improving accuracy for specific use cases.
* **Multi-Modal Input:** Integrate additional inputs such as sound and waveforms, as used by professionals in realistic scenarios, to provide a more comprehensive classification model.

## Contact

For any questions or collaborations, please feel free to reach out.
