# Chest X-Ray Pneumonia Detection using DenseNet121

This project implements a **binary classification model** for pneumonia detection from chest X-rays using **transfer learning with DenseNet121**.


## Pneumonia Detection using Chest X-ray Images

Pneumonia is an inflammatory condition primarily affecting the lungs, characterized by symptoms such as cough, chest pain, fever, and difficulty breathing. The goal of this project is to develop an automated system for detecting and classifying pneumonia in chest X-ray images.



## Motivation

The motivation behind this project is to leverage artificial intelligence to accurately detect and classify pneumonia using medical imaging. Early and reliable detection can significantly aid healthcare professionals in providing timely diagnosis and treatment, reducing the dependency on manual radiology interpretations.

## Approach

This project applies transfer learning using DenseNet121 to build a binary classification model capable of distinguishing between pneumonia and normal cases. The model was implemented in Python using Keras/TensorFlow and trained in PyCharm with GPU acceleration.


**The training process included:**

1. Data preprocessing with augmentation (rotation, zoom, horizontal flips).
2. Model fine-tuning on chest X-ray images.
3. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
4. Visualization of model interpretability using Grad-CAM heatmaps.


## Key Technologies Used:

- Python
- TensorFlow / Keras
- DenseNet121 (Transfer Learning)
- scikit-learn
- Matplotlib & Seaborn
- Grad-CAM (for explainability)
  
---

- **Frameworks**: TensorFlow / Keras
- **Dataset**: [Kermany’s Chest X-Ray Pneumonia dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   It is organized into three folders: train, test, and val, each containing X-ray images (JPEG format) categorized into two classes: Pneumonia and Normal.
Total images: 5,863
Train: ~5,216 images
Validation: ~16 images
Test: ~624 images
- **Key Techniques**: Transfer Learning, Data Augmentation, Grad-CAM visualization
- **Performance**: Achieved ~95% training accuracy, ~87.5% validation accuracy.

##  Repository Structure
```

Chest-X-Ray-Pneumonia-Detection/
│── data/                  # dataset (train/val/test)
│── notebooks/             # Jupyter notebooks
│── src/                   # source code
│   ├── densenet\_model.py # DenseNet121 model builder
│   ├── train.py           # training & evaluation script
│   └── gradcam.py         # Grad-CAM visualization
│── results/               # trained models, plots
│── README.md              # project documentation
│── requirements.txt       # dependencies
│── .gitignore             # ignored files

````

## Results

* **Validation Accuracy**: \~87.5%
* **F1-Score (Test)**: \~0.50 (imbalanced dataset — further tuning needed)
* **Observation**: Model performs well but struggles with **small test set & noise in X-rays**.

##  Interpretability (Grad-CAM)

The repo includes **Grad-CAM visualizations** to highlight pneumonia-infected regions in X-rays.

##  Note

While the training accuracy reached \~95%, **generalization on test data was lower** (\~50% F1). This highlights the **importance of robust validation, cross-validation, and dataset balancing** in medical ML.

---

