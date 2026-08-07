# Deep Learning Practical 1 – Breast Cancer Classification

## 📌 Project Overview

This project implements a **Deep Learning based Binary Classification system** for breast cancer diagnosis using the **Breast Cancer Wisconsin Dataset** available in Scikit-learn.

The practical focuses on understanding and comparing different neural network architectures and regularization techniques, including:

* Single-Layer Perceptron (SLP)
* Multi-Layer Perceptron (MLP)
* Early Stopping
* Dropout Regularization
* L2 Regularization
* Combined Regularization Model

The final model is designed as a **clinical decision-support model**, where Recall is given importance because missing a malignant tumour can have serious consequences.

---

## 🎯 Objectives

The main objectives of this practical are:

1. Load and understand the Breast Cancer dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Analyze class distribution and feature correlations.
4. Split the dataset into training and testing sets.
5. Apply feature scaling using `StandardScaler`.
6. Build a Single-Layer Perceptron as a baseline model.
7. Build Multi-Layer Perceptron models.
8. Study the effect of EarlyStopping.
9. Apply Dropout regularization.
10. Apply L2 regularization.
11. Compare the performance of different models.
12. Build a final combined deep learning model.
13. Evaluate models using Accuracy, Precision, Recall and F1-Score.
14. Understand the importance of Recall in medical diagnosis.

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Dataset** provided by Scikit-learn.

### Dataset Information

* **Total Samples:** 569
* **Total Features:** 30
* **Target:** Binary Classification
* **Malignant (0):** 212 samples
* **Benign (1):** 357 samples

### Target Classes

| Class | Meaning   |
| ----- | --------- |
| `0`   | Malignant |
| `1`   | Benign    |

The dataset contains numerical features related to breast cell characteristics.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow
* Keras

---

## 🔄 Project Workflow

```text
Dataset Loading
      ↓
Exploratory Data Analysis
      ↓
Class Distribution Analysis
      ↓
Feature Correlation Analysis
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Single-Layer Perceptron
      ↓
Multi-Layer Perceptron
      ↓
EarlyStopping
      ↓
Dropout Regularization
      ↓
L2 Regularization
      ↓
Model Comparison
      ↓
Final Combined Model
      ↓
Performance Evaluation
      ↓
Clinical Insights
```

---

# 🔍 Task 1: Data Loading, EDA & Preprocessing

The Breast Cancer Wisconsin dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target
```

### EDA Performed

The notebook performs:

* Dataset information analysis
* Statistical summary
* First few records inspection
* Class distribution analysis
* Feature correlation heatmap

### Class Distribution

The dataset contains:

* Malignant: **212**
* Benign: **357**

There is a mild class imbalance, but it is not severe enough to require special oversampling or undersampling techniques.

---

# ⚙️ Data Preprocessing

## Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

Stratified splitting is used to preserve the original class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

## Feature Scaling

`StandardScaler` is applied to standardize the input features.

The scaler is fitted only on the training data and then applied to both training and testing data.

After scaling:

* Mean ≈ 0
* Standard Deviation ≈ 1

Feature scaling is important because neural networks use gradient-based optimization methods such as **Adam**.

---

# 🧠 Task 2: Single-Layer Perceptron

A Single-Layer Perceptron is implemented as the baseline neural network model.

### Architecture

```text
Input Features: 30
        ↓
Output Neuron: 1
        ↓
Sigmoid Activation
```

### Configuration

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Metric: Accuracy
* Epochs: 50
* Batch Size: 32
* Validation Split: 10%

The SLP provides a baseline against which more advanced neural network architectures can be compared.

---

# 🧠 Multi-Layer Perceptron

A deeper neural network is implemented to learn more complex patterns from the dataset.

The MLP uses:

* Dense layers
* ReLU activation
* Sigmoid output activation
* Adam optimizer
* Binary Crossentropy loss

### Architecture

```text
30 Input Features
       ↓
Dense Layer
       ↓
ReLU
       ↓
Dense Layer
       ↓
ReLU
       ↓
Output Layer
       ↓
Sigmoid
```

The MLP is compared with the simpler Single-Layer Perceptron to understand the benefit of hidden layers.

---

# ⏹️ EarlyStopping

EarlyStopping is used to reduce overfitting and unnecessary training.

The model monitors validation loss and stops training when the validation performance stops improving.

The best model weights are restored using:

```python
restore_best_weights=True
```

### Benefits

* Reduces overfitting
* Saves training time
* Automatically stops unnecessary epochs
* Restores the best-performing model

---

# 🧩 Dropout Regularization

Dropout is introduced to improve the generalization capability of the neural network.

The practical uses a dropout rate of:

```text
0.3
```

This means approximately 30% of neurons are randomly deactivated during training.

### Benefits

* Reduces overfitting
* Prevents excessive dependence on individual neurons
* Improves generalization

---

# ⚖️ L2 Regularization

L2 regularization is applied to penalize large neural network weights.

The practical uses:

```text
L2 λ = 0.001
```

L2 regularization encourages the network to learn smoother and more stable representations.

It is particularly useful when features have correlations with each other.

---

# 🔬 Regularization Comparison

The notebook compares:

1. MLP without regularization
2. MLP + L1 Regularization
3. MLP + L2 Regularization
4. MLP + L1-L2 Regularization

The models are evaluated using test accuracy.

The comparison helps identify which regularization technique provides better generalization performance.

---

# 🏆 Final Combined Model

The final production model combines:

* **L2 Regularization**
* **Dropout**
* **EarlyStopping**

### Architecture

```text
Input: 30 Features
       ↓
Dense: 128 Neurons
ReLU + L2
       ↓
Dropout: 0.3
       ↓
Dense: 64 Neurons
ReLU + L2
       ↓
Dropout: 0.3
       ↓
Output: 1 Neuron
Sigmoid
```

### Configuration

| Parameter              | Value               |
| ---------------------- | ------------------- |
| Optimizer              | Adam                |
| Loss                   | Binary Crossentropy |
| L2 Lambda              | 0.001               |
| Dropout                | 0.3                 |
| Batch Size             | 32                  |
| Maximum Epochs         | 300                 |
| EarlyStopping Patience | 20                  |

The final model combines multiple techniques to improve generalization and reduce overfitting.

---

# 📈 Model Evaluation

The following evaluation metrics are used:

* Accuracy
* Precision
* Recall
* F1-Score

A comparison table is created for:

| Model                |
| -------------------- |
| SLP                  |
| MLP-ReLU             |
| MLP + EarlyStopping  |
| MLP + Dropout        |
| MLP + L2             |
| Final Combined Model |

The notebook identifies the best-performing model based on **Test Accuracy** and **Test F1-Score**.

---

# 🏥 Clinical Insight

For medical diagnosis, **Recall is especially important**.

A False Negative means that a malignant tumour is incorrectly classified as benign. This can be more dangerous than a False Positive because a missed malignant case may delay further medical examination.

Therefore, the notebook recommends considering a slightly lower classification threshold, such as:

```text
0.4
```

instead of the default:

```text
0.5
```

A lower threshold can increase Recall and reduce False Negatives, although it may also increase False Positives.

The model should be considered a **decision-support tool**, not a replacement for professional medical diagnosis.

---

# 📌 Key Learnings

Through this practical, the following concepts were explored:

* Exploratory Data Analysis
* Data preprocessing
* Feature scaling
* Neural network fundamentals
* Single-Layer Perceptron
* Multi-Layer Perceptron
* ReLU activation
* Sigmoid activation
* Binary Crossentropy
* Adam optimizer
* EarlyStopping
* Dropout
* L1 Regularization
* L2 Regularization
* L1-L2 Regularization
* Model comparison
* Accuracy
* Precision
* Recall
* F1-Score
* Classification threshold
* Medical classification considerations

---

# 📁 Project Structure

```text
Deep-Learning-Practical-1/
│
├── Deep_Learning_Pr_1.ipynb
├── Deep_Learning_Pr_1.html
└── README.md
```

---

# ▶️ How to Run

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

## 2. Open the Notebook

Open:

```text
Deep_Learning_Pr_1.ipynb
```

using:

* Jupyter Notebook
* JupyterLab
* Google Colab
* VS Code

## 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## 4. Run the Notebook

Run all cells sequentially.

For final verification:

```text
Kernel → Restart & Run All
```

The notebook should execute without errors.

---

# 📄 Files Included

### `Deep_Learning_Pr_1.ipynb`

Contains the complete implementation, explanations, visualizations, model training and evaluation.

### `Deep_Learning_Pr_1.html`

HTML export of the completed notebook for easy viewing and submission.

### `README.md`

Project documentation and implementation overview.

---

# 🚀 Conclusion

This practical demonstrates how deep learning models can be applied to binary classification using the Breast Cancer Wisconsin dataset.

Different neural network architectures and regularization techniques were implemented and compared. The final model combines **L2 Regularization, Dropout and EarlyStopping** to improve generalization.

For medical diagnosis support, **Recall is an important metric**, as reducing False Negatives is critical when detecting malignant cases.

Overall, the project provides a practical understanding of building, training, regularizing and evaluating neural networks using **TensorFlow/Keras**.

---

## 👨‍💻 Author

**Tirth Patel**

### Technologies

`Python` `TensorFlow` `Keras` `Scikit-learn` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Deep Learning`

---
