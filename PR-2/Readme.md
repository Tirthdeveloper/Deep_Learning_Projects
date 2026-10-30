# Deep Learning Practical 2 – Adult Income Classification

A deep learning practical using an Artificial Neural Network (ANN) to predict whether a person's annual income is **greater than $50K** or **$50K or less**, based on demographic and employment-related attributes from the Adult Income dataset.

## 📌 Project Overview

This notebook explores how different neural-network design choices affect binary classification performance. It builds a baseline ANN and compares activation functions, weight initializers, loss functions, batch normalization, and optimizers before training a final combined model.

**Target column:** `income`

- `0` – Income is `<=50K`
- `1` – Income is `>50K`

## 📂 Files

| File | Description |
|---|---|
| `Deep_Learning_pr_2.ipynb` | Main Jupyter Notebook containing data preparation, experiments, model training, and evaluation |
| `adult.csv` | Adult Income dataset required by the notebook (not included in this upload) |

## 🧰 Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- TensorFlow / Keras
- Scikit-learn
- Jupyter Notebook or Google Colab

## ⚙️ Workflow

### 1. Data Loading and Preprocessing

The notebook:

1. Loads `adult.csv`.
2. Removes leading and trailing whitespace from text columns.
3. Replaces `?` values with missing values and drops rows containing missing values.
4. Removes the `fnlwgt` and `education` columns.
5. Converts the `income` target into a binary label.
6. Explores the class distribution, age, working hours, education number, and numeric-feature correlations.
7. Separates features and target, applies one-hot encoding to categorical features, and standardizes selected numeric columns.
8. Splits the data into training and testing sets using an 80:20 split with stratification.

### 2. Baseline ANN

A reusable `build_ann()` function is used to create configurable feed-forward neural networks. The baseline model is trained and evaluated using a confusion matrix, precision, recall, F1-score, and ROC-AUC.

### 3. Activation Function Comparison

The notebook compares:

- ReLU
- Tanh
- Sigmoid
- ELU

It also examines ReLU dead-neuron behaviour and the gradient magnitudes in a sigmoid network.

### 4. Weight Initialization Comparison

The following initializers are tested:

- Glorot Uniform
- Glorot Normal
- He Uniform
- He Normal
- Zeros

Validation-accuracy convergence and weight distributions are visualized. The notebook also investigates the effect of zero initialization.

### 5. Loss Functions and Class Imbalance

The experiments include:

- Binary Cross-Entropy (BCE)
- Mean Squared Error (MSE)
- Class-weighted BCE
- Focal Loss

Precision, recall, and F1-score are used to compare classification behaviour, especially for the positive income class.

### 6. Batch Normalization

A network with batch normalization is trained and compared with the baseline. The notebook visualizes validation loss and accuracy and inspects the learned gamma and beta parameters.

### 7. Optimizer Comparison and Final ANN

The notebook compares SGD, SGD with momentum, RMSprop, Adam, and explicitly configured Adam. It then trains a final combined ANN using:

- Hidden layers: 128 and 64 units
- ReLU activation
- He Normal initialization
- Batch normalization
- Adam optimizer
- Binary Cross-Entropy loss
- Class weights
- 80 training epochs

The final model is evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and ROC curves. A results table compares the baseline, weighted BCE, focal loss, batch-normalized, and final combined models.

## 🏗️ Model Architecture

The configurable ANN uses two hidden Dense layers (128 and 64 units by default) and a single sigmoid output for binary classification. Batch normalization can be enabled between Dense layers and their activation functions.

```text
Input Features
      |
Dense Layer (128 units)
      |
[Optional Batch Normalization]
      |
Activation (ReLU by default)
      |
Dense Layer (64 units)
      |
[Optional Batch Normalization]
      |
Activation (ReLU by default)
      |
Dense Layer (1 unit, Sigmoid)
      |
Income Prediction (0 or 1)
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn jupyter
```

### 3. Add the dataset

Download or obtain the Adult Income dataset and place the CSV file named `adult.csv` in the location expected by the notebook.

The notebook currently loads the dataset using:

```python
df = pd.read_csv("/content/adult.csv")
```

This path is suitable for Google Colab when the file is available at `/content/adult.csv`. If running locally, update the path to the location of your CSV file.

### 4. Run the notebook

Open `Deep_Learning_pr_2.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run the cells in order.

## 📊 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of positive predictions that are correct |
| Recall | Proportion of actual positive cases identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Measures how well the model separates the two classes |
| Confusion Matrix | Shows correct and incorrect predictions by class |

## 📈 Visualizations

The notebook generates visualizations including:

- Income class distribution
- Age distribution by income
- Working hours by income
- Education-number distribution
- Numeric-feature correlation heatmap
- Training and validation accuracy curves
- ReLU activation distribution
- Sigmoid gradient magnitudes
- Weight-initialization convergence and weight distributions
- Loss-function and batch-normalization comparisons
- Optimizer convergence
- ROC curves

## 📝 Notes

- The `adult.csv` dataset is required but was not included with the uploaded notebook.
- Run all notebook cells in order because later experiments reuse variables, models, and training results created earlier.
- The notebook defines and prints evaluation metrics; this README does not claim a particular model accuracy or best-performing configuration.
- The final results depend on the dataset contents, library versions, and training execution.

## 👨‍💻 Author

**Tirth Patel**

---

⭐ If you find this practical useful, consider giving the repository a star!
