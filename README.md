
# 🛡 Credit Card Fraud Detection

This project is a Machine Learning pipeline designed to detect fraudulent credit card transactions. The dataset is highly imbalanced and contains anonymized features for confidentiality.

## 📊 Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Features: 30 columns (V1 to V28, Time, Amount)
- Target: Class (0 = Not Fraud, 1 = Fraud)

## ⚙️ Workflow

1. Data Loading and Preprocessing
   - Drop duplicates
   - Normalize Amount and Time columns
   - Check for class imbalance

2. Resampling
   - Apply SMOTE to balance the dataset

3. Modeling
   - Train a Random Forest Classifier
   - Evaluate using classification report

4. Evaluation
   - Metrics used:
     - Accuracy
     - Precision
     - Recall
     - F1-Score

5. Visualization
   - Countplot of classes
   - Confusion matrix heatmap

## 📈 Results

| Metric     | Value (on test set) |
|------------|---------------------|
| Accuracy   | ~99%                |
| Precision  | High                |
| Recall     | Optimized via SMOTE |
| F1-Score   | Balanced            |

> ⚠️ Focus was placed on improving Recall due to the cost of missing fraudulent transactions.

## 🧰 Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn (for SMOTE)

## 🚀 How to Run

`bash
# Clone the repository
git clone https://github.com/Heithamking/creditcard_fraud_detection.git
cd creditcard_fraud_detection

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook creditcard_classification.ipynb


Notes
You must download the dataset creditcard.csv manually from Kaggle and place it in the project root.
SMOTE is used only for the training data to avoid data leakage.
🧠 Author
Heitham Bendjamaa
Data Science 
