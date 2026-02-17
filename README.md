# Sampling Techniques on Imbalanced Credit Card Dataset

## 📌 Overview

Real-world datasets are often highly imbalanced, especially in domains like fraud detection.  
This project explores how different sampling techniques affect the performance of various machine learning models on an imbalanced credit card dataset.

The goal is to:
- Balance the dataset
- Apply multiple sampling strategies
- Train different ML models
- Compare accuracy
- Visualize results
- Identify the best model–sampling combination

---

## 🎯 Objectives

- Understand class imbalance problems
- Apply five sampling techniques
- Train five machine learning models
- Compare performance using accuracy
- Visualize results using graphs
- Determine optimal sampling + model pair

---

## 📂 Dataset

Credit Card Fraud Dataset (highly imbalanced)

Downloaded from:
```url
https://github.com/AnjulaMehto/Sampling_Assignment
```

Target column: `Class`

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

---

## 🧪 Sampling Techniques Implemented

1. Random Oversampling  
2. SMOTE  
3. Random Undersampling  
4. SMOTEENN  
5. ADASYN  

---

## 🤖 Machine Learning Models Used

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Naive Bayes  
- K-Nearest Neighbors  

---

## 🔁 Workflow

1. Load dataset
2. Separate features and target
3. Apply feature scaling
4. Balance dataset using sampling techniques
5. Split into train/test sets
6. Train five ML models on each sampled dataset
7. Compute accuracy
8. Store results in a table
9. Visualize results using graphs
10. Identify best combination

---

## 📊 Accuracy Table (%)

| Model | RandomOver | SMOTE | RandomUnder | SMOTEENN | ADASYN |
|------|-----------|-------|------------|---------|-------|
| Logistic | 93.14 | 93.14 | 50.0 | 93.86 | 93.14 |
| DecisionTree | 100.0 | 98.37 | 75.0 | 99.32 | 98.69 |
| RandomForest | 99.67 | 99.35 | 50.0 | 100.0 | 98.04 |
| NaiveBayes | 77.45 | 75.82 | 25.0 | 74.74 | 70.92 |
| KNN | 96.41 | 94.77 | 25.0 | 97.27 | 94.12 |

---

## 🏆 Best Result

### ✅ Random Forest + SMOTEENN  
### 🎯 Accuracy: **100%**

---

## 📈 Visualizations

The project includes:

- Heatmap (Models vs Sampling)
-   <img width="606" height="559" alt="download" src="https://github.com/user-attachments/assets/db9e0b5d-492c-4745-9964-12305d6b94ba" />

- Average accuracy per sampling method
-   <img width="562" height="602" alt="download" src="https://github.com/user-attachments/assets/2252bbf2-548b-444f-ba30-e4ca8c3f069a" />

- Final grouped comparison chart
-   <img width="571" height="549" alt="download" src="https://github.com/user-attachments/assets/74dcfee9-47d0-4324-9364-27d795d07922" />


These help clearly analyze performance trends.

---

## 🔍 Observations

- Tree-based models perform best on imbalanced data
- Random Forest consistently achieves highest accuracy
- Random undersampling causes major accuracy loss
- Naive Bayes performs weakest
- Oversampling methods outperform undersampling
- SMOTEENN gives best overall result

---

## ✅ Conclusion

Sampling has a huge impact on model performance.

After testing all combinations:

### 🥇 Best combination:
**Random Forest + SMOTEENN**

This approach is recommended for highly imbalanced classification problems.

---

## 🚀 How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```

Run:

```bash
python anjula_assignmeny_1.py
```
or execute in Jupyter Notebook.

📌 Author
Anshul Kaushal
Computer Engineering Student

⭐ Acknowledgement
Academic assignment on sampling techniques for imbalanced datasets.
