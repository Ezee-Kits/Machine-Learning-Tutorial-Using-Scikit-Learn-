# 📊 Machine Learning Tutorial Using Scikit-Learn

Welcome to the **Machine Learning Tutorial Using Scikit-Learn** repository!  
This is the official repository containing all the resources, datasets, Python scripts, and trained models used in my **YouTube video series** on **Machine Learning (ML) with Python and Scikit-Learn (sklearn)**.  

Whether you are a beginner or an intermediate learner, this repository provides everything you need for **hands-on practice**.

---

## 🔗 YouTube Series
You can follow the full tutorial series on my channel: [Ezee Kits](https://www.youtube.com/@Ezee_Kits)  
The series covers **predictive data analysis**, **classification**, **regression**, **model evaluation**, and **real-life ML applications** using Python.

---

## 📂 Repository Contents

### 1️⃣ PYTHON FILES
All `.py` scripts used for the tutorial are included, covering **Chapter 1 to Chapter 90**.  
These scripts demonstrate how to implement various ML concepts such as:
- Data preprocessing  
- Model training and evaluation  
- Visualization with matplotlib and seaborn  
- Working with classification and regression algorithms  

---

### 2️⃣ CSV FILES
The repository includes several CSV datasets used throughout the tutorials:  
- `titanic.csv` – Passenger survival data for classification tasks.  
- `football_predictions_train.csv` & `football_predictions_test.csv` – Data from **Prematips** for football match outcome predictions.  
- `stock_trading.csv` – Dataset for stock market prediction examples.  
- `tic_tac_toe.csv` – Dataset for game outcome prediction.  
- `iris.csv` – Classic flower dataset for classification.  
- `email_spam.csv` – Email spam detection dataset.  
- `house_prices.csv` – Dataset for regression and price prediction tasks.  

---

### 3️⃣ TRAINED MODELS FILES
Saved models for classification and regression tasks:  
- `.pickle` files – Many trained models saved in pickle format.  
- `.joblib` files – Many trained models saved in joblib format for faster loading.  

These allow you to **load pre-trained models** without retraining, ideal for testing or experimentation.

---

### 4️⃣ OTHER FILES
Includes other auxiliary files used in the tutorial videos, such as:
- Text files  
- Logs  
- Helper scripts  
- Any additional resources needed to follow along with the videos  

---

## 🧠 What You Will Learn
By exploring this repository, you will learn how to:
- Load and explore datasets using **pandas**  
- Preprocess data for machine learning  
- Train and test models using **scikit-learn**  
- Evaluate models with metrics like **accuracy, precision, recall, and R²**  
- Apply ML to real-life examples such as Titanic survival, stock prediction, football match prediction, and more  

---

## ⚙️ How to Use This Repository
1. **Clone the repository**
   ```bash
   git clone https://github.com/Ezee-Kits/Machine-Learning-Tutorial-Using-Scikit-Learn-.git
   cd Machine-Learning-Tutorial-Using-Scikit-Learn-
   ```

2. **Install Dependencies**
   Make sure Python 3.8+ is installed, then run:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```

3. **Open a Python Script**
   Each `.py` file corresponds to a chapter and contains **step-by-step explanations**. Run them in your IDE or Jupyter Notebook.

4. **Load a CSV Dataset**
   ```python
   import pandas as pd
   df = pd.read_csv('CSV_Files/titanic.csv')
   print(df.head())
   ```

5. **Load a Trained Model**
   ```python
   import joblib
   model = joblib.load('Trained_Models/logistic_regression_model.joblib')
   predictions = model.predict(df[['Pclass', 'Age', 'SibSp', 'Parch']])
   print(predictions)
   ```

---

## 📝 Repository Structure
```
Machine-Learning-Tutorial-Using-Scikit-Learn-/
│
├── PYTHON_FILES/             # Scripts for Chapters 1-90
│   ├── chapter_01.py
│   ├── chapter_02.py
│   └── ...
│
├── CSV_FILES/                # All datasets used in tutorials
│   ├── titanic.csv
│   ├── football_predictions_train.csv
│   ├── football_predictions_test.csv
│   ├── stock_trading.csv
│   ├── tic_tac_toe.csv
│   ├── iris.csv
│   ├── email_spam.csv
│   └── house_prices.csv
│
├── TRAINED_MODELS/           # Saved ML models
│   ├── model1.pickle
│   ├── model2.joblib
│   └── ...
│
├── OTHER_FILES/              # Other resources and helper files
│   └── ...
│
└── README.md                 # This file
```

---

## 👨‍💻 Author
**Peter**  
- YouTube Channel: [Ezee Kits](https://www.youtube.com/@Ezee_Kits)  
- Topics Covered: Python, Machine Learning, Data Analysis, Scikit-Learn, Real-Life ML Projects  

---

## ⚖️ License
This repository is open-source under the **MIT License**.  
You are free to use, modify, and distribute the code for **educational purposes**.

---

## 🌟 Quick Start Example
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('CSV_FILES/titanic.csv')

# Feature selection and target
X = df[['Pclass', 'Age', 'SibSp', 'Parch']]
y = df['Survived']

# Handle missing values
X['Age'].fillna(X['Age'].mean(), inplace=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 🔗 Useful Links
- [YouTube Tutorial Playlist](https://www.youtube.com/@Ezee_Kits)  
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
