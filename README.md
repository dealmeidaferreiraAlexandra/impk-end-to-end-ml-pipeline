# 🧠 End-to-End ML Pipeline (Titanic Dataset)

An interactive machine learning project that demonstrates a **complete end-to-end pipeline**, from raw data to model comparison and deployment.

---

## 🌐 Live Demo

👉 https://impk-end-to-end-ml-pipeline-8bkjwfrgvgjhurmkinthuj.streamlit.app

---

## 🧠 What this project does

This project builds a full machine learning workflow using a real dataset:

* 📊 **Data cleaning** — handles missing values and inconsistencies
* 🧩 **Feature engineering** — creates meaningful features from raw data
* 🧠 **Model training** — trains multiple models
* 📈 **Model comparison** — evaluates and compares performance

Models used:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)

The app allows you to:

* Upload a dataset (or use the built-in Titanic dataset)
* Run the full pipeline step-by-step
* Compare multiple models side by side
* Visualize performance metrics
* Predict outcomes for individual samples
* Download structured reports

---

## 🎯 Why this matters

Most ML projects focus only on training models.

This project shows the **full real-world pipeline**:

* data preparation
* feature engineering
* model selection
* evaluation
* deployment

👉 It reflects how machine learning systems are actually built in production.

---

## 🚀 Features

* 📥 CSV dataset upload
* 🧹 Data cleaning pipeline
* 🧩 Feature engineering (domain-aware features)
* 🧠 Multiple model training (LR, RF, SVM)
* 📊 Model comparison (Accuracy, Precision, Recall, F1, ROC AUC)
* 📉 Visual performance charts
* 🔍 Single sample prediction
* ⚠️ Model disagreement detection
* 💾 Downloadable JSON report
* 🌐 Interactive Streamlit interface

---

## 🛠 Tech Stack

* Python
* Streamlit
* scikit-learn
* pandas
* numpy
* joblib

---

## ⚙️ How it works

1. **Load Data**
   Upload a CSV file or use the default Titanic dataset.

2. **Clean Data**
   Missing values are handled and normalized.

3. **Feature Engineering**
   New features are created (e.g., family size, titles, cabin info).

4. **Train Models**
   Three models are trained in parallel:

   * Logistic Regression
   * Random Forest
   * SVM

5. **Compare Results**
   Metrics are displayed side by side and the best model is selected.

6. **Predict**
   The user can select a row and see predictions from all models.

---

## ▶️ Run locally

```bash
git clone https://github.com/dealmeidaferreiraAlexandra/impk-end-to-end-ml-pipeline.git
cd impk-end-to-end-ml-pipeline

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python train.py

streamlit run app.py
```

---

## 🧪 Notes

* The dataset used is the **Titanic dataset (Kaggle)**
* Feature engineering includes domain-specific transformations (titles, family size, etc.)
* Models are trained locally and saved for reuse
* Metrics are computed on a validation split
* Results may vary depending on preprocessing choices

---

## 👩‍💻 Author

Developed by Alexandra de Almeida Ferreira

GitHub:
[https://github.com/dealmeidaferreiraAlexandra](https://github.com/dealmeidaferreiraAlexandra)

LinkedIn:
[https://www.linkedin.com/in/dealmeidaferreira](https://www.linkedin.com/in/dealmeidaferreira)

---

## 📄 License

This project is licensed under the MIT License.




