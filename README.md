Churn Modelling — Data Preprocessing Pipeline

A step-by-step machine learning preprocessing pipeline built on the **Bank Churn Modelling dataset**. This project covers the full journey from raw, messy data to a clean, encoded, and scaled dataset ready for model training.

---

## 📋 Project Overview

The goal is to preprocess customer banking data to predict churn (whether a customer will leave the bank). Each notebook in this project handles one stage of the preprocessing pipeline:

| Step | Notebook | Description |
|------|----------|-------------|
| 0 | `0_handle_missing_values.ipynb` | Impute missing ages with mean/median; predict missing genders using an LLM (Groq API) |
| 1 | `1_handling_outliers.ipynb` | Detect outliers using KDE plots, box plots, 3-sigma rule, and IQR method |
| 2 | `2_feature_binning.ipynb` | Bin `CreditScore` into ordinal categories (Poor → Excellent) |
| 3 | `3_feature_encoding.ipynb` | One-hot encode nominal columns (Geography, Gender); label encode ordinal column (CreditScoreBins) |
| 4 | `4_feature_scaling.ipynb` | Standardize continuous features (Age, Tenure, Balance, EstimatedSalary) using Z-score scaling |

### Data Flow

```
ChurnModelling.csv  (raw)
        ↓  0_handle_missing_values
ChurnModelling_Missing_values_handled.csv
        ↓  1_handling_outliers
ChurnModelling_Outliers_Handled.csv
        ↓  2_feature_binning
ChurnModelling_Binning_Applied.csv
        ↓  3_feature_encoding
ChurnModelling_Encoded.csv
        ↓  4_feature_scaling
ChurnModelling_Final.csv  ✅ model-ready
```

---

## 🐍 What is Anaconda?

**Anaconda** is a free distribution of Python that comes pre-packaged with hundreds of data science libraries (NumPy, pandas, scikit-learn, etc.) and the **conda** package manager. Instead of manually installing Python and every library one by one, Anaconda gives you everything in one installer.

Think of it like this:
- **Python alone** = a kitchen with no equipment
- **Anaconda** = a fully equipped kitchen, ready to cook

Download it here: https://www.anaconda.com/download

---

## 📦 What is a Virtual Environment?

A **virtual environment** is an isolated Python workspace. It has its own Python version and its own set of installed packages, completely separate from every other project on your machine.

**Why does this matter?**

Imagine two projects:
- Project A needs `pandas==1.5`
- Project B needs `pandas==3.0`

Without environments, installing one would break the other. With environments, each project lives in its own bubble and never interferes with anything else.

```
Your computer
├── env_project_a/    ← pandas 1.5, scikit-learn 1.0
├── env_project_b/    ← pandas 3.0, scikit-learn 1.8
└── env_this_project/ ← exact versions in requirements.txt
```

---

## 🚀 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a conda environment

```bash
conda create -n bprmls python=3.11
```

This creates a fresh isolated environment named `bprmls` running Python 3.11.

### 3. Activate the environment

```bash
conda activate bprmls
```

Your terminal prompt will change to show `(bprmls)` — this means you are now inside the environment. Any packages you install from this point go into this environment only.

### 4. Install dependencies

```bash
pip install -r requirments.txt
```

### 5. Set up your `.env` file (required for notebook 0)

Notebook `0_handle_missing_values.ipynb` uses the **Groq API** to predict missing gender values from customer names. You need a free API key from https://console.groq.com.

Create a file named `.env` in the root of the project:

```
GROQ_API_KEY=your_api_key_here
```

> ⚠️ Never commit your `.env` file to GitHub. It is already listed in `.gitignore`.

### 6. Place the raw data

Make sure the raw dataset is placed at:

```
data/raw/ChurnModelling.csv
```

The `data/processed/` folder will be populated automatically as you run the notebooks in order.

### 7. Open Jupyter and run notebooks in order

```bash
jupyter notebook
```

Run notebooks **0 → 1 → 2 → 3 → 4** in sequence. Each notebook reads the output CSV from the previous step.

---

## 📁 Project Structure

```
├── data/
│   ├── raw/
│   │   └── ChurnModelling.csv
│   └── processed/
│       ├── ChurnModelling_Missing_values_handled.csv
│       ├── ChurnModelling_Outliers_Handled.csv
│       ├── ChurnModelling_Binning_Applied.csv
│       ├── ChurnModelling_Encoded.csv
│       └── ChurnModelling_Final.csv
├── 0_handle_missing_values.ipynb
├── 1_handling_outliers.ipynb
├── 2_feature_binning.ipynb
├── 3_feature_encoding.ipynb
├── 4_feature_scaling.ipynb
├── requirments.txt
├── .env                  ← you create this (not committed)
└── README.md
```

---

## 📚 Techniques Covered

**Missing Value Handling**
- Row/column deletion
- Mean & median imputation for numerical features
- LLM-based imputation for categorical features (Groq API)

**Outlier Detection**
- KDE distribution plots
- Box plots
- 3-Sigma (empirical) rule
- IQR method

**Feature Binning**
- Custom binning of `CreditScore` into interpretable categories: `Poor / Fair / Good / Very Good / Excellent`

**Feature Encoding**
- One-hot encoding for nominal variables (`Geography`, `Gender`) using `pd.get_dummies`
- Ordinal / label encoding for `CreditScoreBins` using a manual mapping dictionary
- sklearn `OneHotEncoder` and `LabelEncoder` as an alternative approach

**Feature Scaling**
- Z-score standardization using `sklearn.preprocessing.StandardScaler` on `Age`, `Tenure`, `Balance`, `EstimatedSalary`

---

## 🔧 Requirements

See `requirments.txt` for the full list. Key libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `openai`
- `groq`
- `python-dotenv`

---

## ⚠️ Notes

- Notebooks must be run **in order** (0 → 4) since each reads the output of the previous one.
- The `.env` file is required only for notebook 0. All other notebooks work without an API key.
- The `data/processed/` folder is gitignored. Run the notebooks to regenerate the processed files.
