# 📈 PolyRegress Studio: Polynomial Regression Model & Streamlit WebApp

An interactive, end-to-end Machine Learning web application and modeling suite for **Polynomial Regression** applied to advertising channels and sales optimization.

---

## 🌟 Overview & Key Highlights

In marketing and advertising, linear regression models frequently fail because they assume a straight-line, independent relationship between each ad channel and revenue. In reality:
1. **Multi-Channel Synergies**: Simultaneous investments in **TV and Radio** amplify consumer recall more than either channel alone ($TV \times Radio$ interaction).
2. **Diminishing Marginal Returns (Saturation)**: Additional advertising spend yields smaller incremental gains as audiences become saturated ($-\beta \cdot TV^2$).

**Polynomial Regression** captures these non-linearities and synergy interactions, boosting test set prediction accuracy ($R^2$) from **~90.6% (Linear Baseline)** up to **~95.3% (Polynomial Degree 2)**.

---

## 🚀 Application Structure

The Streamlit web application is divided into three comprehensive pages accessible via the sidebar:

### 1. 🏠 Homepage (Overview & Exploratory Data Analysis)
- **Executive Metric Cards**: Total records, feature count, target mean, baseline linear $R^2$, and polynomial accuracy gain.
- **Interactive Dataset Explorer**: Data inspection with sorting, searching, and statistical distributions (`describe()`).
- **Correlation Heatmap**: Pearson correlation matrix with Plotly heatmaps.
- **Non-Linear Trendlines**: Feature vs Sales scatter plots fitted with LOWESS curves to highlight non-linear saturation.
- **3D Synergy Visualizer**: Interactive 3D surface plot demonstrating how multi-channel investments compound sales.
- **Distribution & Outlier Analysis**: Histograms with violin plots and boxplots.

### 2. 🧠 Model Page (Training, Diagnostics & Live Predictions)
- **Hyperparameter Controls**:
  - Polynomial Degree ($d \in [1, 5]$)
  - Feature selection & target selection
  - Interaction terms toggle (pairwise only vs full powers)
  - Regularization models: **Ordinary Least Squares (OLS)**, **Ridge (L2)**, **Lasso (L1)**, and **ElasticNet**
  - Train/Test split ratio and random state
- **Performance Evaluation Scorecard**:
  - Train $R^2$, Test $R^2$, Adjusted $R^2$, RMSE, MAE, and 5-Fold Cross-Validation score.
- **Polynomial Feature Expansion & Weights**:
  - Table of all generated polynomial terms and interactive bar chart of top standardized coefficients.
- **Visual Diagnostics**:
  - Actual vs Predicted scatter plot with $y=x$ ideal line.
  - Residuals vs Fitted values (homoscedasticity check).
  - Residual histogram & normality check.
  - Sample tracking line plot.
- **Bias-Variance Tradeoff Analysis**:
  - Interactive curve comparing Train RMSE vs Test RMSE across degrees 1 to 5 to visualize underfitting vs overfitting sweet spot.
- **🔮 Live What-If Scenario Predictor**:
  - Interactive sliders for TV, Radio, and Newspaper ad budgets.
  - Instant predicted sales with an animated sales velocity gauge.
  - **Marginal Return Sensitivity Analysis**: Calculates marginal return of +$10k spend per channel from the current budget.
- **📁 Batch Prediction & Model Export**:
  - Upload an unlabelled CSV and download generated predictions.
  - Download trained pipeline (`.joblib`) and metrics report (`.json`).

### 3. ❓ Help Page (Theory, Math & FAQ)
- **Mathematical Formulations**: Mathematical definitions for Degree $d$ polynomial and multi-variable interaction expansions.
- **Domain Context**: Why polynomial regression fits advertising and marketing dynamics.
- **Bias-Variance & Regularization Guide**: Explaining Ridge and Lasso loss functions with LaTeX equations.
- **Metrics Cheat Sheet**: Formulas and guidelines for interpreting $R^2$, Adjusted $R^2$, RMSE, MAE, and CV scores.
- **Interactive FAQ Accordion**: Common questions and step-by-step guidance.

---

## 📦 Project Files

| File | Description |
| :--- | :--- |
| `advertising.csv` | Dataset containing TV, Radio, Newspaper advertising spends and resulting Sales |
| `train_model.py` | Standalone script to train, evaluate across degrees, and export model pipeline |
| `model.joblib` | Saved scikit-learn pipeline (PolynomialFeatures + StandardScaler + Ridge) |
| `model_metrics.json`| Saved performance metrics, evaluation comparison, and coefficients |
| `app.py` | Full-featured Streamlit web application |
| `run_app.bat` | Double-click Windows launcher for the Streamlit app |
| `requirements.txt` | Python library requirements |

---

## 💻 How to Run

### Method 1: Using the Windows Launcher
Simply double-click the **`run_app.bat`** file in the project directory.

### Method 2: Command Line with Anaconda Python
Run the following in PowerShell or Command Prompt:

```powershell
& "C:\Users\Windows 11\anaconda3\python.exe" -m streamlit run app.py
```

Or if Streamlit is in your system PATH:
```bash
streamlit run app.py
```

### Method 3: Retraining the Model Standalone
To run the standalone model evaluation and export pipeline:
```powershell
& "C:\Users\Windows 11\anaconda3\python.exe" train_model.py
```

---

## 📊 Summary of Model Performance

| Model / Degree | Poly Features | Train $R^2$ | Test $R^2$ | Test RMSE | 5-Fold CV $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Degree 1 (Linear)** | 3 | 0.9001 | 0.9059 | 1.7052 | 0.8793 | Underfitting (High Bias) |
| **Degree 2 (Polynomial)** | 9 | 0.9288 | **0.9533** | **1.2011** | **0.9043** | **Optimal Sweet Spot** |
| **Degree 3 (Polynomial)** | 19 | 0.9366 | 0.9414 | 1.3462 | 0.8379 | Moderate Complexity |
| **Degree 4 (Polynomial)** | 34 | 0.9537 | 0.6517 | 3.2805 | 0.8939 | Overfitting (High Variance) |
