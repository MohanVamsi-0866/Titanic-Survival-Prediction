# 🚢 Titanic Survival Prediction — Streamlit

A B.Tech student project that predicts whether a Titanic passenger survived using Machine Learning, with an optional TensorFlow/Keras Deep Learning section.

## Assignment deliverables

- Jupyter Notebook: `notebooks/Titanic_Analysis.ipynb`
- Dataset: `data/Titanic-Dataset.csv`
- Saved best ML pipeline: `models/titanic_best_model.pkl`
- Streamlit application: `app.py`
- Training script: `train_model.py`
- Model comparison: `results/model_results.csv`
- Requirements: `requirements.txt`
- Full notebook requirements: `requirements-full.txt`

## Dataset

The supplied dataset contains **891 rows and 12 columns**. The basic model uses:

`Pclass, Sex, Age, SibSp, Parch, Fare, Embarked`

Missing numerical values are handled with median imputation and categorical values with most-frequent imputation. Categorical features are one-hot encoded and numerical features are scaled.

## Model comparison from the supplied dataset

Using the required 80/20 stratified train-test split (`random_state=42`) and F1 Score as the primary comparison metric:

| Model               |   Accuracy |   Precision |   Recall |   F1 Score |   ROC-AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|----------:|
| Random Forest       |     0.8101 |      0.7966 |   0.6812 |     0.7344 |    0.8287 |
| Logistic Regression |     0.8045 |      0.7931 |   0.6667 |     0.7244 |    0.8437 |
| Gradient Boosting   |     0.7989 |      0.7895 |   0.6522 |     0.7143 |    0.8181 |
| XGBoost             |     0.7933 |      0.7667 |   0.6667 |     0.7132 |    0.8148 |
| Decision Tree       |     0.7654 |      0.7547 |   0.5797 |     0.6557 |    0.7971 |

The saved Streamlit model is **Random Forest**, selected from the actual experiment by F1 Score.

## Run locally

### 1. Create and activate a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Streamlit dependencies

```bash
pip install -r requirements.txt
```

### 3. Retrain the model if needed

```bash
python train_model.py
```

### 4. Start Streamlit

```bash
streamlit run app.py
```

The terminal will show a local URL. Open it in a browser.

## Streamlit Cloud deployment

1. Create a **public GitHub repository**.
2. Upload the complete contents of this project.
3. Open Streamlit Community Cloud.
4. Create a new app.
5. Select your GitHub repository.
6. Select `app.py` as the main file.
7. Deploy.

The repository must contain the model file because the app loads:

`models/titanic_best_model.pkl`

## Important

Do not put passwords, API keys, tokens, or other secrets in the repository.

The dashboard provides:
- Passenger class selection
- Gender selection
- Age input
- Siblings/spouses input
- Parents/children input
- Fare input
- Embarkation selection
- Prediction result
- Survival probability
- Non-survival probability
- Model comparison
- Dataset preview

## Deep Learning

The assignment brief specifies ANN and 1D CNN using TensorFlow/Keras. Their cells are included in the notebook and run when TensorFlow is available. `requirements-full.txt` contains TensorFlow for notebook/Colab use.

## Submission

Submit these two links after deployment:

1. **Public Streamlit website:** your deployed Streamlit URL
2. **GitHub repository:** your public GitHub repository URL
