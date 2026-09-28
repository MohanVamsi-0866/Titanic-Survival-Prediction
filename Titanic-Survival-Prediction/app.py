import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "titanic_best_model.pkl"
DATA_PATH = BASE_DIR / "data" / "Titanic-Dataset.csv"
BG_PATH = BASE_DIR / "images" / "titanic.svg"
RESULTS_PATH = BASE_DIR / "results" / "model_results.csv"

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)

def set_background():
    encoded = base64.b64encode(BG_PATH.read_bytes()).decode("utf-8")
    css = f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(rgba(3, 15, 28, 0.78), rgba(3, 15, 28, 0.88)),
            url("data:image/svg+xml;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .hero {{
        padding: 26px 30px;
        border-radius: 18px;
        background: rgba(5, 25, 45, 0.82);
        border: 1px solid rgba(255,255,255,.15);
        margin-bottom: 20px;
    }}
    .hero h1 {{ color: white; margin: 0; font-size: 42px; }}
    .hero p {{ color: #d8edf7; margin: 8px 0 0; font-size: 18px; }}
    .result {{
        padding: 24px;
        border-radius: 16px;
        background: rgba(255,255,255,.96);
        color: #102030;
        text-align: center;
        margin-top: 20px;
    }}
    .small-note {{ color: #d8edf7; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background()
model = load_model()
data = load_data()
results = load_results()

st.markdown(
    """
    <div class="hero">
        <h1>🚢 Titanic Survival Predictor</h1>
        <p>Machine Learning based passenger survival prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔮 Predictor", "📊 Model Performance", "📋 Dataset"])

with tab1:
    st.subheader("Passenger Information")
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
        sibsp = st.number_input("Siblings / Spouses", min_value=0, max_value=10, value=0, step=1)

    with col2:
        parch = st.number_input("Parents / Children", min_value=0, max_value=10, value=0, step=1)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0, step=0.01)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

    st.caption("The model uses Pclass, Sex, Age, SibSp, Parch, Fare and Embarked.")

    if st.button("🚢 Predict Survival", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            "Pclass": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Embarked": [embarked],
        })

        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        survival_probability = float(probabilities[1])
        death_probability = float(probabilities[0])

        st.markdown('<div class="result">', unsafe_allow_html=True)
        if prediction == 1:
            st.success("Passenger is predicted to SURVIVE.")
        else:
            st.error("Passenger is predicted NOT TO SURVIVE.")

        c1, c2 = st.columns(2)
        c1.metric("Survival Probability", f"{survival_probability:.2%}")
        c2.metric("Not-Survival Probability", f"{death_probability:.2%}")
        st.progress(survival_probability)
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.subheader("Machine Learning Model Comparison")
    display_results = results.copy()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    display_results[metric_cols] = display_results[metric_cols].round(4)
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    st.bar_chart(display_results.set_index("Model")[metric_cols])
    st.info(
        "The project instructions use F1 Score as the primary comparison metric, "
        "with Accuracy and ROC-AUC as supporting metrics."
    )

with tab3:
    st.subheader("Titanic Dataset")
    st.write(f"Dataset shape: **{data.shape[0]} rows × {data.shape[1]} columns**")
    st.dataframe(data.head(20), use_container_width=True)
    st.write("Missing values")
    st.dataframe(data.isnull().sum().rename("Missing Values"), use_container_width=True)

st.divider()
st.caption("Titanic Survival Prediction • Machine Learning + Streamlit")
