import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dummy_results = {
    ('Random Forest', 'Kepler Object of Interest'): {
        'metrics': {'Precision': 0.834, 'Recall': 0.840, 'F1-Score': 0.835, 'AUC': 0.945},
        'conf_matrix': [[1351, 82, 19],
                        [ 180, 335, 79],
                        [  36,  61, 727]],
        'hyperparams': {'n_estimators': 200, 'max_depth': None, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    },
    ('LightGBM', 'Kepler Object of Interest'): {
        'metrics': {'Precision': 0.842, 'Recall': 0.846, 'F1-Score': 0.843, 'AUC': 0.951},
        'conf_matrix': [[1322, 110, 20],
                        [ 160,  361,  73],
                        [  15,   62, 747]],
        'hyperparams': {'n_estimators': 200, 'learning_rate': 0.1, 'is_unbalance': True, 'random_state': 42}
    },
    ('1D CNN', 'Kepler Object of Interest'): {
        'metrics': {'Precision': 0.729, 'Recall': 0.733, 'F1-Score': 0.706, 'AUC': 0.887},
        'conf_matrix': [[1199, 93, 160],
                        [ 151, 124, 319],
                        [  24,  17, 783]],
        'hyperparams': {'Conv1D Layers': [32, 64], 'Dense Layers': [64], 'Dropout Rates': [0.25, 0.4], 'Optimizer': 'Adam', 'Learning Rate': 0.0005, 'Epochs': 30, 'Batch Size': 256, 'Loss Function': 'categorical_crossentropy'}
    },
    ('Stacking Ensemble(Random Forest + LightGBM)', 'Kepler Object of Interest'): {
        'metrics': {'Precision': 0.747, 'Recall': 0.750, 'F1-Score': 0.840, 'AUC': 0.949},
        'conf_matrix': [[1247, 188, 17],
                        [ 103,  431,  60],
                        [  10,   97, 717]],
        'hyperparams': {'Final Estimator': 'Logistic Regression w/1000 iters and balanced class weights', 'Base Models': ['Random Forest', 'LightGBM'], 'Meta-Features': ['mean', 'std']}
    },
    ('Random Forest', 'TESS Object of Interest'): {
        'metrics': {'AUC': 0.838, 'Recall': 0.750, 'Precision': 0.747, 'F1-Score': 0.720},
        'conf_matrix': [[127, 256, 6],
                        [  42,  1455,  45],
                        [  6,  221, 153]],
        'hyperparams': {'n_estimators': 200, 'max_depth': None, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    },
    ('LightGBM', 'TESS Object of Interest'): {
        'metrics': {'AUC': 0.854, 'Recall': 0.769, 'Precision': 0.759, 'F1-Score': 0.756},
        'conf_matrix': [[ 174,  201,   14],
                        [  75, 1400,   67],
                        [  15,  160,  205]],
        'hyperparams': {'n_estimators': 200, 'learning_rate': 0.1, 'is_unbalance': True, 'random_state': 42}
    },
    ('1D CNN', 'TESS Object of Interest'): {
        'metrics': {'AUC': 0.805, 'Recall': 0.720, 'Precision': 0.704, 'F1-Score': 0.708},
        'conf_matrix': [[ 169,  203,   17],
                        [  99, 1329,  114],
                        [  27,  186,  167]],
        'hyperparams': {'Conv1D Layers': [32, 64], 'Dense Layers': [64], 'Dropout Rates': [0.25, 0.4], 'Optimizer': 'Adam', 'Learning Rate': 0.0005, 'Epochs': 30, 'Batch Size': 256, 'Loss Function': 'categorical_crossentropy'}
    },
    ('Stacking Ensemble(Random Forest + LightGBM)', 'TESS Object of Interest'): {
        'metrics': {'AUC': 0.849, 'Recall': 0.708, 'Precision': 0.749, 'F1-Score': 0.719},
        'conf_matrix': [[ 252,   98,   39],
                        [ 217, 1111,  214],
                        [  37,   69,  274]],
        'hyperparams': {'Final Estimator': 'Logistic Regression w/1000 iters and balanced class weights', 'Base Models': ['Random Forest', 'LightGBM'], 'Meta-Features': ['mean', 'std']}
    },
    ('Random Forest', 'K2 Dataset'): {
        'metrics': {'AUC': 0.969, 'Recall': 0.889, 'Precision': 0.887, 'F1-Score': 0.887},
        'conf_matrix': [[ 59,  24,   5],
                        [ 17, 336,  59],
                        [  4,  23, 668]],
        'hyperparams': {'n_estimators': 200, 'max_depth': None, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    },
    ('LightGBM', 'K2 Dataset'): {
        'metrics': {'AUC': 0.968, 'Recall': 0.887, 'Precision': 0.885, 'F1-Score': 0.885},
        'conf_matrix': [[ 57,  29,   2],
                        [ 21, 330,  61],
                        [  1,  20, 674]],
        'hyperparams': {'n_estimators': 200, 'learning_rate': 0.1, 'is_unbalance': True, 'random_state': 42}
    },
    ('1D CNN', 'K2 Dataset'): {
        'metrics': {'AUC': 0.865, 'Recall': 0.654, 'Precision': 0.756, 'F1-Score': 0.681},
        'conf_matrix': [[ 77,   7,   4],
                        [150, 189,  73],
                        [ 95,  84, 516]],
        'hyperparams': {'Conv1D Layers': [32, 64], 'Dense Layers': [64], 'Dropout Rates': [0.25, 0.4], 'Optimizer': 'Adam', 'Learning Rate': 0.0005, 'Epochs': 30, 'Batch Size': 256, 'Loss Function': 'categorical_crossentropy'}
    },
    ('Stacking Ensemble(Random Forest + LightGBM)', 'K2 Dataset'): {
        'metrics': {'AUC': 0.969, 'Recall': 0.900, 'Precision': 0.907, 'F1-Score': 0.902},
        'conf_matrix': [[ 77,  10,   1],
                        [ 39, 339,  34],
                        [  6,  29, 660]],
        'hyperparams': {'Final Estimator': 'Logistic Regression w/1000 iters and balanced class weights', 'Base Models': ['Random Forest', 'LightGBM'], 'Meta-Features': ['mean', 'std']}
    }
}

# --- Streamlit UI ---
st.set_page_config(page_title="Exoplanet Detection Models", layout="wide")
st.title("NASA Space Apps Challenge 2025 - Exoplanet Detection Models")

# Sidebar selections
st.sidebar.header("Selections")
selected_model = st.sidebar.selectbox(
    "Choose a Model",
    ["Random Forest", "LightGBM", "1D CNN", "Stacking Ensemble(Random Forest + LightGBM)"]
)
selected_dataset = st.sidebar.selectbox(
    "Choose a Dataset",
    ["Kepler Object of Interest", "TESS Object of Interest", "K2 Dataset"]
)

# Display based on selection
key = (selected_model, selected_dataset)
data = dummy_results.get(key)

if data:
    st.subheader(f"Results for **{selected_model}** on **{selected_dataset}**")

    # Create two columns: left for metrics + hyperparameters, right for confusion matrix
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        # --- Metrics ---
        st.markdown("### 📊 Evaluation Metrics")
        metrics_df = pd.DataFrame(data['metrics'], index=["Score"]).T
        st.dataframe(metrics_df.style.format("{:.2f}"))

        # --- Hyperparameters ---
        st.markdown("### ⚙️ Model Hyperparameters")
        st.json(data['hyperparams'])

    with right_col:
        # --- Confusion Matrix ---
        st.markdown("### 🧠 Confusion Matrix")
        conf_matrix = np.array(data['conf_matrix'])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)

else:
    st.warning("No data available for this combination.")