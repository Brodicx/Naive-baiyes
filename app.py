import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


@st.cache_data
def load_data(path: str = "diabetes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def train_model(df: pd.DataFrame, target_col: str = "Outcome"):
    feature_cols = [c for c in df.columns if c != target_col]
    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col])
    class_names = le_target.classes_

    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "class_names": class_names,
        "X_test": X_test,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "confusion_matrix": cm,
    }
    return artifacts


def plot_cm(cm, class_names):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Gaussian NB - Diabetes", layout="wide")
    st.title("Gaussian Naive Bayes - Diabetes")
    st.write("Demo Streamlit: preprocessing, training, evaluasi, dan simulasi prediksi data baru.")

    # Load & preview
    df = load_data()
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Distribusi fitur
    with st.expander("Distribusi Fitur (KDE)"):
        feature_cols = [c for c in df.columns if c != "Outcome"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        for ax, col in zip(axes, feature_cols):
            sns.kdeplot(df[col], fill=True, ax=ax)
            ax.set_title(col)
        plt.tight_layout()
        st.pyplot(fig)

    # Train model
    st.subheader("Training & Evaluasi")
    artifacts = train_model(df)
    st.write(f"Accuracy: **{artifacts['accuracy']:.3f}**")
    plot_cm(artifacts["confusion_matrix"], artifacts["class_names"])

    # Analisis probabilitas
    st.subheader("Analisis Probabilitas pada Sampel Uji")
    y_test_array = artifacts["y_test"].to_numpy()
    mis_idx = np.where(artifacts["y_pred"] != y_test_array)[0]
    if len(mis_idx) > 0:
        default_idx = int(mis_idx[0])
        note = "Sampel salah diprediksi"
    else:
        default_idx = 0
        note = "Tidak ada salah prediksi, gunakan sampel pertama"

    idx = st.slider("Pilih index sampel uji", 0, len(artifacts["X_test"]) - 1, value=default_idx)
    sample_proba = artifacts["model"].predict_proba(
        artifacts["X_test_scaled"][idx].reshape(1, -1)
    )[0]

    st.write(note)
    sample_df = pd.DataFrame({
        "Fitur": artifacts["feature_cols"],
        "Nilai_asli": artifacts["X_test"].iloc[idx].values,
        "Nilai_terstandar": artifacts["X_test_scaled"][idx]
    })
    st.dataframe(sample_df)
    proba_df = pd.DataFrame(sample_proba.reshape(1, -1), columns=artifacts["class_names"])
    st.write("Probabilitas:")
    st.dataframe(proba_df)
    pred_label = artifacts["class_names"][artifacts["model"].predict(
        artifacts["X_test_scaled"][idx].reshape(1, -1)
    )[0]]
    st.write(f"Prediksi kelas: **{pred_label}**")

    # Simulasi prediksi data baru
    st.subheader("Simulasi Prediksi Data Baru")
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    def num_input(col, label, value):
        return col.number_input(label, value=value, step=1.0)

    preg = num_input(c1, "Pregnancies", 2.0)
    glucose = num_input(c2, "Glucose", 120.0)
    bp = num_input(c3, "BloodPressure", 70.0)
    skin = num_input(c4, "SkinThickness", 20.0)
    insulin = num_input(c5, "Insulin", 80.0)
    bmi = num_input(c6, "BMI", 30.0)
    dpf = num_input(c7, "DiabetesPedigreeFunction", 0.5)
    age = num_input(c8, "Age", 33.0)

    if st.button("Prediksi"):
        new_df = pd.DataFrame([{
            "Pregnancies": preg,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])
        new_scaled = artifacts["scaler"].transform(new_df)
        new_pred = artifacts["model"].predict(new_scaled)[0]
        new_proba = artifacts["model"].predict_proba(new_scaled)[0]
        st.write(f"Hasil prediksi: **{artifacts['class_names'][new_pred]}**")
        st.dataframe(pd.DataFrame(new_proba.reshape(1, -1), columns=artifacts["class_names"]))

    st.caption("Gunakan dataset diabetes.csv di folder yang sama. Untuk deployment, jalankan `streamlit run app.py`.")



if __name__ == "__main__":
    main()

