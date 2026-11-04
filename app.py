import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Asteroid ML Explorer", page_icon="🌠", layout="wide"
)

st.title("🌠 Asteroid Intelligence & Hazard Explorer")
st.markdown(
    "Interactive web application with clean UI for "
    "**PHA Classification**, **Diameter Regression**, and **Cluster Analysis**."
)


# ==========================================
# 2. LOAD SAVED MODELS & DATA
# ==========================================
@st.cache_resource
def load_artifacts():
  cls_pipe, reg_pipe, X_cls, X_reg = None, None, None, None

  try:
    cls_pipe = joblib.load("best_pha_model.pkl")
  except Exception as e:
    st.error(f"❌ Failed to load 'best_pha_model.pkl': {e}")

  try:
    reg_pipe = joblib.load("best_regression_model.pkl")
  except Exception as e:
    st.error(f"❌ Failed to load 'best_regression_model.pkl': {e}")

  try:
    X_cls = joblib.load("X_cls_selected.pkl")
  except Exception as e:
    st.warning(
        f"⚠️ Could not load 'X_cls_selected.pkl': {e}. Generating fallback"
        " features."
    )
    X_cls = pd.DataFrame(
        np.random.randn(20, 15), columns=[f"Cls_Feature_{i}" for i in range(15)]
    )

  try:
    X_reg = joblib.load("X_reg_selected.pkl")
  except Exception as e:
    st.warning(
        f"⚠️ Could not load 'X_reg_selected.pkl': {e}. Generating fallback"
        " features."
    )
    X_reg = pd.DataFrame(
        np.random.randn(20, 15), columns=[f"Reg_Feature_{i}" for i in range(15)]
    )

  return cls_pipe, reg_pipe, X_cls, X_reg


cls_pipeline, reg_pipeline, X_cls_sample, X_reg_sample = load_artifacts()

if cls_pipeline is None or reg_pipeline is None:
  st.error(
      "🚨 **Critical Error:** Model pipelines failed to load. Ensure model"
      " files are in the same folder as app.py."
  )
  st.stop()

# Extract underlying estimators safely
cls_model = (
    cls_pipeline.named_steps["classifier"]
    if hasattr(cls_pipeline, "named_steps")
    else cls_pipeline
)
reg_model = (
    reg_pipeline.named_steps["regressor"]
    if hasattr(reg_pipeline, "named_steps")
    else reg_model
)

# Format Feature Names
if isinstance(X_cls_sample, np.ndarray):
  cls_feature_names = [f"Feature_{i}" for i in range(X_cls_sample.shape[1])]
  X_cls_sample = pd.DataFrame(X_cls_sample, columns=cls_feature_names)
else:
  cls_feature_names = list(X_cls_sample.columns)

if isinstance(X_reg_sample, np.ndarray):
  reg_feature_names = [f"Feature_{i}" for i in range(X_reg_sample.shape[1])]
  X_reg_sample = pd.DataFrame(X_reg_sample, columns=reg_feature_names)
else:
  reg_feature_names = list(X_reg_sample.columns)

# ==========================================
# 3. TAB NAVIGATION & SECTION-SPECIFIC INPUTS
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["🚀 PHA Classification", "📏 Diameter Regression", "📊 Cluster Analysis"]
)

with tab1:
  st.subheader("Potentially Hazardous Asteroid (PHA) Classification")
  st.markdown(
      f"Adjust all **{len(cls_feature_names)} trained features** below:"
  )

  cls_inputs = {}
  # Using an expander or multi-column structure for clean UI when features are many
  with st.expander(
      "🎛️ Click to Open/Close All Classification Parameters", expanded=True
  ):
    cols = st.columns(2)
    for idx, feature in enumerate(cls_feature_names):
      default_val = float(X_cls_sample[feature].median())
      min_val = float(X_cls_sample[feature].min())
      max_val = float(X_cls_sample[feature].max())
      if min_val == max_val:
        min_val, max_val = min_val - 1.0, max_val + 1.0

      with cols[idx % 2]:
        cls_inputs[feature] = st.number_input(
            label=feature, value=default_val, format="%.4f", key=f"cls_{feature}"
        )

  input_cls_df = X_cls_sample.iloc[[0]].copy()
  for col, val in cls_inputs.items():
    input_cls_df[col] = val

  st.markdown("---")
  if st.button("Run Classification Prediction", type="primary"):
    try:
      cls_proba = cls_pipeline.predict_proba(input_cls_df)[0]
      cls_pred_raw = cls_pipeline.predict(input_cls_df)
      cls_pred = int(np.ravel(cls_pred_raw)[0])

      res_col1, res_col2 = st.columns(2)
      with res_col1:
        if cls_pred == 1:
          st.error("⚠️ **Prediction: Hazardous (PHA)**")
        else:
          st.success("✅ **Prediction: Non-Hazardous (Safe)**")
        st.metric(
            label="Hazard Probability", value=f"{float(cls_proba[1])*100:.2f}%"
        )

      with res_col2:
        st.markdown("### 🔍 SHAP Explanation (Classification)")
        explainer_cls = shap.TreeExplainer(cls_model)
        shap_vals_cls = explainer_cls(input_cls_df)

        ev_cls = explainer_cls.expected_value
        if len(shap_vals_cls.shape) == 3:
          base_val_cls = (
              ev_cls[1]
              if isinstance(ev_cls, (np.ndarray, list)) and len(ev_cls) > 1
              else ev_cls
          )
          explanation_obj = shap.Explanation(
              values=shap_vals_cls.values[0, :, 1],
              base_values=base_val_cls,
              data=input_cls_df.iloc[0],
              feature_names=cls_feature_names,
          )
        else:
          base_val_cls = (
              ev_cls[0] if isinstance(ev_cls, (np.ndarray, list)) else ev_cls
          )
          explanation_obj = shap.Explanation(
              values=shap_vals_cls.values[0],
              base_values=base_val_cls,
              data=input_cls_df.iloc[0],
              feature_names=cls_feature_names,
          )

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(explanation_obj, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
      st.error(f"Error during classification: {e}")

with tab2:
  st.subheader("Asteroid Diameter Prediction")
  st.markdown(f"Adjust all **{len(reg_feature_names)} trained features** below:")

  reg_inputs = {}
  with st.expander(
      "🎛️ Click to Open/Close All Regression Parameters", expanded=True
  ):
    cols = st.columns(2)
    for idx, feature in enumerate(reg_feature_names):
      default_val = float(X_reg_sample[feature].median())
      min_val = float(X_reg_sample[feature].min())
      max_val = float(X_reg_sample[feature].max())
      if min_val == max_val:
        min_val, max_val = min_val - 1.0, max_val + 1.0

      with cols[idx % 2]:
        reg_inputs[feature] = st.number_input(
            label=feature, value=default_val, format="%.4f", key=f"reg_{feature}"
        )

  input_reg_df = X_reg_sample.iloc[[0]].copy()
  for col, val in reg_inputs.items():
    input_reg_df[col] = val

  st.markdown("---")
  if st.button("Run Diameter Regression", type="primary"):
    try:
      reg_pred_raw = reg_pipeline.predict(input_reg_df)
      reg_pred = float(np.ravel(reg_pred_raw)[0])

      res_col1, res_col2 = st.columns(2)
      with res_col1:
        st.metric(label="Estimated Diameter", value=f"{reg_pred:.4f} km")
        st.info("Regression model evaluated with custom parameters.")

      with res_col2:
        st.markdown("### 🔍 SHAP Explanation (Regression)")
        explainer_reg = shap.TreeExplainer(reg_model)
        shap_vals_reg = explainer_reg(input_reg_df)

        ev_reg = explainer_reg.expected_value
        base_val_reg = (
            float(np.ravel(ev_reg)[0])
            if isinstance(ev_reg, (np.ndarray, list))
            else float(ev_reg)
        )

        reg_explanation_obj = shap.Explanation(
            values=np.ravel(shap_vals_reg.values[0]),
            base_values=base_val_reg,
            data=input_reg_df.iloc[0],
            feature_names=reg_feature_names,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(reg_explanation_obj, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
      st.error(f"Error during regression: {e}")

with tab3:
  st.subheader("Asteroid Clustering Analysis")
  st.markdown(
      "Visualizing how your input asteroid maps relative to the training"
      " clusters:"
  )

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.scatter(
      X_cls_sample.iloc[:, 0],
      X_cls_sample.iloc[:, 1],
      c="gray",
      alpha=0.4,
      label="Training Asteroids",
  )
  ax.scatter(
      input_cls_df.iloc[0, 0],
      input_cls_df.iloc[0, 1],
      c="red",
      s=200,
      marker="*",
      label="Your Input Asteroid",
  )
  ax.set_xlabel(cls_feature_names[0])
  ax.set_ylabel(
      cls_feature_names[1]
      if len(cls_feature_names) > 1
      else cls_feature_names[0]
  )
  ax.set_title("Input Asteroid Position in Feature Space")
  ax.legend()
  st.pyplot(fig)
  plt.clf()