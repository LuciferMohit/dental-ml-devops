import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import SimpleITK as sitk
import plotly.graph_objects as go
import tempfile
import zipfile
import os
import shutil
import mlflow
import time  # Added to track how fast your AI is!

# --- MLFLOW CONFIGURATION ---
# 1. Point Python to your live AWS server
mlflow.set_tracking_uri("http://13.200.239.160:5000")

# 2. Name your experiment (this will create a new folder in the UI)
mlflow.set_experiment("Dental_Implant_Stability")

# --- MODEL CONFIGURATION ---
MODEL_PATH = "implant_model.onnx"

# Use this EXACT list in your ONNX script (Includes your brilliant 6th month duplicate fix)
MODEL_COLUMNS = [
    'Gender', 'Smoking', 'Alcohol', 'Diabetes', 'Hypertension',
    'FOV_full jaw', 'FOV_lower jaw', 'FOV_lower jaw only', 'FOV_narrowFOV',
    'FOV_partial jaw – right side', 'FOV_partial mandible (posterior)', 'FOV_posterior',
    'FOV_upper jaw', 'FOV_upper jaw only',
    'years_placed_ 1 YEAR ', 'years_placed_ 1.5 YEARS ', 'years_placed_ 10 MONTHS ',
    'years_placed_ 11 MONTHS ', 'years_placed_ 3 MONTHS ', 'years_placed_ 4 MONTHS ',
    'years_placed_ 5 MONTHS ', 'years_placed_ 6 MONTHS ', 'years_placed_ 6 MONTHS ',
    'years_placed_ 7 MONTHS ', 'years_placed_ 8 MONTHS ', 'years_placed_ 9 MONTHS ',
    'years_placed_1 YEAR',
    'bone density Misch_D3'
]


# --- 1. DICOM HELPER FUNCTIONS ---
def process_dicom_zip(zip_file):
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(temp_dir)

        if not dicom_names:
            for root, dirs, files in os.walk(temp_dir):
                dicom_names = reader.GetGDCMSeriesFileNames(root)
                if dicom_names: break

        if not dicom_names:
            return None, "No DICOM series found."

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image, None

    except Exception as e:
        return None, str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def create_3d_plot(sitk_image):
    size = sitk_image.GetSize()
    shrink_factor = [max(1, s // 64) for s in size]

    small_img = sitk.Shrink(sitk_image, shrink_factor)
    vol = sitk.GetArrayFromImage(small_img)

    vol = vol.transpose(2, 1, 0)

    X, Y, Z = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol.flatten(),
        isomin=300,
        isomax=2500,
        opacity=0.1,
        surface_count=20,
        colorscale='Gray',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=400
    )
    return fig


# --- 2. STREAMLIT UI ---
st.set_page_config(page_title="Implant Survival AI", layout="wide")

st.title("🦷 Dental Implant Prognosis AI")
st.markdown("### Interactive 3D Visualization & Prediction")

col_img, col_data = st.columns([1.2, 0.8])

with col_img:
    st.header("1. 3D Scan Visualization")
    uploaded_file = st.file_uploader("Upload DICOM Series (.zip)", type="zip")

    if uploaded_file is not None:
        with st.spinner("Reconstructing 3D Volume..."):
            img_3d, error = process_dicom_zip(uploaded_file)

            if error:
                st.error(f"Error: {error}")
            else:
                fig = create_3d_plot(img_3d)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Tip: Click and drag to rotate the jaw. Scroll to zoom.")
    else:
        st.info("Upload a .zip file containing the DICOM folder.")

with col_data:
    st.header("2. Patient Data")

    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            gender = st.selectbox("Gender", ["Male", "Female"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            smoking = st.selectbox("Smoker", ["No", "Yes"])
        with col_b:
            alcohol = st.selectbox("Alcohol", ["No", "Yes"])
            hypertension = st.selectbox("BP/HTN", ["No", "Yes"])

        fov = st.selectbox("FOV", ["Posterior", "Anterior", "Full Jaw", "Lower Jaw", "Upper Jaw", "Narrow"])
        bone_density = st.selectbox("Bone Density", ["D1 (Dense)", "D2", "D3 (Porous)", "D4"])
        time_placed = st.selectbox("Time Placed", ["3 Months", "6 Months", "9 Months", "1 Year", "1.5 Years"])

        submitted = st.form_submit_button("🚀 Predict Survival")

# --- 3. PREDICTION LOGIC & MLOPS TRACKING ---
if submitted:
    # Start tracking this specific prediction event in MLflow!
    with mlflow.start_run():

        # 1. Log the EXACT parameters the user selected in the UI
        mlflow.log_params({
            "Gender": gender,
            "Diabetes": diabetes,
            "Smoking": smoking,
            "Alcohol": alcohol,
            "Hypertension": hypertension,
            "FOV": fov,
            "Bone_Density": bone_density,
            "Time_Placed": time_placed
        })

        input_data = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

        input_data['Gender'] = 1 if gender == "Male" else 0
        input_data['Diabetes'] = 1 if diabetes == "Yes" else 0
        input_data['Smoking'] = 1 if smoking == "Yes" else 0
        input_data['Alcohol'] = 1 if alcohol == "Yes" else 0
        input_data['Hypertension'] = 1 if hypertension == "Yes" else 0

        if fov == "Full Jaw":
            input_data['FOV_full jaw'] = 1
        elif fov == "Lower Jaw":
            input_data['FOV_lower jaw'] = 1
        elif fov == "Upper Jaw":
            input_data['FOV_upper jaw'] = 1
        elif fov == "Posterior":
            input_data['FOV_posterior'] = 1
        elif fov == "Narrow":
            input_data['FOV_narrowFOV'] = 1

        if "D3" in bone_density: input_data['bone density Misch_D3'] = 1

        if time_placed == "3 Months":
            input_data['years_placed_ 3 MONTHS '] = 1
        elif time_placed == "6 Months":
            input_data['years_placed_ 6 MONTHS '] = 1
        elif time_placed == "9 Months":
            input_data['years_placed_ 9 MONTHS '] = 1
        elif time_placed == "1 Year":
            input_data['years_placed_1 YEAR'] = 1
        elif time_placed == "1.5 Years":
            input_data['years_placed_ 1.5 YEARS '] = 1

        try:
            # Start a timer to see how fast your ONNX model is
            start_time = time.time()

            input_array = input_data.values.astype(np.float32)
            session = ort.InferenceSession(MODEL_PATH)
            input_name = session.get_inputs()[0].name
            prediction_output = session.run(None, {input_name: input_array})
            prediction = float(prediction_output[0][0][0])

            # Stop the timer
            inference_time_ms = (time.time() - start_time) * 1000

            # 2. Log the Final Results to MLflow
            mlflow.log_metric("predicted_survival_years", prediction)
            mlflow.log_metric("inference_time_ms", inference_time_ms)

            st.divider()
            st.markdown(f"### Predicted Survival: **{prediction:.2f} Years**")

            if prediction > 10:
                st.success("Excellent Prognosis (> 10 Years)")
            elif prediction > 7:
                st.warning("Moderate Prognosis (7-10 Years)")
            else:
                st.error("High Risk (< 7 Years)")

        except Exception as e:
            mlflow.log_param("error", str(e))
            st.error(f"Error running ONNX model: {str(e)}")