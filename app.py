# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import io
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Vibration Fault Predictor", initial_sidebar_state="expanded")

# ---------------- Helper functions (feature extraction etc.) ----------------
DEFAULT_FS = 1000

def dominant_freqs_and_ratios(sig, fs=DEFAULT_FS, topk=3):
    spec = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), d=1.0/fs)
    if spec.size == 0:
        return [0.0]*topk, [0.0,0.0]
    idx_sorted = np.argsort(spec)
    top_idx = idx_sorted[-topk:] if idx_sorted.size >= topk else idx_sorted
    dom_freqs = list(freqs[top_idx])
    dom_amps = list(spec[top_idx])
    total = np.sum(spec) if np.sum(spec) > 0 else 1.0
    amp_sorted = np.array(dom_amps)
    if amp_sorted.size >= 2:
        two = np.sort(amp_sorted)[-2:]
        ratios = [float(two[-1]/total), float(two[-2]/total)]
    elif amp_sorted.size == 1:
        ratios = [float(amp_sorted[-1]/total), 0.0]
    else:
        ratios = [0.0,0.0]
    while len(dom_freqs) < topk:
        dom_freqs.append(0.0)
    return dom_freqs, ratios

def spectral_entropy(sig, fs=DEFAULT_FS):
    if len(sig) < 2:
        return 0.0
    f, Pxx = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    P = Pxx / np.sum(Pxx) if np.sum(Pxx) > 0 else Pxx
    P = P + 1e-12
    return -np.sum(P * np.log(P))

def spectral_centroid(sig, fs=DEFAULT_FS):
    if len(sig) < 2:
        return 0.0
    X = np.abs(rfft(sig))
    freqs = rfftfreq(len(sig), d=1.0/fs)
    denom = np.sum(X)
    return float(np.sum(freqs * X) / denom) if denom > 0 else 0.0

def extract_features(sig, fs=DEFAULT_FS):
    sig = np.asarray(sig)
    if sig.size == 0:
        return {}
    rms = float(np.sqrt(np.mean(sig**2)))
    peak = float(np.max(np.abs(sig)))
    crest = float(peak / rms) if rms > 0 else 0.0
    kurt = float(kurtosis(sig, fisher=False))
    sk = float(skew(sig))
    shape = float(rms / (np.mean(np.abs(sig)) + 1e-12))
    ent = float(spectral_entropy(sig, fs=fs))
    sc = float(spectral_centroid(sig, fs=fs))
    domf, ratios = dominant_freqs_and_ratios(sig, fs=fs)
    stdv = float(np.std(sig, ddof=0))
    return {
        "rms": rms, "peak": peak, "crest_factor": crest,
        "kurtosis": kurt, "skewness": sk, "shape_factor": shape,
        "spectral_entropy": ent, "spectral_centroid": sc,
        "dominant_freq_1": float(domf[0]), "dominant_freq_2": float(domf[1]), "dominant_freq_3": float(domf[2]),
        "amplitude_ratio_1": float(ratios[0]), "amplitude_ratio_2": float(ratios[1]),
        "std": stdv
    }

def window_and_extract(df, win_len=1024, step=1024, fs=DEFAULT_FS):
    X = []
    y = []
    n = len(df)
    i = 0
    while (i + win_len) <= n:
        w = df.iloc[i:i+win_len]
        x = w["x"].values
        ysig = w["y"].values
        z = w["z"].values
        # majority label
        label = None
        if "label" in df.columns:
            label = int(pd.Series(w["label"].values).mode()[0])
        fx = extract_features(x, fs=fs)
        fy = extract_features(ysig, fs=fs)
        fz = extract_features(z, fs=fs)
        order = ["rms", "peak", "crest_factor", "kurtosis", "skewness",
                 "shape_factor", "spectral_entropy", "spectral_centroid",
                 "dominant_freq_1", "dominant_freq_2", "dominant_freq_3",
                 "amplitude_ratio_1", "amplitude_ratio_2", "std"]
        combined = []
        for axis in (fx, fy, fz):
            for nm in order:
                combined.append(axis.get(nm, 0.0))
        X.append(combined)
        y.append(label if label is not None else -1)
        i += step
    feature_names = []
    axes = ["x","y","z"]
    for ax in axes:
        for nm in order:
            feature_names.append(f"{ax}_{nm}")
    return np.array(X), np.array(y), feature_names

# ---------------- UI layout ----------------
st.title("Vibration Fault Predictor — Streamlit App (FYP)")
st.write("Predicts: normal, imbalance, misalignment, bearing_fault")
st.sidebar.header("Options")

# Sidebar: Uploads & parameters
uploaded_csv = st.sidebar.file_uploader("Upload vibration CSV (time,x,y,z[,label])", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Or upload trained model (.pkl)", type=["pkl","joblib"])
sample_rate = st.sidebar.number_input("Sampling rate (Hz)", min_value=10, max_value=20000, value=1000, step=1)
window_len = st.sidebar.number_input("Window length (samples)", min_value=128, max_value=100000, value=1024, step=128)
window_step = st.sidebar.number_input("Window step (samples)", min_value=128, max_value=100000, value=1024, step=128)
train_model = st.sidebar.checkbox("Train model from uploaded labelled dataset (if provided)", value=False)
use_random_forest = st.sidebar.checkbox("Use RandomForest classifier (fast) if training", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("*Save model:* after training you can download the .pkl from the button shown.")

# Utility: read CSV into DataFrame
def read_vibration_csv(file_like):
    # Accept both normal and bytes buffer
    try:
        df = pd.read_csv(file_like, header=None)
        # Try to detect columns (if header present)
        if df.shape[1] >= 4 and not set(['time','x','y','z']).issubset(set(df.columns)):
            # assume columns 0..3 are time,x,y,z and optional label at 4
            df = df.rename(columns={0:"time",1:"x",2:"y",3:"z"})
            if df.shape[1] > 4:
                df = df.rename(columns={4:"label"})
        else:
            df = df.rename(columns={0:"time",1:"x",2:"y",3:"z"})
        # convert to numeric
        df[["time","x","y","z"]] = df[["time","x","y","z"]].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["time","x","y","z"]).reset_index(drop=True)
        
        # If 'label' exists, ensure it's numeric and drop NaN labels
        if 'label' in df.columns:
            df["label"] = df["label"].apply(pd.to_numeric, errors="coerce")
            df = df.dropna(subset=["label", "time", "x", "y", "z"]).reset_index(drop=True)

        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

# Load model if provided
model = None
scaler = None
if uploaded_model is not None:
    try:
        model_data = joblib.load(uploaded_model)
        # if user uploaded a dict with model and scaler
        if isinstance(model_data, dict) and "model" in model_data:
            model = model_data["model"]
            scaler = model_data.get("scaler", None)
            st.success("Loaded model + scaler from uploaded file.")
        else:
            model = model_data
            st.success("Loaded model from uploaded file.")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# If CSV uploaded, preview and extract features
df = None
if uploaded_csv is not None:
    df = read_vibration_csv(uploaded_csv)
    if df is not None:
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head(200))
        st.write(f"Samples: {len(df)}")
else:
    st.info("Upload vibration CSV file (time,x,y,z[,label]) to continue.")

# Training (if dataset includes labels and user checked train)
if df is not None and train_model:
    if "label" not in df.columns:
        st.warning("No 'label' column found in uploaded CSV. Training requires labelled windows (column 4 = label).")
    else:
        st.info("Building features and training model — this may take a short while.")
        X, y_arr, feat_names = window_and_extract(df, win_len=window_len, step=window_step, fs=sample_rate)
        # Remove windows with no label (label == -1)
        mask = y_arr != -1
        X = X[mask]; y_arr = y_arr[mask]
        if len(X) < 2:
            st.error("Not enough labeled windows to train. Need more data.")
        else:
            # scale and train
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            if use_random_forest:
                clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            else:
                clf = SVC(kernel='rbf', probability=True, random_state=42)
            clf.fit(Xs, y_arr)
            model = clf
            st.success("Model trained from uploaded dataset.")
            # show simple eval (train-test split)
            Xtr, Xte, ytr, yte = train_test_split(Xs, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
            preds = model.predict(Xte)
            report = classification_report(yte, preds, output_dict=True)
            st.subheader("Quick evaluation on hold-out 20%")
            st.write(pd.DataFrame(report).transpose())

            # Provide button to download model+scaler
            buffer = io.BytesIO()
            joblib.dump({"model": model, "scaler": scaler}, buffer)
            buffer.seek(0)
            st.download_button("Download trained model (.pkl)", buffer, file_name="vib_model.pkl", mime="application/octet-stream")

# Prediction & Visualization when a CSV is uploaded and model is available
if df is not None and model is not None:
    st.header("Predict & Visualize")
    st.write("Choose a window to analyze and predict. Default is the first window.")
    # choose window index
    max_windows = max(1, (len(df) - window_len) // window_step + 1)
    win_idx = st.number_input("Window index (0-based)", min_value=0, max_value=max_windows-1, value=0, step=1)
    start = win_idx * window_step
    end = start + window_len
    if end > len(df):
        st.error("Window exceeds available samples. Reduce window length or index.")
    else:
        window = df.iloc[start:end]
        # signals
        t = window["time"].values
        x = window["x"].values
        ysig = window["y"].values
        z = window["z"].values

        # show time-series (plotly)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=t, y=x, mode="lines", name="x"))
        fig_ts.add_trace(go.Scatter(x=t, y=ysig, mode="lines", name="y"))
        fig_ts.add_trace(go.Scatter(x=t, y=z, mode="lines", name="z"))
        fig_ts.update_layout(title="Time Series (Window)", xaxis_title="Time", yaxis_title="Amplitude", height=350)
        st.plotly_chart(fig_ts, use_container_width=True)

        # FFT magnitude for selected axis (user choice)
        axis_choice = st.selectbox("Axis for FFT / Spectrum", options=["x","y","z"], index=0)
        sig = {"x": x, "y": ysig, "z": z}[axis_choice]
        n = len(sig)
        fs = sample_rate
        fft_mag = np.abs(rfft(sig))
        freqs = rfftfreq(n, d=1.0/fs)

        # prepare FFT plot
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs, y=fft_mag, mode="lines", name="FFT magnitude"))
        # annotate top 3 peaks
        idx_top = np.argsort(fft_mag)[-5:]
        top_freqs = freqs[idx_top]
        top_amps = fft_mag[idx_top]
        for f_, a_ in zip(top_freqs, top_amps):
            if f_ > 0 and a_ > 0.01 * np.max(fft_mag): # Simple filter for non-zero/dominant peaks
                 fig_fft.add_annotation(x=f_, y=a_, text=f"{f_:.1f}Hz", showarrow=True, arrowhead=2, yshift=10, font=dict(size=10))
        fig_fft.update_layout(title=f"FFT Magnitude ({axis_choice} axis)", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", height=350)
        st.plotly_chart(fig_fft, use_container_width=True)

        # PSD via Welch
        f_w, Pxx = welch(sig, fs=fs, nperseg=min(256, n))
        fig_psd = go.Figure()
        fig_psd.add_trace(go.Scatter(x=f_w, y=Pxx, mode="lines", name="PSD (Welch)"))
        fig_psd.update_layout(title="Power Spectral Density (Welch)", xaxis_title="Frequency (Hz)", yaxis_title="Power", height=300)
        st.plotly_chart(fig_psd, use_container_width=True)

        # Extract features and predict
        fx = extract_features(x, fs=fs)
        fy = extract_features(ysig, fs=fs)
        fz = extract_features(z, fs=fs)
        order = ["rms", "peak", "crest_factor", "kurtosis", "skewness",
                 "shape_factor", "spectral_entropy", "spectral_centroid",
                 "dominant_freq_1", "dominant_freq_2", "dominant_freq_3",
                 "amplitude_ratio_1", "amplitude_ratio_2", "std"]
        combined = []
        for axis in (fx, fy, fz):
            for nm in order:
                combined.append(axis.get(nm, 0.0))
        Xvec = np.array(combined).reshape(1, -1)
        if scaler is not None:
            Xs = scaler.transform(Xvec)
        else:
            # Predict without scaling if scaler is missing, but warn (assumes model was trained without one or user accepts risk)
            Xs = Xvec

        pred_label = None
        probs = None
        try:
            pred_label = model.predict(Xs)[0]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(Xs)[0]
        except Exception as e:
            st.error(f"Prediction failed. Ensure the model was trained with the correct number of features, and the necessary scaler is included if used during training. Error: {e}")

        # Show features table
        st.subheader("Extracted features (single window)")
        feats_df = pd.DataFrame([combined], columns=[f"{ax}_{nm}" for ax in ["x","y","z"] for nm in order])
        st.dataframe(feats_df.T.rename(columns={0:"value"}))

        # Prediction display
        st.subheader("Prediction")
        if pred_label is not None:
            # label mapping (if you used numeric labels during training, ensure mapping)
            # This mapping is based on typical vibration fault datasets (0: Bearing, 1: Misalign, 2: Normal, 3: Imbalance)
            label_out = pred_label
            if isinstance(pred_label, (int, np.integer)):
                map_label = {0:"bearing_fault", 1:"misalignment", 2:"normal", 3:"imbalance"}
                label_out = map_label.get(int(pred_label), f"Unknown Label ({pred_label})")
            
            st.metric("Predicted Fault", f"**{label_out.upper()}**")
            
            if probs is not None:
                st.subheader("Class Probabilities")
                # Build probability table
                classes = list(model.classes_)
                # Convert numeric class labels to string labels if they are ints
                if all(isinstance(c, (int, np.integer)) for c in classes):
                    map_label = {0:"bearing_fault", 1:"misalignment", 2:"normal", 3:"imbalance"}
                    classes = [map_label.get(int(c), str(c)) for c in classes]
                    
                prob_df = pd.DataFrame({"Fault Class": classes, "Probability": probs})
                prob_df['Probability'] = (prob_df['Probability'] * 100).map('{:.2f}%'.format)
                st.dataframe(prob_df.sort_values("Probability", ascending=False).reset_index(drop=True))
        else:
            st.info("Model not available to predict — train or upload a model.")

# Footer / help
st.markdown("---")
st.markdown("*Notes & Usage*")
st.markdown("""
- **CSV Format:** Columns expected are `time,x,y,z`. If training, a 5th column `label` is required (integer labels are typical: e.g., 0, 1, 2, 3).
- **Workflow:**
    1. **Training:** Upload a labelled CSV, check **"Train model from uploaded labelled dataset"**, train the model, and **download the `.pkl` file** (which includes the necessary data scaler).
    2. **Prediction:** Upload your new, real-world data CSV (time,x,y,z) and upload the saved model `.pkl` file. The **Predicted Fault** will be displayed for the selected time window.
- The GUI will successfully show the predicted fault under the **Prediction** section when both a data CSV and a trained model are uploaded.
""")
