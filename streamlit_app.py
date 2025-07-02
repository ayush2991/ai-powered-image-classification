import streamlit as st
import os
import numpy as np
import cv2
from glob import glob
from PIL import Image
import tensorflow as tf

# --- Load model and class names ---
MODEL_PATH = "best_model.keras"
TEST_DIR = "split_dataset/test"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def get_class_names():
    return sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])

@st.cache_data
def get_sample_images(class_name, max_images=20):
    pattern = os.path.join(TEST_DIR, class_name, "*")
    return sorted(glob(pattern))[:max_images]

def preprocess_image(img_path, img_height=224, img_width=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, img_array, class_names):
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    confidence = float(preds[0][idx])
    return class_names[idx], confidence, preds[0]

def gradcam_heatmap(model, img_array, class_idx, layer_name=None):
    # Use the functional API to access the base model and its input/output
    base_model = model.layers[0]
    # Ensure base_model is built
    _ = base_model(np.zeros((1, 224, 224, 3), dtype=np.float32))
    if layer_name is None:
        conv_layers = [l for l in base_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No Conv2D layer found in the model for GradCAM.")
        layer_name = conv_layers[-1].name
    # Build a grad model using the base model's input, outputs conv layer and full model prediction
    # The prediction must be computed as model(sequential_input), not model(base_model.output)
    grad_model = tf.keras.models.Model(
        [model.input], [base_model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    output = conv_outputs[0]
    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = gate_f * gate_r * grads
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.zeros(output.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    cam = np.maximum(cam, 0)
    max_cam = np.max(cam)
    if max_cam != 0:
        cam = cam / max_cam
    cam = cv2.resize(cam, (224, 224))
    return cam

def show_probability_bar_chart(class_names, all_preds, top_k=10):
    import pandas as pd
    import plotly.express as px
    # Get top_k predictions
    idxs = np.argsort(all_preds)[::-1][:top_k]
    data = {
        "Class": [class_names[i] for i in idxs],
        "Probability": [all_preds[i] for i in idxs]
    }
    df = pd.DataFrame(data)
    fig = px.bar(
        df,
        x="Probability",
        y="Class",
        orientation="h",
        color="Probability",
        color_continuous_scale="Blues",
        range_x=[0, 1],
        labels={"Probability": "Probability", "Class": "Class"},
        title=f"Top {top_k} Class Probabilities"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# --- Streamlit UI ---
st.set_page_config(page_title="Caltech101 Image Classifier", layout="wide", page_icon="🖼️")

# Custom CSS for better visuals
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    .st-bb { background: #f8fafc; }
    .st-cq { background: #f8fafc; }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg, #3b82f6, #06b6d4);
    }
    .stSlider > div[data-baseweb="slider"] > div {
        background: #e0e7ef;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        border: none;
        border-radius: 6px;
        padding: 0.5em 1.5em;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🖼️ Caltech101 Image Classifier Demo")
st.markdown(
    """
    <div style="font-size:1.2em;">
    <ul>
      <li><b>Browse</b> sample images by class.</li>
      <li><b>See</b> the model's prediction and confidence.</li>
      <li><b>Visualize</b> model attention with <b>GradCAM</b>.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True
)

model = load_model()
class_names = get_class_names()

# Sidebar for class and image selection
st.sidebar.header("🔎 Explore Dataset")
selected_class = st.sidebar.selectbox("Select a class", class_names, index=15)
sample_images = get_sample_images(selected_class)
if not sample_images:
    st.sidebar.warning("No images found for this class.")
    st.stop()
img_idx = st.sidebar.slider("Select image", 0, len(sample_images)-1, 0)
img_path = sample_images[img_idx]

# Main layout
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Sample Image")
    st.image(img_path, caption=f"Class: {selected_class}", use_container_width=True)
    st.markdown(f"<div style='text-align:center; color: #64748b;'>Image {img_idx+1} of {len(sample_images)}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Model Prediction")
    img_array = preprocess_image(img_path)
    pred_class, confidence, all_preds = predict(model, img_array, class_names)
    st.markdown(
        f"<span style='font-size:1.3em;'>🔮 <b>Predicted:</b> <span style='color:#3b82f6'>{pred_class}</span></span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='font-size:1.1em;'>Confidence: <b>{confidence:.2%}</b></span>",
        unsafe_allow_html=True
    )
    st.progress(confidence)
    # Show top-5 predictions as a table
    top5_idx = np.argsort(all_preds)[::-1][:5]
    st.markdown("**Top-5 Predictions:**")
    top5_table = {
        "Class": [class_names[i] for i in top5_idx],
        "Confidence": [f"{all_preds[i]:.2%}" for i in top5_idx]
    }
    st.table(top5_table)

    # Probability bar chart instead of GradCAM
    with st.expander("🔬 Show Top Class Probabilities Chart"):
        st.plotly_chart(show_probability_bar_chart(class_names, all_preds, top_k=10), use_container_width=True)

# Gallery of images for quick browsing
st.markdown("---")
st.subheader("📷 Quick Gallery")
gallery_cols = st.columns(6)
for i, img in enumerate(sample_images[:6]):
    with gallery_cols[i]:
        st.image(img, use_container_width=True)
        if i == img_idx:
            st.markdown("<div style='text-align:center; color:#3b82f6; font-weight:bold;'>Selected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; color:#64748b;'>Sample</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Made with Streamlit · Powered by TensorFlow/Keras · Caltech101 Demo")
