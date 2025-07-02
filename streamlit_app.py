import streamlit as st
import os
import numpy as np
from glob import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from train import AdvancedImageClassifierTorch

# --- Load model and class names ---
MODEL_PATH = "best_finetuned_model.pt"
TEST_DIR = "split_dataset/test"

@st.cache_resource
def load_model():
    class_names = get_class_names()
    num_classes = len(class_names)
    model = AdvancedImageClassifierTorch(num_classes=num_classes, base_model_name='mobilenet_v2')
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache_data
def get_class_names():
    return sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])

@st.cache_data
def get_sample_images(class_name, max_images=20):
    pattern = os.path.join(TEST_DIR, class_name, "*")
    return sorted(glob(pattern))[:max_images]

def preprocess_image(img_path, img_height=300, img_width=200):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img.unsqueeze(0)

def predict(model, img_array, class_names):
    with torch.no_grad():
        outputs = model(img_array)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        idx = np.argmax(probs)
        confidence = float(probs[idx])
    return class_names[idx], confidence, probs

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

st.title("🖼️ Caltech101 Image Classifier Demo")
st.markdown("""
- **Browse** sample images by class.
- **See** the model's prediction and confidence.
- **Visualize** model attention with GradCAM.
""")

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
    st.write(f"Image {img_idx+1} of {len(sample_images)}")

with col2:
    st.subheader("Model Prediction")
    img_array = preprocess_image(img_path)
    pred_class, confidence, all_preds = predict(model, img_array, class_names)
    st.write(f"🔮 **Predicted:** {pred_class}")
    st.write(f"Confidence: **{confidence:.2%}**")
    st.progress(confidence)
    # Show top-5 predictions as a table
    top5_idx = np.argsort(all_preds)[::-1][:5]
    st.write("**Top-5 Predictions:**")
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
            st.write("Selected")
        else:
            st.write("Sample")

st.markdown("---")
st.caption("Made with Streamlit · Powered by PyTorch · Caltech101 Demo")