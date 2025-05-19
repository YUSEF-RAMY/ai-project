import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained model
model = load_model("mobilenetAICropPrediction (1).h5")

# Class names
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# Disease descriptions
plant_disease_classes = {
    "Pepper__bell___Bacterial_spot": "Bell pepper leaf infected with bacterial spot disease, causing dark lesions.",
    "Pepper__bell___healthy": "Healthy bell pepper leaf with no visible disease symptoms.",
    "Potato___Early_blight": "Potato leaf infected with early blight, showing dark concentric spots.",
    "Potato___Late_blight": "Potato leaf infected with late blight, causing brown or black patches, often leading to leaf rot.",
    "Potato___healthy": "Healthy potato leaf with no disease symptoms.",
    "Tomato_Bacterial_spot": "Tomato leaf infected with bacterial spot, causing dark, water-soaked lesions.",
    "Tomato_Early_blight": "Tomato leaf infected with early blight, typically showing concentric ring patterns.",
    "Tomato_Late_blight": "Tomato leaf infected with late blight, leading to large, dark, and rapidly expanding spots.",
    "Tomato_Leaf_Mold": "Tomato leaf infected with leaf mold, often showing yellow patches on top and fuzzy mold underneath.",
    "Tomato_Septoria_leaf_spot": "Tomato leaf showing many small, circular spots caused by Septoria fungus.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato leaf damage due to spider mites, often with tiny yellow spots and webbing.",
    "Tomato__Target_Spot": "Tomato leaf infected with target spot disease, showing large brown spots with concentric rings.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato plant infected with yellow leaf curl virus, causing leaf yellowing and curling.",
    "Tomato__Tomato_mosaic_virus": "Tomato leaf showing mottled appearance due to mosaic virus infection.",
    "Tomato_healthy": "Healthy tomato leaf with no disease symptoms."
}

# Set up the page
st.set_page_config(
    page_title="🌱 Crop_Prediction",
    layout="wide",
    page_icon="🌱"
)

# Sidebar
st.sidebar.markdown(
    """
    <style>
        .sidebar-container {
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            color: #ffffff;
            background-color: #4CAF50;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        }
        .sidebar-container h2 {
            color: #ffffff;
            font-size: 24px;
        }
        .sidebar-container p {
            font-size: 16px;
            margin: 10px 0;
        }
        .footer {
            font-size: 12px;
            color: #cccccc;
            margin-top: 50px;
        }
    </style>

    <div class="sidebar-container">
        <h2>🌱 Crop Disease Detection</h2>
        <p>🔍 Crop Classification</p>
        <p>📊 3 Crop Types</p>
        <p>💾 Trained with CNN</p>
        <p>👩‍💻 By: <strong style="color:#ffffff;">Eng. Heba Allah</strong></p>
    </div>
    <div class="footer">2025 © All Rights Reserved</div>
    """,
    unsafe_allow_html=True
)

# Navigation
page = st.sidebar.radio("📑 Navigation", ["🏠 Home", "📸 Predict", "📊 Accuracy"])

# Home Page
if page == "🏠 Home":
    st.markdown("""
        <style>
            .custom-home {
                background-color: #f9f9f9;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
                font-family: 'Segoe UI', sans-serif;
            }
            .custom-home h1 {
                color: #4CAF50;
                font-size: 40px;
                margin-bottom: 10px;
            }
            .custom-home h3 {
                color: #4A4A4A;
                margin-bottom: 30px;
            }
            .highlight {
                background-color: #ffffff;
                padding: 20px;
                border-radius: 12px;
                box-shadow: 1px 1px 8px rgba(0,0,0,0.05);
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-home">', unsafe_allow_html=True)
    st.markdown('<h1>Disease Classifier for 3 Crops 🌱</h1>', unsafe_allow_html=True)
    st.markdown('<h3>Classification of leaf diseases using a CNN model</h3>', unsafe_allow_html=True)

    selected_class = st.selectbox("Select a class to view its description:", list(plant_disease_classes.keys()))

    st.markdown(f"""
        <div class="highlight">
            <h4>📌 {selected_class}</h4>
            <p>{plant_disease_classes[selected_class]}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
# Accuracy Page
if page == "📊 Accuracy":
    st.title("📊 Model Performance Metrics")

    # Replace these with actual values from your model
    accuracy = 0.956
    precision = 0.956
    recall = 0.956

    st.markdown("### ✅ **Evaluation Metrics**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy * 100:.2f} %")
    col2.metric("Precision", f"{precision * 100:.2f} %")
    col3.metric("Recall", f"{recall * 100:.2f} %")

    # Bar Chart for metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Score': [accuracy, precision, recall]
    })

    fig1, ax1 = plt.subplots()
    ax1.bar(metrics_df['Metric'], metrics_df['Score'], color='#4CAF50')
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Performance Overview")
    for i, v in enumerate(metrics_df['Score']):
        ax1.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', color='black')
    st.pyplot(fig1)

    st.markdown("### 🧩 Confusion Matrix")

    # Load real confusion matrix
    confusion_matrix = np.array([
        [4987, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [7, 2264, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1524, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 0, 0, 1483, 32, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 8, 222, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 223, 8, 0, 24, 0, 0, 24, 0, 0, 0],
        [0, 9, 0, 0, 0, 0, 3012, 826, 8, 54, 25, 70, 0, 8, 0],
        [0, 0, 40, 0, 0, 0, 75, 777, 15, 31, 7, 15, 0, 0, 0],
        [0, 0, 0, 0, 0, 8, 16, 16, 1603, 29, 8, 22, 0, 0, 0],
        [0, 0, 14, 8, 0, 0, 7, 7, 67, 624, 0, 24, 0, 0, 0],
        [0, 0, 0, 7, 0, 0, 15, 15, 162, 466, 91, 0, 0, 7, 0],
        [0, 0, 0, 0, 0, 21, 25, 8, 17, 25, 170, 895, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 955, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 559, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69, 54, 0, 0, 2343]
    ])

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.matshow(confusion_matrix, cmap='Greens')
    for (i, j), val in np.ndenumerate(confusion_matrix):
        ax2.text(j, i, f'{val}', ha='center', va='center', color='red', fontsize=8)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    st.pyplot(fig2)
# Predict Page
if page == "📸 Predict":
    st.title("🔍 Plant Disease Prediction")

    uploaded_file = st.file_uploader("📤 Upload a Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption='🖼️ Uploaded Leaf Image', use_column_width=True)

        # Preprocess the image
        img_resized = image_pil.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        final_result = class_names[predicted_class]

        # Show result
        st.success(f"🌿 **Predicted Class:** `{final_result}`")

        # Class Descriptions
        plant_disease_classes = {
            "Pepper__bell___Bacterial_spot": "Bell pepper leaf infected with bacterial spot disease, causing dark lesions.",
            "Pepper__bell___healthy": "Healthy bell pepper leaf with no visible disease symptoms.",
            "Potato___Early_blight": "Potato leaf infected with early blight, showing dark concentric spots.",
            "Potato___Late_blight": "Potato leaf infected with late blight, causing brown or black patches, often leading to leaf rot.",
            "Potato___healthy": "Healthy potato leaf with no disease symptoms.",
            "Tomato_Bacterial_spot": "Tomato leaf infected with bacterial spot, causing dark, water-soaked lesions.",
            "Tomato_Early_blight": "Tomato leaf infected with early blight, typically showing concentric ring patterns.",
            "Tomato_Late_blight": "Tomato leaf infected with late blight, leading to large, dark, and rapidly expanding spots.",
            "Tomato_Leaf_Mold": "Tomato leaf infected with leaf mold, often showing yellow patches on top and fuzzy mold underneath.",
            "Tomato_Septoria_leaf_spot": "Tomato leaf showing many small, circular spots caused by Septoria fungus.",
            "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato leaf damage due to spider mites, often with tiny yellow spots and webbing.",
            "Tomato__Target_Spot": "Tomato leaf infected with target spot disease, showing large brown spots with concentric rings.",
            "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato plant infected with yellow leaf curl virus, causing leaf yellowing and curling.",
            "Tomato__Tomato_mosaic_virus": "Tomato leaf showing mottled appearance due to mosaic virus infection.",
            "Tomato_healthy": "Healthy tomato leaf with no disease symptoms."
        }

        # Show description
        if final_result in plant_disease_classes:
            st.markdown(f"**📘 Description:** {plant_disease_classes[final_result]}")
        else:
            st.warning("⚠️ No description available for this class.")
