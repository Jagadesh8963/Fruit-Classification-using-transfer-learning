import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Path to the saved model
MODEL_PATH = 'c:/Users/jagad/OneDrive/Desktop/Fruitclasiif/Fruits Classification/fruit_classifier.pth'

# Class names
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence to display a class

# Add custom background and title style with animations
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        font-family: Arial, sans-serif;
    }
    .title {
        color: #FF6347;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        animation: fadeIn 2s ease-in-out;
    }
    .subtitle {
        color: #4682B4;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        animation: slideIn 1.5s ease-in-out;
    }
    .prediction {
        color: #32CD32;
        font-size: 22px;
        font-weight: bold;
        animation: scaleUp 1s ease-in-out;
    }
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
        }
        to {
            transform: translateX(0);
        }
    }
    @keyframes scaleUp {
        from {
            transform: scale(0.8);
        }
        to {
            transform: scale(1);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title
st.markdown('<div class="title">Fruit Classifier 🍎🍌🍇🥭🍓</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a fruit to classify it</div>', unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    # Define the model architecture
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(class_names))
    )
    # Load the state dictionary
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing pipeline
def preprocess_image(image):
    # Ensure the image has three channels (convert grayscale to RGB)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Load the model
    model = load_model()
    
    # Make prediction
    st.write("Classifying...")
    probabilities = predict(image, model)
    predicted_class_index = probabilities.argmax().item()
    predicted_class = class_names[predicted_class_index]
    confidence = probabilities[predicted_class_index].item()
    
    # Display prediction with animations
    st.markdown(f'<div class="prediction">Predicted Class: {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction">Confidence: {confidence:.4f}</div>', unsafe_allow_html=True)
    
    # Display probability chart
    probabilities_np = probabilities.cpu().numpy()
    fig, ax = plt.subplots()
    colors = ['#FF9999', '#FFD700', '#ADD8E6', '#90EE90', '#FFB6C1']  # Custom bar colors
    ax.bar(class_names, probabilities_np, color=colors)
    ax.set_xlabel("Classes", fontsize=14)
    ax.set_ylabel("Probabilities", fontsize=14)
    ax.set_title("Prediction Probabilities", fontsize=16)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, fontsize=12)
    st.pyplot(fig)
