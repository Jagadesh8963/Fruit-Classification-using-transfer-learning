# Fruit Image Classification using ResNet50

## Project Overview

This project is a **fruit image classifier** built using **PyTorch** and deployed as a Streamlit web application. It utilizes a **ResNet50** model trained on a dataset of fruit images to classify uploaded images into predefined categories.

## Dataset

The dataset contains images of different fruit categories:

- 🍏 **Apple**

- 🍌 **Banana**

- 🍇 **Grape**

- 🥭 **Mango**

- 🍓 **Strawberry**

Each image is processed and classified using a deep learning model.

## Model Details

- **Base Model**: ResNet50 (Pretrained on ImageNet)

- **Custom Layers**:

   - Fully connected (Linear) layer with 512 neurons and ReLU activation

   - Dropout (0.5) for regularization

   - Final classification layer with softmax activation

- **Training**: Model fine-tuned on fruit images

- **Inference**: Uses Softmax probabilities to determine class

## Web Application

A **Streamlit** web application is developed for real-time image classification.

### Features:

- ✅ Upload an image for classification
- ✅ Display the uploaded image
- ✅ Show predicted class and confidence score
- ✅ Visualize class probabilities using a bar chart
- ✅ Custom animations and styling for better user experience

## Installation

### Dependencies

Ensure you have **Python 3.12** installed. Then, install the required packages:
``` bash
pip install -r requirements.txt
```
## Running the Application

To run the Streamlit web app, execute:
``` bash
streamlit run app.py
```
## File Structure
``` bash
📂 Fruit-Classification
├── 📜 fruit_classifier.pth  # Trained model weights
├── 📜 app.py                # Streamlit web application
├── 📜 requirements.txt      # Dependencies
└── 📜 README.md             # Project documentation
```
## How the Application Works

1) The **ResNet50 model** is loaded from ``` bashfruit_classifier.pth.```

2) Uploaded images are **preprocessed**:

    - Converted to **RGB** (if grayscale)

    - Resized to **224x224** pixels

    - Normalized using ImageNet mean and standard deviation

    - Converted to tensor

3) The **model predicts** the fruit class and returns confidence scores.

4) The results are **displayed** in the Streamlit UI, including:

- 🎯 **Predicted class**

- 📊 **Confidence score**

- 📈 **Probability distribution (bar chart)**

## Example Output

When an image is uploaded, the app displays:

- **Predicted Class**: 🍎 Apple

- **Confidence**: 97.2%

- **Probability Chart**:

   - 🍏 Apple: 97.2%

   - 🍌 Banana: 1.5%

   - 🥭 Mango: 0.8%

   - 🍇 Grape: 0.3%

   - 🍓 Strawberry: 0.2%

## Future Improvements

- 🔹 Expand dataset with more fruit categories
- 🔹 Train a deeper model like EfficientNet
- 🔹 Implement real-time webcam classification
- 🔹 Deploy as a cloud-based API

## Contributing

Feel free to fork this repository and submit pull requests. Any contributions to improve the model or application are welcome!
