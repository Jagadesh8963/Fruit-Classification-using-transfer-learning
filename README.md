Fruit Classifier 🍎🍌🍇🥭🍓
This is a fruit classification web app built using Streamlit and PyTorch. The app allows users to upload an image of a fruit and classifies it into one of the predefined categories such as Apple, Banana, Grape, Mango, or Strawberry. The model is based on ResNet-50 and utilizes transfer learning to achieve high accuracy.

Features
Image Upload: Users can upload an image of a fruit (JPG, PNG, JPEG).
Real-time Classification: The app classifies the uploaded image and displays the predicted fruit class.
Confidence Score: Shows the confidence level of the classification.
Prediction Probabilities: Visualizes the probabilities of each class through a bar chart.
Custom Animations: The app has a sleek interface with animated transitions for the title, subtitle, and prediction results.
Technologies Used
Python: Main programming language.
Streamlit: Used for building the web interface.
PyTorch: Deep learning library used to load and predict using the pre-trained model.
ResNet-50: Convolutional Neural Network used for fruit image classification.
Matplotlib: Used to plot prediction probabilities.
Installation
To run the app locally, follow these steps:

Clone the repository:

git clone https://github.com/yourusername/fruit-classifier.git
cd fruit-classifier

Install the required dependencies:

pip install -r requirements.txt
Download the model:

Ensure you have the model file fruit_classifier.pth saved in the correct directory (as specified in the code).
Run the Streamlit app:


streamlit run app.py
Open the URL provided by Streamlit in your browser to use the app.

Model
The app uses a ResNet-50 model, fine-tuned on a dataset of fruit images to classify them into the following categories:

Apple
Banana
Grape
Mango
Strawberry
The model is saved as fruit_classifier.pth and is loaded into the app to make predictions on uploaded images.

File Structure

fruit-classifier/
│
├── app.py                  # Streamlit application script
├── fruit_classifier.pth    # Pre-trained model (download and place it in the directory)
├── requirements.txt        # Python dependencies
├── README.md               # Project overview and instructions
└── assets/
    └── logo.png            # Logo for the app (optional)
Usage
Upload an image of a fruit by clicking the "Choose an image..." button.
The app will process the image, classify it, and display:
The predicted fruit class.
The confidence level of the classification.
A bar chart of prediction probabilities for each fruit class.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Streamlit for providing an easy-to-use web framework.
PyTorch for the deep learning framework.
ResNet-50 for pre-trained models used in image classification.
