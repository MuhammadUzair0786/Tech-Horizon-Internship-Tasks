# 🧠 AI & Deep Learning Projects – TechHorizon Internship

This repository contains three deep learning projects completed during the **TechHorizon Internship**.  
The projects cover **digit classification**, **image classification with a pre-trained model**, and **a Streamlit-based digit recognition web app**.

📌 **GitHub Repository Link:** [Tech-Horizon-Internship-Tasks](https://github.com/MuhammadUzair0786/Tech-Horizon-Internship-Tasks)

---

## 📂 Project Files

1. **Task1 – Classify Handwritten Digits** (`Task1-Classify Handwritten Digits_TechHorizon.ipynb`)  
   - A neural network trained on the **MNIST dataset** to classify handwritten digits (0–9).
   - Built from scratch using **TensorFlow/Keras**.
   - Includes data preprocessing, model building, training, evaluation, and visualization.

2. **Task2 – Image Classification with MobileNetV2** (`Task2_Image_Classification_MobileNetV2.ipynb`)  
   - Uses **transfer learning** with **MobileNetV2** for image classification.
   - Demonstrates loading a pre-trained model, fine-tuning, and making predictions.
   - Includes image preprocessing, model performance evaluation, and test predictions.

3. **Digit Recognition Web App** (`app.py`)  
   - **Streamlit** web application for real-time handwritten digit recognition.
   - Allows users to draw digits on a canvas and get instant predictions.
   - Uses a trained digit classification model (`Final_Digit_Classify_model.h5`).
   - Displays prediction confidence and probability graph.

---

## 🚀 Features

### **Task 1 – Handwritten Digit Classifier**
- Dataset: **MNIST**
- Model: Custom-built Neural Network
- Input shape: `28x28` grayscale images
- Activation: `ReLU`, Output: `Softmax`
- Loss function: `Categorical Crossentropy`
- Optimizer: `Adam`
- Achieves high accuracy on test data

### **Task 2 – MobileNetV2 Image Classifier**
- Dataset: Custom / Example Image Dataset
- Model: Pre-trained **MobileNetV2** (ImageNet weights)
- Fine-tuned for a new classification task
- Preprocessing using **Keras ImageDataGenerator**
- Achieves efficient inference with minimal resources

### **Web App – Streamlit Digit Recognition**
- Draw any digit from **0 to 9**
- Processes drawing into model-readable format
- Predicts digit with confidence score
- Displays probability distribution graph
- Fully styled with custom **CSS** for a better UI

---

## 📦 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MuhammadUzair0786/Tech-Horizon-Internship-Tasks.git
cd Tech-Horizon-Internship-Tasks
