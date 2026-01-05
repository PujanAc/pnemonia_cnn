# Pneumonia Detection using CNN & ResNet50 🫁📊

A **Chest X-ray Pneumonia Detection System** built using **Deep Learning (CNN & ResNet50)** and deployed with **Django**.  
The application allows users to upload chest X-ray images, get predictions, visualize **Grad-CAM heatmaps**, and **compare model performance**.

---

## 🚀 Features

- Pneumonia detection from chest X-ray images
- Custom CNN model
- Transfer Learning using **ResNet50**
- Binary classification (Normal vs Pneumonia)
- Grad-CAM visualization for explainability
- Model comparison with confidence scores
- Django-based web interface

---

## 🧠 Models Used

### Custom CNN
- Convolution + Pooling layers
- Fully connected layers
- Sigmoid output

### ResNet50 (Transfer Learning)
- Pretrained on ImageNet
- GlobalAveragePooling2D
- Dense layers with Dropout & Batch Normalization
- Sigmoid output for binary classification

---

## 🛠️ Tech Stack

- Python 3.11
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Django
- SQLite
- HTML, CSS, Bootstrap

---

## 📂 Project Structure

chest_xray/
│
├── pneumonia_detection/
│ ├── detector/
│ ├── templates/
│ ├── static/
│ ├── media/
│ ├── db.sqlite3
│ └── manage.py
│
├── venv/ # ignored
├── .gitignore
└── README.md



---

## ⚙️ Installation & Setup

### Clone Repository
```bash
git clone https://github.com/PujanAc/pnemonia_cnn.git
cd pnemonia_cnn


Create Virtual Environment
python -m venv venv
venv\Scripts\activate

Run Migrations
python manage.py migrate

Start Server
python manage.py runserver