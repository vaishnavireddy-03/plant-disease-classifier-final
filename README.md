
# Plant Disease Classification Using CNN

## Project Overview

This project develops a Convolutional Neural Network (CNN) to classify plant leaf diseases using images from the PlantVillage dataset.

The model helps identify diseases in crops such as tomato, potato, and bell pepper from leaf images, enabling faster diagnosis and supporting precision agriculture.

---

## Objective

To build, train, evaluate, and deploy a CNN-based image classification model capable of detecting multiple plant diseases from leaf images.

---

## Dataset

Dataset: PlantVillage Dataset

Number of Classes: 15

Classes:

- Pepper__bell___Bacterial_spot
- Pepper__bell___healthy
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy
- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite
- Tomato__Target_Spot
- Tomato__Tomato_YellowLeaf__Curl_Virus
- Tomato__Tomato_mosaic_virus
- Tomato_healthy

---

## Model Architecture

The CNN consists of:

- Conv2D (3 → 16)
- ReLU
- MaxPool2D

- Conv2D (16 → 32)
- ReLU
- MaxPool2D

- Conv2D (32 → 64)
- ReLU
- MaxPool2D

- Fully Connected Layer (12544 → 128)
- ReLU

- Output Layer (128 → 15)

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 20 |
| Loss Function | CrossEntropyLoss |

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Project Structure

```text
plant-disease-classifier/
│
├── app.py
├── plantcnn.pth
├── requirements.txt
├── README.md
└── Plant_Disease_CNN.ipynb
```

## Deployment

The trained model is deployed using Streamlit.

Features:

- Upload a leaf image
- Automatic disease prediction
- Fast inference using PyTorch

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Future Improvements

- Increase model depth
- Apply transfer learning (ResNet, EfficientNet)
- Add confidence scores
- Support more crop species
- Improve accuracy using data augmentation

---

## Author

Vaishnavi Reddy

BITS Pilani Hyderabad Campus  
