import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -----------------------------
# Disease Classes
# -----------------------------

classes = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# -----------------------------
# CNN Model
# -----------------------------

class PlantCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():

    model = PlantCNN(num_classes=15)

    model.load_state_dict(
        torch.load(
            "plantcnn.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

model = load_model()

# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌿 Plant Disease Classifier")

st.write(
    "Upload a plant leaf image to identify the disease."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    st.success(
        f"Prediction: {classes[predicted.item()]}"
    )

    st.info(
        f"Confidence: {confidence.item()*100:.2f}%"
    )
