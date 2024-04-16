### Import Packages ###

import torch 
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

### Make initial model ###

class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # 256x256x3 -> 128x128x32 (halved because of maxpooling layer)
            nn.Conv2d(3, 32, kernel_size= (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  
            
            # 128x128x32 -> 64x64x64 (halved because of maxpooling layer)
            nn.Conv2d(32, 64, kernel_size= (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  
            
            # 64x64x64 -> 32x32x128 (halved because of maxpooling layer)
            nn.Conv2d(64, 128, kernel_size= (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),  
            
            nn.Flatten(),
            nn.Linear(131072, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid())
    
    def forward(self, xb):
        return self.network(xb)
    
### Load trained model ###

model = ImageClassification()
model.load_state_dict(torch.load("dl/trained_models/initial_CNN_model.pth"))
model.eval()

### Preprocess image ###

def preprocess_image(image_path, mean, std):
    # Load image
    image = Image.open(image_path)
    
    # Define the same transformations as used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std) 
    ])
    
    # Apply transformations
    image_tensor = transform(image).unsqueeze(0) 
    return image_tensor

### Make prediction ###

def predict(image_tensor):
    with torch.no_grad():  
        outputs = model(image_tensor)
        predicted = (outputs >= 0.5).int()
        return "sick" if predicted.item() == 1 else "not sick"

stats = torch.load("dl/data/stats/statistics.pt")
test_data = pd.read_parquet("dl/data/metadata/test_data.parquet")

### Get output ###

# Diseased image from test set
random_path_one = test_data[test_data['diseased'] == 1]['path'].sample(n=1).iloc[0]
image_tensor_one = preprocess_image(random_path_one, stats['mean'], stats['std']) 
prediction_one = predict(image_tensor_one)
print(f"The prediction for the image is: {prediction_one}")

image = Image.open(random_path_one)
plt.imshow(image)
plt.title(f"Prediction: {prediction_one}")
plt.show()

# Not diseased image from test set
random_path_two = test_data[test_data['diseased'] == 0]['path'].sample(n=1).iloc[0]
image_tensor_two = preprocess_image(random_path_two, stats['mean'], stats['std']) 
prediction_two = predict(image_tensor_two)
print(f"The prediction for the image is: {prediction_two}")

image = Image.open(random_path_two)
plt.imshow(image)
plt.title(f"Prediction: {prediction_two}")
plt.show()

