### Import Packages ###

import torch
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
    
model = ImageClassification()
model.load_state_dict(torch.load("dl/trained_models/initial_CNN_model.pth"))
model.eval()

correct = 0
total = 0
test_data = torch.load("dl/data/tensordata/test_data_for_CNN.pt")

for batch, (images, labels) in test_data.items():
    labels = labels.view(-1, 1).float()
    pred_labels = model(images)
    predicted = (pred_labels >= 0.5).int()
    total += labels.size(0)
    correct += (predicted.flatten() == labels.flatten()).sum().item()
    
accuracy = 100 * correct/ total
print(f"Accuracy on test set: {accuracy:.2f}%")