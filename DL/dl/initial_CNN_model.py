### Import Packages ###

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt

### Get data ###

train_data = torch.load("dl/data/tensordata/train_data_for_CNN.pt")  
test_data = torch.load("dl/data/tensordata/test_data_for_CNN.pt")
val_data = torch.load("dl/data/tensordata/val_data_for_CNN.pt")

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


initial_CNN_model = ImageClassification()
optimizer = optim.Adam(initial_CNN_model.parameters(), lr = 0.001)

### Train and validate model ###

def train_model(model, patience, n_epochs):
    train_losses = []
    train_accuracies = [] 
    validation_losses = []
    validation_accuracies = []
    
    early_stopping = EarlyStopping(patience = patience, verbose = True)
    model.train()    
     
    for epoch in range(1, n_epochs + 1):
        ####### train the model #######
        train_loss, correct_train, total_train = 0, 0, 0
        for batch, (images, labels) in train_data.items():
            optimizer.zero_grad()
            labels = labels.view(-1, 1).float()
            pred_labels = model(images)
            loss = F.binary_cross_entropy(pred_labels, labels)
            loss.backward()
            optimizer.step() 
            
            train_loss += loss.item() * images.size(0)
            predicted = (pred_labels >= 0.5).int()
            total_train += labels.size(0)
            correct_train += (predicted.flatten() == labels.flatten()).sum().item()
        
        train_loss /= len(train_data)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)    
            
        ####### validate the model #######
        validation_loss, correct_val, total_val = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for batch, (images, labels) in val_data.items():
                labels = labels.view(-1, 1).float()
                pred_labels = model(images)
                loss = F.binary_cross_entropy(pred_labels, labels)
                
                validation_loss += loss.item() * images.size(0)
                predicted = (pred_labels >= 0.5).int()
                total_val += labels.size(0)
                correct_val += (predicted.flatten() == labels.flatten()).sum().item()
                  
        validation_loss /= len(val_data)
        validation_accuracy = 100 * correct_val/ total_val
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        
        early_stopping(validation_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Print metrics
        print(f'Epoch [{epoch}/{n_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, \
            Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2f}%')
    
    return model, train_losses, validation_losses, train_accuracies, validation_accuracies

initial_CNN_model, train_losses, validation_losses, train_accuracies, validation_accuracies  = train_model(initial_CNN_model, 1, 15)

### Make graphs of loss and accuracy ###

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
# find position of lowest validation loss
minposs = validation_losses.index(min(validation_losses))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('dl/plots/val_and_train_loss.png')

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('dl/plots/val_and_train_accuracy.png')
 
### Save trained model ###

# Save the trained model
model_path = 'dl/trained_models/initial_CNN_model.pth'
torch.save(initial_CNN_model.state_dict(), model_path)
print(f"Model saved at {model_path}")        