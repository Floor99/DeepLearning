### Import Packages ###

import pandas as pd 
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms 
from sklearn.model_selection import train_test_split

####### Split the data #######

metadata_path = "dl/data/manipulated/selected_categories_data.parquet"
meta_data = pd.read_parquet(metadata_path)

X = meta_data.drop(columns = ['diseased'])
y = meta_data['diseased']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Split the training data into testing and validation sets (50% test, 50% validation)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

# Combine X and y
train_data = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
train_data.to_parquet("dl/data/metadata/train_data.parquet")
test_data = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
test_data.to_parquet("dl/data/metadata/test_data.parquet")
val_data = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
val_data.to_parquet("dl/data/metadata/val_data.parquet")
 
####### Make Class #######

class PlantDataset(Dataset):
    def __init__(self, metadata_path, transform=None, target_transform=None) -> None:
        self.metadata_path = metadata_path
        self.transform = transform
        self.target_transform = target_transform
        
        self.meta_data = pd.read_parquet(metadata_path)
            
    def __len__(self):
        return self.meta_data.shape[0]
    
    def __getitem__(self, index):
        image_path = self.meta_data.loc[index, "path"]
    
        image = Image.open(image_path)
        label = self.meta_data.loc[index, "diseased"
                                   ]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        label = torch.tensor(label, dtype=torch.int64)
        return image, label 

####### Normalize data #######

def get_mean_std(loader):
    mean, variance, total_images = 0.0, 0.0, 0.0

    for images, _ in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        # Update total_images
        total_images += images.size(0)
        # Compute mean and variance here
        mean += images.float().mean(2).sum(0) 
        variance += images.float().var(2).sum(0)

    # Final mean and variance
    mean /= total_images
    variance /= total_images

    return mean, variance

train_data = PlantDataset("dl/data/metadata/train_data.parquet", transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size = 2, shuffle = True) 

mean, variance = get_mean_std(train_loader)
std = variance.sqrt()

statistics = {
    "mean": mean.tolist(),
    "std": std.tolist(),
    "variance": variance.tolist()
}
torch.save(statistics, "dl/data/stats/statistics.pt")

# Apply normalization and transformation to the dataset
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

####### Transform data #######

train_data = PlantDataset("dl/data/metadata/train_data.parquet", transform=transform)
test_data = PlantDataset("dl/data/metadata/test_data.parquet", transform=transform)
val_data = PlantDataset("dl/data/metadata/val_data.parquet", transform=transform)

train_loader = DataLoader(train_data, batch_size = 128, shuffle=True)
test_loader = DataLoader(test_data, batch_size = 128, shuffle=True)
val_loader = DataLoader(val_data, batch_size = 128, shuffle=True)

####### Save data #######

# Function to save batch-wise data
def save_data(loader, file_name):
    batch_data = {}
    for i, (images, labels) in enumerate(loader):
        batch_data[i] = (images, labels)
    torch.save(batch_data, f"dl/data/tensordata/{file_name}.pt")

# Save concatenated training data
save_data(train_loader, "train_data_for_CNN")

# Save concatenated test data
save_data(test_loader, "test_data_for_CNN")

# Save concatenated validation data
save_data(val_loader, "val_data_for_CNN")

 