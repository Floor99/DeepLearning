### Import Packages ###

import os
import pandas as pd 
from torchvision.io import read_image

### Get Data and Store in CSV ###

# Our selected categories
nine_categories = ["Apple", "Cherry", "Corn", "Grape", "Peach", "Pepper", "Potato", "Strawberry", "Tomato"]

# Source directory
src_directory = "dl/data/raw/color/"

# Initialize a list to store data dictionaries
data_list = []

# Get all folders in source directory and add them to one dataframe
for folder in os.listdir(src_directory):
    for category in nine_categories:
        # Check whether category (of nine categories) in folder name
        if category in folder:
            category_name = category.split("__")[0]
            # Check if healthy in folder name
            # If yes, add this files to healthy_category folder
            if 'healthy' in folder:
                files = [os.path.join(src_directory, folder, file) for file in os.listdir(os.path.join(src_directory, folder))]
                data_list.extend([{'path': file, 'species': category_name, 'diseased': 0} for file in files])
            # If no, add this files to diseased_category folder 
            else:
                files = [os.path.join(src_directory, folder, file) for file in os.listdir(os.path.join(src_directory, folder))]
                data_list.extend([{'path': file, 'species': category_name, 'diseased': 1} for file in files])
  
# Function to drop incorrect images               
def drop_incorrect_images(df, expected_img_shape):
    for row in range(len(df)):
        image = read_image(df.loc[row, "path"])
        if tuple(image.shape) != tuple(expected_img_shape):
            df = df.drop(row)
            
    return df.reset_index(drop=True) 

# Function to remove images for specified species
def remove_images_for_species(df, species, count):
    species_rows = df[(df['diseased'] == 1) & (df['species'] == species)]
    rows_to_remove = species_rows.sample(n=int(count), random_state=42)
    df.drop(rows_to_remove.index, inplace=True)   
  
# Convert the list of dictionaries into a dataframe
df = pd.DataFrame(data_list)      

# Remove incorrect images    
df = drop_incorrect_images(df, [3, 256, 256])

# Remove the specified number of images for each species based on predefined amounts (see visualization.ipynb)
removals = {
    'Grape': 2925.307529,
    'Peach': 1846.504917,
    'Potato': 1607.753520,
    'Tomato': 13319.434034
}

for species, count in removals.items():
    remove_images_for_species(df, species, count)
    
# Reset index after removal
df = df.reset_index(drop=True)

# Save dataframe
dest_folder = f"dl/data/manipulated/selected_categories_data.parquet"
df.to_parquet(dest_folder)
    