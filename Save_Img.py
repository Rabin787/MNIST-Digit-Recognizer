# Saving the image by creating their respective folders 
import os
from PIL import Image
import numpy as np
import pandas as pd

#1. Load the dataset
df = pd.read_csv("digit-recognizer//train.csv")

#2. Create a directory to save images 
output_dir = 'digit_images'
os.makedirs(output_dir, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)


#3. Iterate through the DataFrame and save images
for idx, row in df.iterrows():
    label = int(row['label'])  
    pixels = row[1:].values.reshape(28, 28).astype(np.uint8)  # reshape to 28x28
    
    # Convert to image
    img = Image.fromarray(pixels, mode='L')  # 'L' for grayscale
    
    # Save in respective label folder
    img.save(os.path.join(output_dir, str(label), f'{idx}.jpg'))
    
    #  Print progress every 1000 images
    if idx % 1000 == 0:
        print(f'Saved {idx} images...')



print("All images saved successfully!")







