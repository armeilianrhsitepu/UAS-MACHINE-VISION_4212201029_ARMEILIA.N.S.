# -*- coding: utf-8 -*-
"""AAS_Machine Vision.ipynb

Armeilia Nurhasanah Sitepu (4212201029)
ASESMEN AKHIR SEMESTER

***Klasifikasi Karakter Tulisan Tangan pada Dataset EMNIST (Extended MNIST) menggunakan Convolutional Neural Network (CNN) dan Transfer Learning***

***IMPORT LIBRARY***
"""

import pandas as pd
import torch
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from tqdm import tqdm

"""***Load Dataset***

"""

train_data_path = r'C:\Users\user\Documents\SEMESTER5\VisionMachine\emnist-bymerge-train.csv'
val_data_path = r'C:\Users\user\Documents\SEMESTER5\VisionMachine\emnist-bymerge-test.csv'

"""***Load a subset of 1000 samples for training and validation for faster processing***

"""

data_train = pd.read_csv(train_data_path, header=None, nrows=1000)
data_val = pd.read_csv(val_data_path, header=None, nrows=1000)
print("Dataset loaded successfully.")

"""***Preprocessing function for raw pixel Data***

"""

def preprocess_image(data):

    data = np.clip(data, 0, 255).astype(np.uint8).reshape(28, 28)
    return Image.fromarray(data).convert("RGB")

"""***Custom dataset class to handle EMNIST Data***

"""

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Initialize the dataset with a DataFrame and optional image transformations.
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves the image and label for a given index, applies preprocessing and transformations.
        """
        label = self.dataframe.iloc[idx, 0]
        img_data = self.dataframe.iloc[idx, 1:].values
        image = preprocess_image(img_data)
        if self.transform:
            image = self.transform(image)
        return image, label

"""***Transformations for input data to be compatible with AlexNet (224x224, tensor format)***

"""

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create datasets and data loaders for training and validation

train_dataset = CustomDataset(data_train, transform=transform)
val_dataset = CustomDataset(data_val, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Training data loader
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize a pretrained AlexNet model for transfer learning

model = models.alexnet(pretrained=True)
model.classifier[6] = nn.Linear(4096, 47)

# Freeze feature extraction layers to only train the classifier

for param in model.features.parameters():
    param.requires_grad = False

# Define loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Configure the device for GPU acceleration (if available)

device = torch.device('cpu')
model.to(device)

# Training Loop

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

# Convert the training data into a NumPy array for LOOCV

data_array = data_train.to_numpy()

# Initialize lists to store predictions and labels for evaluation

all_preds, all_labels = [], []

print("Starting LOOCV...")

# Leave-One-Out Cross Validation implementation

loo = LeaveOneOut()
for train_idx, test_idx in tqdm(loo.split(data_array)):

    # Split data into training and test sets for this fold
    train_samples = data_array[train_idx]
    test_sample = data_array[test_idx]

    # Create datasets and dataloaders for the current LOOCV split
    train_dataset = CustomDataset(pd.DataFrame(train_samples), transform=transform)
    test_dataset = CustomDataset(pd.DataFrame(test_sample), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Reinitialize the model and optimizer for each LOOCV iteration
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 200)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop for the current fold
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


    # Validation loop for the current fold
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.append(torch.argmax(outputs, dim=1).cpu().item())
            all_labels.append(labels.cpu().item())

# Calculate evaluation metrics***"""

conf_matrix = confusion_matrix(all_labels, all_preds)  # Confusion matrix
accuracy = accuracy_score(all_labels, all_preds)  # Accuracy score
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)  # Precision score
f1 = f1_score(all_labels, all_preds, average='macro')  # F1 score

# Display evaluation results

print("\nEvaluation Results:")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
