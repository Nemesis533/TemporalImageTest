from custom_net import CustomAutoencoder
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

val_source_images = "./val_source"
val_inference_images = "./val_reconstructed"
# DataLoaders for batching
batch_size = 64

def return_dataloaders():


    # Transformation pipeline: Resize to 128x128 and normalize
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to 128x128
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Download the dataset
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Dataset split sizes (e.g., 80% train, 10% validation, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Randomly split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8, pin_memory=True)

    return train_loader,val_loader,test_loader 

def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #create dataloaders
    train_loader,val_loader,test_loader = return_dataloaders()

    # Initialize the model, loss, and optimizer
    model = CustomAutoencoder().to(device)
    criterion = nn.MSELoss()  # Reconstruction loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 20
    print("Started training and validation")
    for epoch in range(epochs):        
        train_loss = model.train_epoch( train_loader, optimizer, criterion, device, epoch, profile= False)

        print(f"Train Loss: {train_loss:.4f}")

        val_loss = model.validate_epoch(val_loader, criterion, device,epoch,val_source_images, val_inference_images)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Val Loss: {val_loss:.4f}")


if __name__ == '__main__':
    main()
