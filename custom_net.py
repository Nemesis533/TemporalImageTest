import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import helper_functions as hf



class CustomNeuron(nn.Module):
    def __init__(self, in_features, min_value, max_value):
        super(CustomNeuron, self).__init__()
        self.linear = nn.Linear(in_features, 1)  # Single neuron
        self.min_value = min_value
        self.max_value = max_value

    def range_based_activation(self, x, min_value, max_value):
        # Custom activation logic, now using min_value and max_value
        return torch.where((x >= min_value) & (x <= max_value), x, torch.zeros_like(x))

    def forward(self, inputs,min_value, max_value):
        # Apply linear transformation
        weighted_sum = self.linear(inputs)  # Output: (batch_size, 1)
        
        # Apply custom activation
        output = self.range_based_activation(self, weighted_sum , min_value, max_value)
        
        return output


# defines a custom neuron class where the basis is not a linear equation but a tanh representation itself, with 
class CustomNeuron(nn.Module):
    def __init__(self, in_features,out_features, min_value, max_value):
        super(CustomNeuron, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # Single neuron
        self.min_value = min_value
        self.max_value = max_value

    def range_based_activation(self, x, min_value, max_value):
        return torch.where((x >= min_value) & (x <= max_value), x, torch.zeros_like(x))

    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sum = self.linear(inputs)
        self.output = self.range_based_activation(self.weighted_sum, self.min_value, self.max_value)
        return self.output

class CustomLayer(nn.Module):
    def __init__(self, input_size, output_size, min_value, max_value):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # Linear layer with input_size to output_size
        self.custom_neuron = CustomNeuron(output_size, output_size, min_value, max_value)  # Neuron operates on output_size

    def forward(self, x):
        x = self.linear(x)  # Output size: (batch_size, output_size)
        x = self.custom_neuron(x)  # Custom neuron operates on the output of the linear layer
        return x


class CustomAutoencoder(nn.Module):
    def __init__(self):
        super(CustomAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)

        # Custom layers
        self.layer1 = CustomLayer(8 * 32 * 32, 2048, 0.2, 0.5)
        self.layer2 = CustomLayer(2048, 256, 0.2, 0.5)
        self.layer3 = CustomLayer(256, 128, 0.2, 0.5)
        self.layer4 = CustomLayer(128, 50, 0.2, 0.5)  # Bottleneck

        # Decoder
        self.fc4 = nn.Linear(50, 128)
        self.fc5 = nn.Linear(128, 256)
        self.fc6 = nn.Linear(256, 8 * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # New layer

    def forward(self, x):
        # Encoder
        x = self.conv1(x)  # (100, 16, 64, 64)
        x = self.conv2(x)  # (100, 8, 32, 32)
        x = x.view(x.size(0), -1)  # Flatten to (100, 8192)

        # Custom layers
        x = self.layer1(x)  # (100, 2048)
        x = self.layer2(x)  # (100, 256)
        x = self.layer3(x)  # (100, 128)
        x = self.layer4(x)  # (100, 50)

        # Decoder
        x = self.fc4(x)  # (100, 128)
        x = self.fc5(x)  # (100, 256)
        x = self.fc6(x)  # (100, 2048)
        x = x.view(x.size(0), 8, 16, 16)  # Reshape to (100, 8, 16, 16)
        x = self.deconv1(x)  # (100, 16, 32, 32)
        x = self.deconv2(x)  # (100, 1, 64, 64)
        x = self.deconv3(x)  # (100, 1, 128, 128)

        return x
    
    # Training loop with image saving
    def train_epoch(self, train_loader, optimizer, criterion, device, epoch):
        self.train()
        train_loss = 0

        for batch_idx, (inputs, _) in enumerate(train_loader):
            hf.progress_bar(iteration = batch_idx+1, total = len(train_loader), prefix = f'Training Batch (Epoch: {epoch}): {batch_idx}', suffix = ' Complete ', length = 50)
            inputs = inputs.to(device)

            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, inputs)  # Compare output to input

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    # Validation function with image saving
    def validate_epoch(self, val_loader, criterion, device, epoch, save_dir,save_dir2):
        self.eval()
        val_loss = 0


        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(val_loader):

                hf.progress_bar(iteration = batch_idx+1, total = len(val_loader), prefix = f'Validating Batch (Epoch: {epoch}): {batch_idx}', suffix = ' Complete ', length = 50)
                inputs = inputs.to(device)

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

                # Save images for the first batch in each epoch
                if batch_idx == 0:
                    hf.save_images(inputs, save_dir, f"val_input_{epoch}")
                    hf.save_images(outputs, save_dir2, f"val_output_{epoch}")
            return val_loss / len(val_loader)
    
    def test_model(self, test_loader, criterion, device):
        self.eval()
        test_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs, _ = batch
                inputs = inputs.to(device)

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()

        print(f"Test Loss: {test_loss / len(test_loader):.4f}")


