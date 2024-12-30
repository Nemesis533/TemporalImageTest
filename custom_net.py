import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import helper_functions as hf
import torch.profiler as profiler
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

class RangeTanhNeuron(nn.Module):
    def __init__(self, input_size):
        super(RangeTanhNeuron, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Single neuron
        self.low_cutoff = nn.Parameter(torch.rand(1)*10e3)  # Start of range
        self.high_cutoff = nn.Parameter(torch.rand(1)*10)  # End of range
        #self.steepness = nn.Parameter(torch.rand(1) * 10)  # Controls sharpness

    def forward(self, x):
        # Compute linear transformation
        linear_output = self.linear(x)  # Shape: (batch_size, 1)

        activation = torch.tanh(torch.tanh(self.low_cutoff*linear_output-self.high_cutoff))       
        
        return activation

class CustomLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # Linear layer with input_size to output_size
        # Define multiple neurons, each with its own parameters
        self.neurons = nn.ModuleList([RangeTanhNeuron(input_size) for _ in range(output_size)])

    def forward(self, x):
        # Apply the linear transformation
        linear_output = self.linear(x)  # Shape: [batch_size, output_size]
        
        # Apply all neurons to the input at once
        # Stack neuron outputs in a tensor: [batch_size, output_size]
        neuron_outputs = torch.stack([neuron(x) for neuron in self.neurons], dim=1)
        
        # Element-wise multiplication between the linear output and neuron outputs
        # Shape: [batch_size, output_size]
        multiplied_output = linear_output + neuron_outputs.squeeze(2)
        
        return multiplied_output  # Shape: [batch_size, output_size]

class CustomAutoencoder(nn.Module):
    def __init__(self):
        super(CustomAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(4, 2, kernel_size=3, stride=2, padding=1)
        
        self.dropout1 = nn.Dropout(0.2)  # Dropout after conv1
        self.dropout2 = nn.Dropout(0.2)  # Dropout after conv2

        # Custom layers
        # self.layer1 = nn.Linear(8 * 32 * 32, 512)
        self.layer2 = nn.Linear(2*8*8,256)
        self.layer3 = CustomLayer(256, 512) 
       # self.dropout3 = nn.Dropout(0.2)  # Dropout after custom layers
        self.layer4 = nn.Linear(512, 256)

        self.layerL= nn.Linear(256, 256)  # Bottleneck
      
        # Decoder
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 8 * 8 * 8)
       #self.fc6 = nn.Linear(512, 1024)
        #self.layer7 = CustomLayer(1024, 2048)  # Bottleneck
        #self.fc7 = nn.Linear(512, 8 * 32 * 32)

        # Dropout after fully connected layers
        self.dropout_fc4 = nn.Dropout(0.2)
        self.dropout_fc5 = nn.Dropout(0.1)
        self.dropout_fc6 = nn.Dropout(0.1)
        self.dropout_fc7 = nn.Dropout(0.1)

        self.deconv1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)


        self.model_name = "CustomAE"
        self.model_save_dir ="./Models"
        #self.writer = SummaryWriter(log_dir=f"runs/{self.model_name}")  # Initialize TensorBoard writer
        self.loss = nn.Parameter(torch.tensor(10e9), requires_grad=False)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))  # (100, 16, 64, 64)
        x = F.relu(self.conv2(x))  # (100, 8, 32, 32)
        x = F.relu(self.conv3(x))  # (100, 8, 32, 32)
        x = F.relu(self.conv4(x))  # (100, 8, 32, 32)
        x = x.view(x.size(0), -1)  # Flatten to (100, 8192)
        x = F.relu(self.layer2(x)) # (100, 50)
        x = F.relu(self.layer3(x)) # (100, 50)

        x = F.relu(self.layer4(x)) # (100, 50)

        x = F.leaky_relu(self.layerL(x))
        # Decoder
        x = F.leaky_relu(self.fc4(x))  # (100, 128)
        x = F.relu(self.fc5(x))  # (100, 128)
      
        x = x.view(x.size(0), 8, 8, 8)  # Reshape to (100, 8, 16, 16)

        x = self.deconv1(x)  # (100, 16, 32, 32)
        x = self.deconv2(x)  # (100, 1, 64, 64)
        x = self.deconv3(x)  # (100, 1, 64, 64)
        x = self.deconv4(x)  # (100, 1, 64, 64)

        return x
    
    def train_epoch(self, train_loader, optimizer, criterion, device, epoch, profile=False, grad_clip_value=1.0):
        self.train()
        train_loss = 0
        last_save_cycle = 0
        last_save_loss = 0
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000001)
        
        # If profiling is enabled
        if profile:
            from torch.utils.data import DataLoader
            self.profile_training_loop(train_loader, criterion, optimizer, device, num_batches=10)        
            return train_loss  # Skip normal training after profiling
        
        # Regular training loop
        for batch_idx, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)

            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, inputs)  # Compare output to input
            
            # Print progress
            hf.progress_bar(
                iteration=batch_idx + 1, 
                total=len(train_loader), 
                prefix=f'Training Batch (Epoch: {epoch}): cycle {batch_idx} - loss is {loss.item()}', 
                suffix=f' Complete - Last Saved: {last_save_cycle} with loss {last_save_loss:.4f}', 
                length=50
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping: clip gradients by norm to avoid exploding gradients
            #torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_value)
            
            optimizer.step()

            train_loss += loss.item()
            scheduler.step()
            
            # Save model if current loss is better than the previous best
            if loss.item() < self.loss.item():
                self.loss = nn.Parameter(loss.detach(), requires_grad=False)
                self.save_model()
                last_save_cycle = batch_idx
                last_save_loss = loss.item()
                # print(f'Model {self.model_name} saved successfully')

        return train_loss / len(train_loader)


    def validate_epoch(self, val_loader, criterion, device, epoch, save_dir, save_dir2):
        best_model = CustomAutoencoder()
        best_model.model_name = self.model_name
        best_model.load_model()
        best_model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(val_loader):
                hf.progress_bar(iteration=batch_idx+1, total=len(val_loader),
                                prefix=f'Validating Batch (Epoch: {epoch}): {batch_idx}', 
                                suffix=' Complete ', length=50)
                inputs = inputs.to(device)

                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

                # Save images and log to TensorBoard for the first batch in each epoch
                if batch_idx == 0:
                    hf.save_images(inputs, save_dir, f"val_input_{epoch}")
                    hf.save_images(outputs, save_dir2, f"val_output_{epoch}")

                    # Log images to TensorBoard
                    #self.writer.add_images(f"Inputs/Epoch_{epoch}", vutils.make_grid(inputs, normalize=True), epoch)
                    #self.writer.add_images(f"Outputs/Epoch_{epoch}", vutils.make_grid(outputs, normalize=True), epoch)

            # Log the average validation loss to TensorBoard
            avg_val_loss = val_loss / len(val_loader)
            #self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

            return avg_val_loss

    def close_writer(self):
        self.writer.close()
    
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



    def profile_training_loop(self, train_loader, criterion, optimizer, device, num_batches=10):
        # Create a profiler
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            on_trace_ready=tensorboard_trace_handler("./profiler_logs"),  # Save logs for TensorBoard
            record_shapes=True,  # Record tensor shapes
            with_stack=True      # Include stack traces
        ) as prof:
            self.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):

                hf.progress_bar(iteration=batch_idx+1, total=num_batches, 
                                prefix=f'Profiling {num_batches-1} batches: {batch_idx}', 
                                suffix=' Complete ', length=50)
                if batch_idx >= num_batches:
                    break  # Profile only a limited number of batches

                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = self(inputs)

                # Compute loss
                loss = criterion(outputs, inputs)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Ensure profiler results are properly flushed and summarized
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace("./profiler_logs/trace.json")  # Export a Chrome trace file


    def save_model(self):
        # Ensure the directory exists
        os.makedirs(f'{self.model_save_dir}{self.model_name}', exist_ok=True)

        # Now save the model
        torch.save(self.state_dict(), f'{self.model_save_dir}{self.model_name}/{self.model_name}.pth')

        #print(f'Model {self.model_name} saved successfully')

    def load_model(self):
        self.load_state_dict(torch.load(f'{self.model_save_dir}{self.model_name}/{self.model_name}.pth'))
        
        print(f'Model {self.model_name} loaded successfully - loss was {self.loss}')