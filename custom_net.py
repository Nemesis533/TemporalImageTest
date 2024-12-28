import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import helper_functions as hf
import torch.profiler as profiler
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
import os
from torch.optim.lr_scheduler import CosineAnnealingLR


class RangeTanhNeuron(nn.Module):
    def __init__(self, input_size):
        super(RangeTanhNeuron, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # Single neuron
        self.low_cutoff = nn.Parameter(torch.tensor(-1.0))  # Start of range
        self.high_cutoff = nn.Parameter(torch.tensor(1.0))  # End of range
        #self.steepness = nn.Parameter(torch.tensor(10.0))   # Controls sharpness

    def forward(self, x):
        # Smooth gating between the two tanh functions
        low_gate = torch.tanh(1 * (x - self.low_cutoff))
        high_gate = torch.tanh(1 * (self.high_cutoff - x))
        return 0.5 * (1 + low_gate) * 0.5 * (1 + high_gate)  # Smooth range mask

# # defines a custom neuron class where the basis is not a linear equation but a tanh representation itself, with 
# class CustomNeuron(nn.Module):
#     def __init__(self, in_features):
#         super(CustomNeuron, self).__init__()
#         self.linear = nn.Linear(in_features, 1)  # Single neuron
#         # Learnable parameters for each neuron's min_value and max_value
#         # Initialize min_value and max_value with random numbers in the range [0, 1]
#         self.min_value = nn.Parameter(torch.rand(1)*0.1, requires_grad=True)  # Random value between 0 and 1
#         self.max_value = nn.Parameter(data=torch.Tensor(1), requires_grad=True)


#     def range_based_activation(self, x, min_value, max_value):
#         return torch.where((x >= min_value) & (x <= max_value), torch.ones_like(x), torch.zeros_like(x))

#     def forward(self, inputs):
#         self.inputs = inputs
#         self.weighted_sum = self.linear(inputs)

#         # Apply clamping on the parameter's data to keep it within a valid range
#         #self.min_value.data = torch.clamp(self.min_value.data, min=0.0, max=1.0)  
#         #self.max_value.data = torch.clamp(self.max_value.data, min=0.0, max=1.0) 

#         self.output = self.range_based_activation(self.weighted_sum, self.min_value, self.max_value)
#         return self.output

class CustomLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # Linear layer with input_size to output_size
        # Define multiple neurons, each with its own parameters
        self.neurons = nn.ModuleList([RangeTanhNeuron(input_size) for _ in range(output_size)])

    def forward(self, x):
        # Apply each neuron to the input
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron(x))
        return torch.cat(outputs, dim=1)


class CustomAutoencoder(nn.Module):
    def __init__(self):
        super(CustomAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.1)  # Dropout after conv1
        self.dropout2 = nn.Dropout(0.1)  # Dropout after conv2

        # Custom layers
        self.layer1 = nn.Linear(8 * 32 * 32, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.1)  # Dropout after custom layers
        self.layer4 = CustomLayer(128, 64)  # Bottleneck
       

        # Decoder
        self.fc4 = nn.Linear(8192, 8192)
        #self.fc5 = nn.Linear(128, 256)
        #self.fc6 = nn.Linear(256, 512)
        #self.fc7 = nn.Linear(512, 8 * 32 * 32)

        # Dropout after fully connected layers
        self.dropout_fc4 = nn.Dropout(0.1)
        self.dropout_fc5 = nn.Dropout(0.1)
        self.dropout_fc6 = nn.Dropout(0.1)
        self.dropout_fc7 = nn.Dropout(0.1)


        self.deconv1 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)


        self.model_name = "CustomAE"
        self.model_save_dir ="./Models"
        self.loss = nn.Parameter(torch.tensor(10e9), requires_grad=False)

    def forward(self, x):
        # Encoder
        x = F.leaky_relu(self.conv1(x))  # (100, 16, 64, 64)
        x = self.dropout1(x)  # Dropout after conv1
        x = F.leaky_relu(self.conv2(x))  # (100, 8, 32, 32)
        x = self.dropout2(x)  # Dropout after conv2
        x = x.view(x.size(0), -1)  # Flatten to (100, 8192)

        # Custom layers
        # x = self.layer1(x)  # (100, 2048)
        # x = self.layer2(x)  # (100, 256)
        # x = self.layer3(x)  # (100, 128)
        # x = self.dropout3(x)  # Dropout after custom layers
        # x = self.layer4(x)  # (100, 50)
        

        # Decoder
        x = F.leaky_relu(self.fc4(x))  # (100, 128)
       # x = self.dropout_fc4(x)  # Dropout after fc4
        #x = F.leaky_relu(self.fc5(x)) # (100, 256)
        #x = self.dropout_fc5(x)  # Dropout after fc5
        #x = F.leaky_relu(self.fc6(x)) # (100, 2048)
        #x = self.dropout_fc6(x)  # Dropout after fc6
        #x = F.leaky_relu(self.fc7(x)) # (100, 2048)
        #x = self.dropout_fc7(x)  # Dropout after fc

        x = x.view(x.size(0), 8, 32, 32)  # Reshape to (100, 8, 16, 16)

        x = self.deconv1(x)  # (100, 16, 32, 32)
        x = self.deconv2(x)  # (100, 1, 64, 64)

        return x
    
    # Training loop with image saving and optional profiling
    def train_epoch(self, train_loader, optimizer, criterion, device, epoch, profile=False):
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
            hf.progress_bar(iteration=batch_idx+1, total=len(train_loader), prefix=f'Training Batch (Epoch: {epoch}): cycle {batch_idx} - loss is {loss.item()}', suffix=f' Complete - Last Saved: {last_save_cycle} with loss {last_save_loss:.4f}', length=50)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            scheduler.step()
            if loss.item() <self.loss.item() :
                    self.loss = nn.Parameter(loss.detach(), requires_grad=False)
                    self.save_model()
                    last_save_cycle = batch_idx
                    last_save_loss =loss.item()
                    #print(f'Model {self.model_name} saved successfully')

        return train_loss / len(train_loader)


    # Validation function with image saving
    def validate_epoch(self, val_loader, criterion, device, epoch, save_dir,save_dir2):

        best_model = CustomAutoencoder()
        best_model.model_name = self.model_name
        best_model.load_model()
        best_model.eval()
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