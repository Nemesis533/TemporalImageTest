# the equation combined 2 tanh functions to get a signal band where the neuron is activated and other regions where it's not
# the steepness is fixed to make it very steep at 10e3. the window value from the right right_limit and the limit from the left need to be strictly positve 
# and between 10e-2 9or anothe rnon zero small number) and the steepness values at the most (clamp it to 9*10e2 or something like that)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import helper_functions as hf
import os
import torch.nn.functional as F

train_model = True
batch_size=100

def create_dataset(num_samples=100000):
    """Generate a dataset where the second column is 1 if the first column is between 0.3 and 0.7,
    otherwise 0 (three classes: 0, 1, 2)."""
    x = torch.rand(num_samples, 1)  # Inputs between 0 and 1
    # Create a one-hot encoded label for three classes
    y = torch.zeros(num_samples, 3)  # Initialize a tensor for three classes
    x = x.squeeze()  # Convert x from shape (num_samples, 1) to (num_samples,)
    
    # Class 0: 0.0 <= x < 0.3
    y[(x >= 0.0) & (x < 0.3), 0] = 1
    # Class 1: 0.3 <= x < 0.7
    y[(x >= 0.3) & (x < 0.7), 1] = 1
    # Class 2: 0.7 <= x <= 1.0
    y[(x >= 0.7) & (x <= 1.0), 2] = 1
    
    return x, y

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.output = CustomLayer(1,3)        
        self.model_name = "CustomNeuronTest"
        self.model_save_dir ="./Models"
        self.loss = nn.Parameter(torch.tensor(10e9), requires_grad=False)
    
    def forward(self, x):

        x = self.output(x)
        return x
    
    def save_model(self):
        # Ensure the directory exists
        os.makedirs(f'{self.model_save_dir}{self.model_name}', exist_ok=True)

        # Now save the model
        torch.save(self.state_dict(), f'{self.model_save_dir}{self.model_name}/{self.model_name}.pth')

        #print(f'Model {self.model_name} saved successfully')

    def load_model(self):
        self.load_state_dict(torch.load(f'{self.model_save_dir}{self.model_name}/{self.model_name}.pth'))
        
        print(f'Model {self.model_name} loaded successfully - loss was {self.loss}')

class CustomLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomLayer, self).__init__()
        self.neurons = nn.ModuleList([CustomNeuron() for _ in range(output_size)])

    def forward(self, x):
       
        neuron_outputs = torch.stack([neuron(x) for neuron in self.neurons], dim=1)
        output = neuron_outputs.squeeze(1)
        
        return output  

class CustomNeuron(nn.Module):
    def __init__(self):
        super(CustomNeuron, self).__init__()
        self.steepness  = nn.Parameter(torch.tensor(1.0),requires_grad=False)   
        self.left_limit = nn.Parameter(torch.rand(1) ,requires_grad=True)
        self.param_diff = nn.Parameter(torch.rand(1) ,requires_grad=True) 
        self.base_multiplier = nn.Parameter(torch.rand(1)*1e3 ,requires_grad=True) 
        self.param_multiplier = nn.Parameter(self.base_multiplier/10  ,requires_grad=False)  
        self.gain = nn.Parameter(torch.rand(1) ,requires_grad=True) 
        self.negative_gain = nn.Parameter(torch.rand(1) ,requires_grad=True) 
        # Store learning rates
        self.lr_dict = {
            self.left_limit: 1,
            self.param_diff: 1,
            self.base_multiplier: 1,
            self.gain: 0.1,
            self.negative_gain : 0.1,
        }

    def apply_custom_lr(self):
        # Scale gradients based on the specified learning rate
        for param, lr in self.lr_dict.items():
            if param.grad is not None:
                param.grad *= lr  # Modify the gradient in-place

    def forward(self, x):
        #y\ =\ \tanh\left(\tanh\left(b\cdot x-p\right)\ -\ \tanh\left(c\cdot x-d\right)\right)
        steepness = self.base_multiplier*self.steepness
        first_param =self.param_multiplier*self.left_limit
        second_param = self.param_multiplier*(self.left_limit + self.param_diff)

        activation = self.gain*torch.tanh(torch.tanh(steepness*x-first_param) - torch.tanh(steepness*x-second_param))-self.negative_gain
       

        return activation

# 3. Simple nn.Linear Neuron Class
class LinearNeuron(nn.Module):
    def __init__(self):
        super(LinearNeuron, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input, single output
        self.output = nn.Linear(1,3)    


    def forward(self, x):
        x = F.sigmoid(self.linear(x.unsqueeze(1)))
        x = self.output(x)
        return x

# 4. Training Loop
def train(model, criterion, optimizer, dataloader,device, epochs=100):
    model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)
    last_save_cycle = 0
    last_save_loss = 0
    highest_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx,(x_batch, y_batch) in enumerate(dataloader):
            model.train()
            # Forward pass: get raw outputs (logits)
            raw_outputs = model(x_batch.to(device))
            
            # Apply BCEWithLogitsLoss 
            loss = criterion(raw_outputs, y_batch.to(device))  
            

            probabilities = torch.sigmoid(raw_outputs)
            
            # Get the predicted classes: 1 if probability > 0.5, else 0
            predictions = (probabilities > 0.5).float()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Apply custom learning rates
            try:
                for neuron in model.output.neurons:
                    neuron.apply_custom_lr()
            except:
                pass

            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Assuming predictions are thresholded binary (0/1) and y_batch is binary
            correct = (predictions == y_batch.to(device)).float().mean()  # Element-wise comparison, then average
            accuracy = correct.item()*100  # Convert to Python float

            # Save model if current loss is better than the previous best
            try:
                if loss.item() < model.loss.item():
                    highest_accuracy = accuracy
                    model.loss = nn.Parameter(loss.detach(), requires_grad=False)
                    model.save_model()
                    last_save_cycle = batch_idx
                    last_save_loss = loss.item()
                    # print(f'Model {self.model_name} saved successfully')
            
            except:
                pass

            
                

            # Print progress
            hf.progress_bar(
                    iteration=batch_idx + 1, 
                    total=len(train_loader), 
                    prefix=f'Training Batch (Epoch: {epoch}): cycle {batch_idx} - accuracy is: {accuracy:.2f}%, loss is {loss.item()}', 
                    suffix=f' Complete - Last Saved: {last_save_cycle} with loss {last_save_loss:.4f} - best accuracy is {highest_accuracy:.2f}', 
                    length=50
                )
   

# Validation function
def validate(model, dataloader, device):
    try:
        model.load_model()
    except:
        pass
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass: Get raw logits from the model
            raw_outputs = model(x_batch)
            
            # Apply sigmoid and threshold to get binary predictions
            predictions = (raw_outputs) > 0.5  # Threshold at 0.5
            
            # Calculate the number of correct predictions (compare each class)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.numel()  # Total number of elements

    accuracy = (correct / total) * 100  # Accuracy in percentage
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
# Example usage
if __name__ == "__main__":
    

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dataset
    x, y = create_dataset()

    # Create DataLoader for training and validation
    dataset = TensorDataset(x, y)
    dataset_length = len(dataset)

    # Calculate split sizes
    train_size = int(0.8 * dataset_length)  # 80% for training
    val_size = dataset_length - train_size  # Remaining for validation

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    custom_neuron = CustomNet().to(device)
    linear_neuron = LinearNeuron().to(device)

    # Loss function and optimizers
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
    custom_optimizer = optim.SGD(custom_neuron.parameters(), lr=1)
    linear_optimizer = optim.SGD(linear_neuron.parameters(), lr=0.001)
    if train_model:
        print("Training Custom Neuron:")
        train(custom_neuron, criterion, custom_optimizer, train_loader, device, epochs=10)
        
        print("\nValidating Custom Neuron:")
        validate(custom_neuron, val_loader, device)

        print("\nTraining Linear Neuron:")
        train(linear_neuron, criterion, linear_optimizer, train_loader, device,epochs=10)
        
        print("\nValidating Linear Neuron:")
        validate(linear_neuron, val_loader, device)
    else:
        print("\nValidating Custom Neuron:")
        validate(custom_neuron, val_loader, device)
        
        print("\nValidating Linear Neuron:")
        validate(linear_neuron, val_loader, device)



