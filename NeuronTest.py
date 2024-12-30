# the equation combined 2 tanh functions to get a signal band where the neuron is activated and other regions where it's not
# the steepness is fixed to make it very steep at 10e3. the window value from the right right_limit and the limit from the left need to be strictly positve 
# and between 10e-2 9or anothe rnon zero small number) and the steepness values at the most (clamp it to 9*10e2 or something like that)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

train_model = False


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


class CustomNeuron(nn.Module):
    def __init__(self):
        super(CustomNeuron, self).__init__()
        self.steepness  = nn.Parameter(torch.tensor(1.0),requires_grad=False)   
        self.left_limit = nn.Parameter(torch.tensor(3.0),requires_grad=True)
        self.param_diff = nn.Parameter(torch.tensor(4.0),requires_grad=True) 
        self.output = nn.Linear(1,3)

    def forward(self, x):
        #y\ =\ \tanh\left(\tanh\left(b\cdot x-p\right)\ -\ \tanh\left(c\cdot x-d\right)\right)
        #self.param_diff.data = torch.abs(self.param_diff) + self.epsilon
        #right_limit = self.left_limit + self.param_diff
        #right_limit= torch.clamp(right_limit, max=self.max_value)

        #self.left_limit.data = torch.clamp(torch.tensor(torch.tensor(1e-5),device='cuda'), max=right_limit-self.min_value)

        activation = torch.tanh(torch.tanh(1e3*self.steepness*x-1e2*self.left_limit) + torch.tanh(1e3*self.steepness*x-1e2*(self.left_limit + self.param_diff)))
        # (torch.round(activation * 1e6) / 1e6)/torch.tensor(0.964028)
        x = self.output(activation.unsqueeze(1))
        return x

# 3. Simple nn.Linear Neuron Class
class LinearNeuron(nn.Module):
    def __init__(self):
        super(LinearNeuron, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input, single output
        self.output = nn.Linear(1,3)    


    def forward(self, x):
        x = self.linear(x.unsqueeze(1))
        x = self.output(x)
        return x

# 4. Training Loop
def train(model, criterion, optimizer, dataloader,device, epochs=100):
    model.to(device)
    #scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

    for x_batch, y_batch in dataloader:
        # Forward pass: get raw outputs (logits)
        raw_outputs = model(x_batch.to(device))
        
        # Apply BCEWithLogitsLoss directly (no need to apply sigmoid here)
        loss = criterion(raw_outputs, y_batch.to(device))  # y_batch should be the same shape as raw_outputs
        
        # Apply sigmoid to the raw logits to get probabilities (optional, for analysis)
        probabilities = torch.sigmoid(raw_outputs)
        
        # Get the predicted classes: 1 if probability > 0.5, else 0
        predictions = (probabilities > 0.5).float()

            # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate the number of correct predictions (for multilabel classification, compare each class)
        correct = (predictions == y_batch.to(device)).float().sum()  # This works for multilabel classification
        
        # Calculate accuracy: correct predictions divided by total predictions
        accuracy = correct / y_batch.size(0)  # y_batch.size(0) gives the batch size

        #if (epoch + 1) % 10 == 0:
            #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.6f}")
            # Print or log the loss and accuracy
    print(f"Loss: {loss.item()}, Accuracy: {accuracy.item()/3 * 100:.2f}%")
        #print(f'Epoch Average Loss = {total_loss / len(dataloader):.6f}')
   

# Validation function
def validate(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass: Get raw logits from the model
            raw_outputs = model(x_batch)
            
            # Apply sigmoid and threshold to get binary predictions
            predictions = torch.sigmoid(raw_outputs) > 0.5  # Threshold at 0.5
            
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
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

    # Initialize models
    custom_neuron = CustomNeuron().to(device)
    linear_neuron = LinearNeuron().to(device)

    # Loss function and optimizers
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
    custom_optimizer = optim.SGD(custom_neuron.parameters(), lr=1)
    linear_optimizer = optim.SGD(linear_neuron.parameters(), lr=0.001)
    if train_model:
        print("Training Custom Neuron:")
        train(custom_neuron, criterion, custom_optimizer, train_loader, device, epochs=10000)
        
        print("\nValidating Custom Neuron:")
        validate(custom_neuron, val_loader, device)

        print("\nTraining Linear Neuron:")
        train(linear_neuron, criterion, linear_optimizer, train_loader, device,epochs=10000)
        
        print("\nValidating Linear Neuron:")
        validate(linear_neuron, val_loader, device)
    else:
        print("\nValidating Custom Neuron:")
        validate(custom_neuron, val_loader, device)
        
        print("\nValidating Linear Neuron:")
        validate(linear_neuron, val_loader, device)



