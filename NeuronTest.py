# the equation combined 2 tanh functions to get a signal band where the neuron is activated and other regions where it's not
# the steepness is fixed to make it very steep at 10e3. the window value from the right right_limit and the limit from the left need to be strictly positve 
# and between 10e-2 9or anothe rnon zero small number) and the steepness values at the most (clamp it to 9*10e2 or something like that)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Function to create a simple dataset
def create_dataset(num_samples=10000):
    """Generate a dataset where the second column is 1 if the first column is between 0.3 and 0.7, otherwise 0."""
    x = torch.rand(num_samples, 1)  # Inputs between 0 and 1
    y = ((x >= 0.3) & (x <= 0.7)).float()  # Labels: 1 if 0.3 <= x <= 0.7, else 0
    return x, y

class CustomNeuron(nn.Module):
    def __init__(self):
        super(CustomNeuron, self).__init__()
        self.steepness  = nn.Parameter(torch.tensor(1.0),requires_grad=False)   
        self.left_limit = nn.Parameter(torch.tensor(3.0),requires_grad=True)
        self.param_diff = nn.Parameter(torch.tensor(4.0),requires_grad=True) 
        self.max_value = nn.Parameter(torch.tensor(1),requires_grad=False)  
        self.min_value = nn.Parameter(torch.tensor(0.01),requires_grad=False) 
        self.epsilon = torch.tensor(1e-3,device='cuda')

    def forward(self, x):
        #y\ =\ \tanh\left(\tanh\left(b\cdot x-p\right)\ -\ \tanh\left(c\cdot x-d\right)\right)
        #self.param_diff.data = torch.abs(self.param_diff) + self.epsilon
        #right_limit = self.left_limit + self.param_diff
        #right_limit= torch.clamp(right_limit, max=self.max_value)

        #self.left_limit.data = torch.clamp(torch.tensor(torch.tensor(1e-5),device='cuda'), max=right_limit-self.min_value)

        activation = torch.tanh(torch.tanh(1e3*self.steepness*x-1e2*self.left_limit) -torch.tanh(1e3*self.steepness*x-1e2*(self.left_limit + self.param_diff)))
        return (torch.round(activation * 1e6) / 1e6)/torch.tensor(0.964028)
		

# 3. Simple nn.Linear Neuron Class
class LinearNeuron(nn.Module):
    def __init__(self):
        super(LinearNeuron, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input, single output

    def forward(self, x):
        return self.linear(x)

# 4. Training Loop
def train(model, criterion, optimizer, dataloader,device, epochs=100):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in dataloader:
            # Forward pass
            raw_outputs = model(x_batch.to(device))
            predictions = torch.sigmoid(raw_outputs) # Apply sigmoid here
            loss = criterion(raw_outputs, y_batch.to(device))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.6f}")
        print(f'Epoch Average Loss = {total_loss / len(dataloader):.6f}')

# 5. Validation Loop
def validate(model, dataloader, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            raw_outputs = model(x_batch.to(device))
            predictions = torch.sigmoid(raw_outputs)  # Apply sigmoid here
            loss = nn.MSELoss()(predictions, y_batch.to(device))
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader):.4f}")

# Example usage
if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate dataset
    x, y = create_dataset()
    
    # Display the first 20 values of the dataset
    print("First 20 values of the dataset:")
    for i in range(20):
        print(f"Input: {x[i].item():.4f}, Label: {y[i].item():.0f}")
    
    # Create DataLoader for training and validation
    dataset = TensorDataset(x, y)
    dataset_length = len(dataset)

    # Calculate split sizes
    train_size = int(0.8 * dataset_length)  # 80% for training
    val_size = dataset_length - train_size  # Remaining for validation

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=30, shuffle=False)

    # Initialize models
    custom_neuron = CustomNeuron().to(device)
    linear_neuron = LinearNeuron().to(device)

    # Loss function and optimizers
    criterion = nn.MSELoss()  # Binary Cross Entropy Loss
    custom_optimizer = optim.SGD(custom_neuron.parameters(), lr=0.1)
    linear_optimizer = optim.SGD(linear_neuron.parameters(), lr=0.001)

    print("Training Custom Neuron:")
    train(custom_neuron, criterion, custom_optimizer, train_loader, device, epochs=1000)
    
    print("\nValidating Custom Neuron:")
    validate(custom_neuron, val_loader, device)

    print("\nTraining Linear Neuron:")
    train(linear_neuron, criterion, linear_optimizer, train_loader, device,epochs=1000)
    
    print("\nValidating Linear Neuron:")
    validate(linear_neuron, val_loader, device)



