class HeteroGCLSTM:
    def __init__(self, input_size, hidden_size, num_layers):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Further initialization code here

    def forward(self, x):
        # Forward pass implementation here
        pass

    def predict(self, x):
        # Prediction implementation here
        return self.forward(x)

class HeteroGCLSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.wrapper = ThermalModelWrapper(input_size, hidden_size, output_size)

    def train(self, data_loader, criterion, optimizer):
        self.wrapper.model.train()
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = self.wrapper.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        return loss.item()
