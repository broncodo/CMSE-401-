import torch
import torch.nn as nn
import numpy as np

# Generate dummy time series data
x = np.linspace(0, 100, 1000)
y = np.sin(x)

# Create sequences
SEQ_LEN = 30
X, Y = [], []
for i in range(len(y) - SEQ_LEN):
    X.append(y[i:i+SEQ_LEN])
    Y.append(y[i+SEQ_LEN])
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
Y = torch.tensor(Y, dtype=torch.float32)

# Simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = SimpleLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    output = model(X)
    loss = criterion(output.squeeze(), Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
