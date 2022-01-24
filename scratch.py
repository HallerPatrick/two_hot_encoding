import torch
from torch import nn
from torch import optim



class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.n_hidden = 32
        self.n_layers = 3
        
        self.lstm = nn.LSTM(
            input_size=24,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True
        )
        self.linear = nn.Linear(in_features=self.n_hidden, out_features=24)
    
    def forward(self, x):
        h0 = torch.zeros((self.n_layers, x.size(0), self.n_hidden))
        c0 = torch.zeros((self.n_layers, x.size(0), self.n_hidden))
        
        output, state = self.lstm(x, (h0, c0))
        output = self.linear(output)
        
        return output, state

model = LSTM()
loss = nn.CrossEntropyLoss()

t = torch.zeros(1, 7, 24)
t1 = torch.zeros(1, 7, 24)
print(t.size()) # [1, 7, 24]
print(t1.size()) # [1, 7, 24]
output, hidden = model(t)
l = loss(output, t1)
print(l)
