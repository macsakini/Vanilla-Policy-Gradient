import torch
import torch.nn as nn
import torch.optim as optim


class VPGModel(nn.Module):
    def __init__(self, state_space, action_space, lr):
        super(VPGModel, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr

        self.model = nn.Sequential(
            nn.Linear(self.state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space)
        )

        self.softmax = nn.Softmax()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.loss = nn.SmoothL1Loss()

    def forward(self, x):
        logits = self.model(x)
        probs = self.softmax(logits)
        return probs
