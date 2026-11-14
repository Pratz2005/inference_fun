# Day 16 notes:
# - Weights are saved once, then loaded every run. Without the load line, each run gets fresh random weights and CPU vs GPU can never match.
# - model.eval() turns dropout off at inference. It never changes the saved weights, dropout only masks activations in training-mode forward passes.
# - model.to(device) moves the weights and buffers(running mean and variance)
# - torch.inference_mode() stops gradient tracking during inference.
# - CPU vs GPU logits can differ by tiny float noise (~1e-6 for values ~1). Compare with a tolerance like 1e-5, not exact equality.

import torch
import argparse
import torch.nn as nn
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', help='Choose device', default = "cpu")
args = parser.parse_args()

class NeuralNetwork(nn.Module):
    def _init_(self):
        super()._init_()
        self.linear_relu = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.linear_relu(x)

device = args.device
input_features = torch.tensor([[1,2,3], [4,5,6], [7,8,9]], dtype=torch.float32).to(device)

model = NeuralNetwork().to(device)

if not os.path.exists("model_weights.pt"):
    torch.save(model.state_dict(), "model_weights.pt")  # first run only
model.load_state_dict(torch.load("model_weights.pt", map_location=device))
model.eval()

with torch.inference_mode():
    logits = model(input_features)

print(model.linear_relu[0].weight.shape)
print(logits)
print(f"Shape of logits is {logits.shape}")
