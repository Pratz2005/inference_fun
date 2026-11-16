import torch 
from torch import nn
from gpu_timer import GPUTimer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(3, 8),
                nn.ReLU(),
                nn.Linear(8, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
timer = GPUTimer(warmup = 10, reps=20)


for bs in [1, 8, 64, 512, 4096]:
    x = torch.randn((bs, 3), dtype = torch.float32, device = device)
    
    # print(x.shape)

    torch.cuda.memory.reset_peak_memory_stats()

    result = timer.time(lambda: model(x))
    peak_memory = torch.cuda.memory.max_memory_allocated(device = device) / (1024**2)

    latency = result["median_ms"]
    throughput = bs / (latency / 1000)

    print(f"bs={bs:5d}  latency={latency:8.3f} ms  throughput={throughput:10.1f}/s  peak={peak_memory:7.1f} MB")


"""
With each batch the throught increases as we get more samples per sec but latency increases and the model takes more time to process the request

offline vs interactive will have diff latency and throughput requirements so need to check accordingly

for offline u would want to get more samples per sec and for interactive you would want to maximise throughput keeping latency to a minimum. Offline has bigger batch so it uses gpu better but interactive since bs is usually less gpu is idle and the fixed overhead is spread over fewer samples
"""


    
    




