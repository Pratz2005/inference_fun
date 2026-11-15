import torch
import time
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor_a = torch.randn((4096, 4096), dtype = torch.float32, device = device)
tensor_b = torch.randn((4096, 4096), dtype = torch.float32, device = device)

# Using perf counter
no_sync = []
for i in range(20):
    start_time = time.perf_counter()
    res = tensor_a @ tensor_b
    end_time = time.perf_counter()
    time_ms = (end_time - start_time) * 1000
    no_sync.append(time_ms)

print(f"No sync has median of {statistics.median(no_sync)} ms")

# Warmup operations

for i in range(10):
    output = tensor_a @ tensor_b


start_event = torch.cuda.Event(enable_timing = True)
end_event = torch.cuda.Event(enable_timing = True)

sync = []
for i in range(20):
    start_event.record() # Sets bookmark at stream of ops going to GPU

    output = tensor_a @ tensor_b

    end_event.record() # Records end of ops

    torch.cuda.synchronize() # prevents CPU thread from executing other tasks while gpu is doing the matmul
    elapsed_ms = start_event.elapsed_time(end_event)
    sync.append(elapsed_ms)

print(f" Sync has median time {statistics.median(sync)} ms")

"""
The torch.cuda.event() creates a stream here it sends kernel tasks to the gpu. Each task runs its instuctions in parallel but the stream of tasks is executed sequentially. if i have multiple matmuls in a stream they are executed one after the other.

time.perf_counter measures time it take for cpu to hand off task to GPU which is why its so much lesser in time. torch.cuda.event along with syncronize hodls the cpu thread so we measure actual execution time.
"""



