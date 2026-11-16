class GPUTimer:
    def __init__(self, warmup=10, reps=20):
        self.warmup = warmup
        self.reps = reps

    def time(self, fn):
        # Warmup operations
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)  # records start of ops
        end_event = torch.cuda.Event(enable_timing=True)  # records end of ops
        times = []
        for _ in range(self.reps):
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()  # prevents CPU thread from executing other tasks
            times.append(start_event.elapsed_time(end_event))
        return {
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
        }

# usage:
# timer = GPUTimer()
# result = timer.time(lambda: tensor_a @ tensor_b)
# print(result["median_ms"])
