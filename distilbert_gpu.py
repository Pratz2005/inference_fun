import torch
import transformers
from transformers import pipeline
from accelerate import Accelerator
import random
from gpu_timer import GPUTimer
random.seed(42)

device = Accelerator().device

classifier = pipeline(task="text-classification", model = "distilbert-base-uncased-finetuned-sst-2-english", device = device)

positive = ["loved", "enjoyed", "a masterpiece,", "a triumph,", "wonderful", "brilliant"]
negative = ["hated", "boring,", "a disaster,", "painful to watch,", "a waste of time,", "tedious"]
subjects = ["the movie", "this film", "the sequel", "the acting", "the plot", "the ending"]

texts = []
for i in range(256):
    words = random.randint(3, 80)
    base = random.choice(subjects) + " " + random.choice(positive + negative)
    filler = " ".join(random.choices(["really", "very", "quite", "honestly", "surprisingly", "somewhat"], k=words))
    texts.append(f"{base} {filler}".strip())

timer = GPUTimer(warmup = 3, reps = 5)

for bs in [1, 16, 64, 256]:
    def run():
        out = classifier(texts, batch_size = bs)
        # print(out)

    r = timer.time(run)
    total_ms = r["median_ms"]
    print(f"bs={bs:4d}  total={total_ms:9.1f} ms  throughput={256/(total_ms/1000):8.1f} texts/s")
    

"""
THe pipeline runs the model on batches sequentially so gpu is underutilized. To run it parallely best to wrap it in a Hugging Face dataset

here padding makes many sentences in batch wait on the longest. fix is to sort by length then batch. COntinuous batching as well(finioshed sequences leaev and new ones enter
"""
