import json
import os
import torch

print("GPUs available: ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(f"cuda:{i}"))


