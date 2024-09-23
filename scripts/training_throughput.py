import os
import sys
import time

import torch
import yaml

# Adjust the Python path to include the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.helper import get_model
from NeuroCompress.NeuroPress import freeze_model

# Load configuration from YAML file
with open("../config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Define model architecture parameters
backbone_arch = "ternary_vit"
agg_arch = "cls"

# Initialize the model
model = get_model(
    image_size=[224, 224],
    backbone_arch=backbone_arch,
    agg_arch=agg_arch,
    model_config=config["Model"],
    normalize_output=True,
)

# Create dummy input and target tensors
batch_size = 64
img_batch = torch.randn(batch_size, 3, 224, 224)
# Adjust the target shape according to your model's output
# For example, if it's a classification model with 768 classes:
target = torch.randint(
    0, 100, (batch_size,)
)  # Assuming classification with integer targets

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
img_batch = img_batch.to(device)
target = target.to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # Change if using a different loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Optionally freeze parts of the model
# freeze_model(model)  # Uncomment if you want to freeze the model

# Set the model to training mode
model.train()

# Define number of warm-up and timed iterations
warmup_iterations = 10
timed_iterations = 100

# Warm-up phase (not timed)
print("Warming up...")
for _ in range(warmup_iterations):
    optimizer.zero_grad()
    outputs = model(img_batch)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

# Synchronize CUDA before starting the timer
if torch.cuda.is_available():
    torch.cuda.synchronize()

print("Starting timed training iterations...")
start_time = time.time()

for _ in range(timed_iterations):
    optimizer.zero_grad()
    outputs = model(img_batch)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

# Synchronize CUDA after finishing the iterations
if torch.cuda.is_available():
    torch.cuda.synchronize()

end_time = time.time()
total_time = end_time - start_time

# Calculate throughput
total_samples = batch_size * timed_iterations
throughput = total_samples / total_time

print(f"Training Throughput: {throughput:.2f} samples/second")
print(f"Total Time for {timed_iterations} iterations: {total_time:.2f} seconds")
