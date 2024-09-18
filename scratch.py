import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup

# Simulate parameters
num_training_steps = 1000  # Total number of training steps
num_warmup_steps = 100  # Number of warmup steps
learning_rate = 0.001  # Initial learning rate

# Create a dummy optimizer
model = torch.nn.Linear(10, 2)  # Simple model for demonstration
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Create the scheduler
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

# Store learning rates for plotting
learning_rates = []

for step in range(num_training_steps):
    # Simulate a training step
    optimizer.step()

    # Update the scheduler
    scheduler.step()

    # Record the learning rate
    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)

# Plot the learning rate schedule
plt.figure(figsize=(10, 6))
plt.plot(learning_rates)
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.title("Cosine Schedule with Warmup")
plt.grid(True)
plt.show()
