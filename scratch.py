import torch
from torch.utils.data import DataLoader, Dataset


# Create a dummy dataset for testing purposes
class DummyDataset(Dataset):
    def __init__(self, size, num_features):
        self.data = torch.arange(size * num_features).reshape(size, num_features)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define a small test dataset
dummy_dataset = DummyDataset(size=5, num_features=2)
dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)


# Your infinite dataloader function
def infinite_dataloader(dataloader):
    while True:
        for data in dataloader:
            yield data


# Test script
def test_infinite_dataloader(dataloader, num_iterations=15):
    infinite_loader = infinite_dataloader(dataloader)

    print("Testing infinite dataloader:")
    for i in range(num_iterations):
        data, labels = next(infinite_loader)
        print(f"Iteration {i+1}:")
        print(f"  Data: {data}")
        print(f"  Labels: {labels}")
        print("  Batch size:", len(data))
        print("-" * 30)


# Run the test
test_infinite_dataloader(dataloader, num_iterations=15)
