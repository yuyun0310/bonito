# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from pytorch_quantization import quant_modules
# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization.tensor_quant import QuantDescriptor

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjusted for output from Conv2D
#         self.lstm = nn.LSTM(128, 128, batch_first=True)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = x.view(-1, 32 * 32 * 32)  # Flatten the output for the Linear layer
#         x = self.relu(self.fc1(x))
#         x = x.view(-1, 1, 128)  # Adjust shape for LSTM
#         x, _ = self.lstm(x)
#         x = self.fc2(x[:, -1, :])
#         return x

# class SyntheticDataset(Dataset):
#     def __init__(self, num_samples=1000, image_size=(64, 64)):
#         self.num_samples = num_samples
#         self.data = torch.randn(num_samples, 1, *image_size)
#         self.targets = torch.randint(0, 10, (num_samples,))

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx]

# # Initialize quantization
# quant_modules.initialize()

# # Create and quantize the model
# model = MyModel()
# model = quant_modules.quantize_module(model)

# # Check if CUDA is available and move the model to GPU if possible
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Create the synthetic dataset and dataloader for calibration
# dataset = SyntheticDataset(num_samples=100, image_size=(64, 64))
# calib_loader = DataLoader(dataset, batch_size=10, shuffle=True)

# # Perform the calibration
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     for inputs, _ in calib_loader:
#         inputs = inputs.to(device)
#         model(inputs)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_quantization import quant_modules
from pytorch_quantization.quant_module import QuantWrapper
from pytorch_quantization.nn import QuantLinear, QuantConv2d
from pytorch_quantization import calib

# Initialize quantization
quant_modules.initialize()

# Define the synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(32, 32), seq_length=50):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.image_size = image_size
        # Simulate single-channel image data
        self.data = torch.randn(num_samples, 1, *image_size)
        # Simulate sequence data for LSTM
        self.targets = torch.randint(0, 10, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = QuantConv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.lstm = nn.LSTM(32 * 16 * 16, 128, batch_first=True)  # Adjusted for the output size of conv1
        self.fc = QuantLinear(128, 10)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = x.view(x.size(0), 1, -1)  # Flatten and prepare for LSTM
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Assuming the output of the last sequence is used for classification
        return x

# Wrap model for quantization
model = QuantWrapper(MyModel())

# Prepare data loaders
dataset = SyntheticDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Calibration
model.eval()
with torch.no_grad():
    for inputs, _ in dataloader:
        model(inputs)

# You would typically collect and analyze statistics here to optimize quantization parameters
# This example omits detailed calibration steps for brevity

print("Model defined, data prepared, and basic calibration done.")
