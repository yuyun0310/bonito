# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.quantization import quantize_dynamic

# torch.backends.quantized.engine = 'qnnpack'

# """
# Model A: Original
# """
# class ModelA(nn.Module):
#     def __init__(self):
#         super(ModelA, self).__init__()
#         self.fc1 = nn.Linear(784, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

# print("A")

# model_a = ModelA()
# model_a.eval()  # Ensure the model is in evaluation mode

# # Check parameter types in Model A
# for name, param in model_a.named_parameters():
#     print(f"{name}: {param.dtype}")
#     print(f"{name}: {param}")

# print("*" * 50)

# """
# Model B: Quantized
# """
# print("B")

# # Apply dynamic quantization
# model_b = quantize_dynamic(model_a, {nn.Linear}, dtype=torch.qint8)
# print(model_b.named_parameters)

# # Check parameter types in Model B
# for name, param in model_b.named_parameters():
#     print(f"{name}: {param.dtype}")
#     print(f"{name}: {param}")

# print("*" * 50)

# """
# Model C: Quantized but with full float
# """
# print("C")

# model_c = ModelA()

# # Assuming we want to copy weights directly, though they are quantized
# with torch.no_grad():
#     for param_c, param_b in zip(model_c.parameters(), model_b.parameters()):
#         param_c.data = param_b.data

# for name, param in model_c.named_parameters():
#     print(f"{name}: {param.dtype}")
#     print(f"{name}: {param}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import get_default_qconfig
from torch.quantization import prepare
from torch.quantization import convert

# For x86 architectures, use FBGEMM:
# torch.backends.quantized.engine = 'fbgemm'

# For ARM architectures, use QNNPACK:
torch.backends.quantized.engine = 'qnnpack'

class QuantizedModel(nn.Module):
    def __init__(self):
        super(QuantizedModel, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x
    
def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Database
"""
    
# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the MNIST dataset
])

# Load the MNIST dataset for training
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a DataLoader for the calibration process
# Here, we're using a subset of the training data for calibration
calibration_dataset = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)


"""
Model
"""
model = QuantizedModel()
model.to(device)
model.eval()

# Prepare the model for static quantization
model_prepared = prepare(model, inplace=False)

# Assuming calibration_dataset is a DataLoader object
for data, target in calibration_dataset:
    model_prepared(data)

# Convert the prepared model to a fully quantized model
model_quantized = convert(model_prepared, inplace=False)

# Saving the model
torch.save(model_quantized.state_dict(), "model_quantized.pth")

accuracy = calculate_accuracy(model, calibration_dataset, device)
print(f'Accuracy: {accuracy:.2f}%')

accuracy = calculate_accuracy(model_quantized, calibration_dataset, device)
print(f'Accuracy: {accuracy:.2f}%')

for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")
    # print(f"{name}: {param}")

for name, param in model_quantized.named_parameters():
    print(f"{name}: {param.dtype}")
    # print(f"{name}: {param}")

for name, module in model_quantized.named_modules():
    print(f"{name}: {type(module)}")