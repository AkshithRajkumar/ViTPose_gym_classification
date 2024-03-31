import torch
import os
import numpy as np
from model_arch import TCN  # Assuming your model architecture is defined in a separate file

# Load the saved model
model_path = 'saved_models_150\model_e39_l0.7096.pth'  # Path to your saved model
model = TCN(input_size=(150, 17, 2), num_channels=[8, 8, 8, 8], kernel_size=3, dropout=0.2)
model.load_state_dict(torch.load(model_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prepare your data for inference
# Assuming your data is in the form of a NumPy array or a PyTorch tensor
npz_dir = 'snatch\snatch_4'

for root, dirs, files in os.walk(npz_dir):
    for file in files:
        if file.endswith('.npz'):
            npz_path = os.path.join(root, file)
            
            # Load .npz file
            npz_data = np.load(npz_path)
            npz_data = npz_data['reconstruction'][:, :, :2]
            # Pad the data to a fixed length
            last_row = np.expand_dims(npz_data[-1], 0)
            fixed_length = 150
            if len(npz_data) < fixed_length:
                npz_data = np.concatenate([npz_data, np.tile(last_row, (fixed_length - len(npz_data), 1, 1))])
            else:
                npz_data = npz_data[:fixed_length]
# Your data of size (batch_size, 150, 17, 2)

# Convert data to PyTorch tensor and move to device
# Convert data to PyTorch tensor and add a dimension
data_tensor = torch.tensor(npz_data, dtype=torch.float32).unsqueeze(0).to(device)


# Perform inference using the model
model.eval()
with torch.no_grad():
    predictions = model(data_tensor)

# Process the model output to get predictions
# For example, if you want the class with the highest probability
predicted_classes = torch.argmax(predictions, dim=1)

print("Predicted classes:", predicted_classes)
