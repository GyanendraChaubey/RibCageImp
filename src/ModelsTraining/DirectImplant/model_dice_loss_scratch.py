print("Started Importing Necessary Libraries...")
import os
import glob
import nibabel as nib
import numpy as np
import random
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

print("Necessary Libraries imported")

# Initialize wandb
wandb.init(project="M.TechProject")

print("\nData Loading Started...")

# Function to resize data
def resizing(data, target_shape=(128, 128, 64)):
    """Resize the data to the target shape."""
    a, b, c = data.shape
    return zoom(data, (target_shape[0] / a, target_shape[1] / b, target_shape[2] / c), order=2, mode='constant')


# Custom Dataset Class
class MedicalDataset(Dataset):
    def __init__(self, data_list, label_list, transform=None):
        """
        Args:
            data_list (list): List of paths to the data files.
            label_list (list): List of paths to the label files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert len(data_list) == len(label_list), "Data and label lists must be the same length"
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_file = self.data_list[idx]
        label_file = self.label_list[idx]

        data = nib.load(data_file).get_fdata()
        label = nib.load(label_file).get_fdata()

        data_resized = resizing(data)
        label_resized = resizing(label)

        #data_resized = np.expand_dims(data_resized, axis=(0, 1))
        #label_resized = np.expand_dims(label_resized, axis=(0, 1))

        data_tensor = torch.from_numpy(data_resized).float()
        label_tensor = torch.from_numpy(label_resized).float()

        sample = {'data': data_tensor, 'label': label_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Create DataLoader
def create_dataloader(data_list, label_list, batch_size=4, shuffle=True, num_workers=2):
    """Create DataLoader for the dataset."""
    dataset = MedicalDataset(data_list, label_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


# Define directories
train_data_dir = '/workspace/RibCage/train-ribfrac-defected'
train_label_dir = '/workspace/RibCage/train-ribfrac-implants'
test_data_dir = '/workspace/RibCage/test-ribfrac-defected'
test_label_dir = '/workspace/RibCage/test-ribfrac-implants'

# Get list of files
train_data_list = sorted(glob.glob(os.path.join(train_data_dir, '*.nii')) + glob.glob(os.path.join(train_data_dir, '*.nii.gz')))
train_label_list = sorted(glob.glob(os.path.join(train_label_dir, '*.nii')) + glob.glob(os.path.join(train_label_dir, '*.nii.gz')))
test_data_list = sorted(glob.glob(os.path.join(test_data_dir, '*.nii')) + glob.glob(os.path.join(test_data_dir, '*.nii.gz')))
test_label_list = sorted(glob.glob(os.path.join(test_label_dir, '*.nii')) + glob.glob(os.path.join(test_label_dir, '*.nii.gz')))

# Ensure data and labels are paired properly
assert len(train_data_list) == len(train_label_list), "Training data and labels are not of the same length"
assert len(test_data_list) == len(test_label_list), "Training data and labels are not of the same length"

# Create DataLoader for training and testing
train_loader = create_dataloader(train_data_list, train_label_list, batch_size=2, shuffle=True)
test_loader = create_dataloader(test_data_list, test_label_list, batch_size=2, shuffle=True)

print("Train and Test Loader has been created...")

# Iterate through the DataLoader
print("\nTesting DataLoader for training data...")
for i, batch in enumerate(train_loader):
    data_tensor = batch['data']
    label_tensor = batch['label']
    print(f'Batch {i + 1}: Data shape: {data_tensor.shape}, Label shape: {label_tensor.shape}')
    if i == 2:  # Load a few batches for demonstration
        break

# Iterate through the DataLoader
print("\nTesting DataLoader for testing data...")
for i, batch in enumerate(test_loader):
    data_tensor = batch['data']
    label_tensor = batch['label']
    print(f'Batch {i + 1}: Data shape: {data_tensor.shape}, Label shape: {label_tensor.shape}')
    if i == 2:  # Load a few batches for demonstration
        break

print("Started defining the convolution Layers...")
        
# Define the convolutional layer with initialization
class Conv3dLayer(nn.Module):
    def __init__(self, input_chn, output_chn, kernel_size, stride, bias=False):
        super(Conv3dLayer, self).__init__()
        padding = (kernel_size - 1) // 2  # Calculate padding
        self.conv = nn.Conv3d(input_chn, output_chn, kernel_size, stride, padding=padding, bias=use_bias)
        nn.init.trunc_normal_(self.conv.weight, std=0.01)
        if use_bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

# Define the block with convolution, batch normalization, and ReLU
class ConvBnReLU(nn.Module):
    def __init__(self, input_chn, output_chn, kernel_size, stride):
        super(ConvBnReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(input_chn, output_chn, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm3d(output_chn, momentum=0.9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Define the deconvolutional layer with initialization
class Deconv3dLayer(nn.Module):
    def __init__(self, input_chn, output_chn):
        super(Deconv3dLayer, self).__init__()
        self.deconv = nn.ConvTranspose3d(input_chn, output_chn, kernel_size=4, stride=2, padding=1)
        nn.init.normal_(self.deconv.weight, std=0.01)
        nn.init.zeros_(self.deconv.bias)

    def forward(self, x):
        return self.deconv(x)

# Define the block with deconvolution, batch normalization, and ReLU
class DeconvBnReLU(nn.Module):
    def __init__(self, input_chn, output_chn):
        super(DeconvBnReLU, self).__init__()
        self.deconv = Deconv3dLayer(input_chn, output_chn)
        self.bn = nn.BatchNorm3d(output_chn, momentum=0.9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Define the block with repeated convolution, batch normalization, and ReLU
class ConvBnReLUX3(nn.Module):
    def __init__(self, input_chn, output_chn, kernel_size, stride, use_bias):
        super(ConvBnReLUX3, self).__init__()
        self.conv1 = ConvBnReLU(input_chn, output_chn, kernel_size, stride, use_bias)
        self.conv2 = ConvBnReLU(output_chn, output_chn, kernel_size, stride, use_bias)
        self.conv3 = ConvBnReLU(output_chn, output_chn, kernel_size, stride, use_bias)

    def forward(self, x):
        z = self.conv1(x)
        z_out = self.conv2(z)
        z_out = self.conv3(z_out)
        return z + z_out        
 
print("the convolution Layers has been defined...")

print("Started defining Autoencoder Model...")


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = ConvBnReLU(2, 64, kernel_size=5, stride=2)
        self.conv2 = ConvBnReLU(64, 128, kernel_size=5, stride=2)
        self.conv3 = ConvBnReLU(128, 256, kernel_size=5, stride=2)
        self.conv4 = ConvBnReLU(256, 512, kernel_size=5, stride=2)
        self.conv5 = ConvBnReLU(512, 512, kernel_size=5, stride=1)
        self.deconv1 = DeconvBnReLU(512, 256)
        self.deconv2 = DeconvBnReLU(256, 128)
        self.deconv3 = DeconvBnReLU(128, 64)
        self.deconv4 = DeconvBnReLU(64, 32)
        self.pred_prob1 = ConvBnReLU(32, 2, kernel_size=5, stride=1)
        self.pred_prob2 = nn.Conv3d(2, 2, kernel_size=5, stride=1, padding='same', bias=True)
        self.pred_prob3 = nn.Conv3d(2, 2, kernel_size=5, stride=1, padding='same', bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #print("input x:",x.shape)
        conv1 = self.conv1(x)
        #print("input conv1:",conv1.shape)
        conv2 = self.conv2(conv1)
        #print("input conv2:",conv2.shape)
        conv3 = self.conv3(conv2)
        #print("input conv3:",conv3.shape)
        conv4 = self.conv4(conv3)
        #print("input conv4:",conv4.shape)
        conv5 = self.conv5(conv4)
        #print("input conv5:",conv5.shape)
        deconv1 = self.deconv1(conv5)
        #print("input deconv1:",deconv1.shape)
        deconv2 = self.deconv2(deconv1)
        #print("input deconv2:",deconv2.shape)
        deconv3 = self.deconv3(deconv2)
        #print("input deconv3:",deconv3.shape)
        deconv4 = self.deconv4(deconv3)
        #print("input deconv4:",deconv4.shape)
        pred_prob1 = self.pred_prob1(deconv4)
        pred_prob2 = self.pred_prob2(pred_prob1)
        pred_prob3 = self.pred_prob3(pred_prob2)
        soft_prob = self.softmax(pred_prob3)
        return soft_prob
    
print("Autoencoder Model has been defined...")

print("Started defining loss function...")

def dice_loss(pred, target):
    smooth = 1.
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    loss = (2. * intersection + smooth) / (pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + smooth)
    loss = 1 - loss.mean()
    return loss

print("Loss function has been defined...")


print("Model object, criterion and optimizer has been called...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoEncoder().to(device)
criterion = dice_loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Device: ",device)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print(f"=> Checkpoint saved to '{filename}'")

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    if os.path.isfile(filename):
        print("=> Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        return True
    else:
        print("=> No checkpoint found at '{}'".format(filename))
        return False

def test_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['data']
            labels = batch['label']
            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

    test_loss = running_loss / len(dataloader.dataset)
    return test_loss

print("Training has been started...")

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval=10, early_stopping_patience=10):
    best_loss = float('inf')
    no_improvement_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch['data']
            labels = batch['label']
            inputs = inputs.unsqueeze(0)
            labels = labels.unsqueeze(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
        
        # Log training loss to wandb
        wandb.log({"Training Loss": epoch_loss, "epoch": epoch + 1})

        # Evaluate on the test set
        test_loss = test_model(model, test_loader, criterion)
        print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.4f}')
        
        # Log test loss to wandb
        wandb.log({"Test Loss": test_loss, "epoch": epoch + 1})

        # Save the best model every five intervals
        if ((epoch + 1) // checkpoint_interval) % 5 == 0:
            if test_loss < best_loss:
                best_loss = test_loss
                no_improvement_counter = 0
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(state, f'checkpoint_best_interval_{(epoch + 1) // checkpoint_interval}.pth.tar')
            else:
                no_improvement_counter += 1

        # Early stopping
        if no_improvement_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print('Training finished!')
    wandb.finish()

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)

print("Training has been completed...")