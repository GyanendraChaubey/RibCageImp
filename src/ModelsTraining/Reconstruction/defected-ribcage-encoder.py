import logging
import os
import re
import glob
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import time
import segmentation_models_pytorch_3d as smp
from torch.cuda.amp import autocast, GradScaler

# Logging setup
log_filename = f'defected-ribcage-encoder_{time.strftime("%Y%m%d_%H%M%S")}.log'
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger.info("Started Importing Necessary Libraries")

def extract_ribfrac_number(filename):
    base = os.path.basename(filename)
    match = re.search(r'RibFrac(\d+)', base)
    if match:
        return int(match.group(1))
    return None

def find_matching_files(data_files, label_files):
    data_dict = {extract_ribfrac_number(f): f for f in data_files}
    label_dict = {extract_ribfrac_number(f): f for f in label_files}
    
    matched_pairs = []
    for num in data_dict.keys():
        if num in label_dict:
            matched_pairs.append((data_dict[num], label_dict[num]))
        elif num - 1 in label_dict:  # Check for off-by-one match
            matched_pairs.append((data_dict[num], label_dict[num - 1]))
    
    return matched_pairs

def is_valid_pair(data_file, label_file):
    logger.debug(f"Checking pair: {os.path.basename(data_file)} - {os.path.basename(label_file)}")
    
    try:
        label = nib.load(label_file).get_fdata()
        if np.all(label == 0) or np.isnan(label).any() or np.isinf(label).any():
            logger.debug(f"Label file contains invalid data (all zeros, NaNs, or Infs): {label_file}")
            return False
        logger.debug(f"Valid pair: {os.path.basename(data_file)} - {os.path.basename(label_file)}")
        return True
    except Exception as e:
        logger.error(f"Error loading {label_file}: {e}")
        return False

# Custom Dataset Class with Dynamic Filtering
class MedicalDataset(Dataset):

    def __init__(self, data_list, label_list, transform=None):
        logger.info(f"Initializing dataset with {len(data_list)} data files and {len(label_list)} label files")
        
        self.matched_pairs = find_matching_files(data_list, label_list)
        self.valid_pairs = [pair for pair in self.matched_pairs if is_valid_pair(*pair)]
        
        logger.info(f"Total pairs: {len(self.matched_pairs)}, Valid pairs: {len(self.valid_pairs)}")
        
        self.transform = transform

        # Log all valid pairs
        for data, label in self.valid_pairs:
            logger.debug(f"Valid pair: {os.path.basename(data)} - {os.path.basename(label)}")

        # Mean and std for grayscale (1 channel) medical images
        self.mean = torch.tensor([0.485]).view(1, 1, 1, 1)
        self.std = torch.tensor([0.229]).view(1, 1, 1, 1)

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / std
    
    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        if idx >= len(self.valid_pairs):
            raise IndexError(f"Index {idx} out of range for valid_pairs.")
        
        data_file, label_file = self.valid_pairs[idx]

        # Load the data and label
        data = nib.load(data_file).get_fdata()
        label = nib.load(label_file).get_fdata()

        logger.debug(f"File: {os.path.basename(data_file)} - Raw data shape: {data.shape}")
        logger.debug(f"File: {os.path.basename(label_file)} - Raw label shape: {label.shape}")

        # Convert data and label to tensors
        data_tensor = torch.from_numpy(data).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)

        # Normalize tensors
        data_tensor = self.normalize(data_tensor, self.mean, self.std)
        #label_tensor = self.normalize(label_tensor, self.mean, self.std)

        sample = {'data': data_tensor, 'label': label_tensor, 'data_file': data_file, 'label_file': label_file}

        # Apply any transforms (e.g., resizing)
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Transform to resize the data
class ResizeTransform:
    def __init__(self, target_shape=(256, 256, 128)):
        self.target_shape = target_shape

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        data = F.interpolate(data.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        #label = F.interpolate(label.unsqueeze(0), size=self.target_shape, mode='trilinear', align_corners=False).squeeze(0)
        label = F.interpolate(label.unsqueeze(0), size=self.target_shape, mode='nearest').squeeze(0)
        logger.debug(f"Transform data shape: {data.shape}")
        logger.debug(f"Transform label shape: {label.shape}")
        return {'data': data, 'label': label, 'data_file': sample['data_file'], 'label_file': sample['label_file']}

# DataLoader creation function
def create_dataloader(data_list, label_list, transform=None, batch_size=2, shuffle=True, num_workers=8):
    dataset = MedicalDataset(data_list, label_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=True)
    return dataloader

# Define directories
train_data_dir = '/workspace/RibCage/train-ribfrac-defected-new'  # defected rib
train_label_dir = '/workspace/RibCage/train-segmented_ribfrac'  # original rib
val_data_dir = '/workspace/RibCage/val-ribfrac-defected-new'
val_label_dir = '/workspace/RibCage/val-segmented_ribfrac'

# Get list of files
train_data_list = sorted(glob.glob(os.path.join(train_data_dir, '*.nii')) + glob.glob(os.path.join(train_data_dir, '*.nii.gz')))
train_label_list = sorted(glob.glob(os.path.join(train_label_dir, '*.nii')) + glob.glob(os.path.join(train_label_dir, '*.nii.gz')))
val_data_list = sorted(glob.glob(os.path.join(val_data_dir, '*.nii')) + glob.glob(os.path.join(val_data_dir, '*.nii.gz')))
val_label_list = sorted(glob.glob(os.path.join(val_label_dir, '*.nii')) + glob.glob(os.path.join(val_label_dir, '*.nii.gz')))

# Define the transform
resize_transform = ResizeTransform(target_shape=(256, 256, 128))

# Create DataLoader for training and validation
train_loader = create_dataloader(train_data_list, train_label_list, transform=resize_transform, batch_size=2, shuffle=True)
val_loader = create_dataloader(val_data_list, val_label_list, transform=resize_transform, batch_size=2, shuffle=False)

# Log the number of batches in each loader
logger.info(f"Number of batches in train_loader: {len(train_loader)}")
logger.info(f"Number of batches in val_loader: {len(val_loader)}")

# Function to check and log a few samples from a dataloader
def check_dataloader(loader, name):
    logger.info(f"Checking {name} dataloader:")
    for i, batch in enumerate(loader):
        logger.info(f"Batch {i}:")
        for j in range(len(batch['data'])):
            logger.info(f"  Sample {j}:")
            logger.info(f"    Data file: {os.path.basename(batch['data_file'][j])}")
            logger.info(f"    Label file: {os.path.basename(batch['label_file'][j])}")
        if i == 2:  # Check only first 3 batches
            break

# Check both dataloaders
check_dataloader(train_loader, "Training")
check_dataloader(val_loader, "Validation")

# Function to save data and label as NIfTI files
def save_nifti(data_tensor, label_tensor, save_dir, batch_idx, is_train=True):
    mode = 'train' if is_train else 'val'
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays and detach from GPU (if applicable)
    data_np = data_tensor.cpu().numpy().astype(np.float32)  # Convert to NumPy and ensure float32
    label_np = label_tensor.cpu().numpy().astype(np.float32)
    
    # Save the data as NIfTI
    data_filename = os.path.join(save_dir, f'{mode}_data_batch{batch_idx}.nii.gz')
    label_filename = os.path.join(save_dir, f'{mode}_label_batch{batch_idx}.nii.gz')

    data_np=data_np.squeeze(1)
    label_np=label_np.squeeze(1)

    # Since the data shape is (B, 1, D, H, W), we need to remove the singleton dimension (the 1 in the second position)
    #nib.save(nib.Nifti1Image(data_np[0, 0], np.eye(4)), data_filename)  # Save the first image in the batch
    #nib.save(nib.Nifti1Image(label_np[0, 0], np.eye(4)), label_filename)  # Save the first label in the batch
    for i in range(data_np.shape[0]):  # Iterate through the batch
        nib.save(nib.Nifti1Image(data_np[i], np.eye(4)), f'{save_dir}/{mode}_data_batch{batch_idx}_instance_{i}.nii.gz')
        nib.save(nib.Nifti1Image(label_np[i], np.eye(4)), f'{save_dir}/{mode}_label_batch{batch_idx}_instance_{i}.nii.gz')

    print(f"Saved {data_filename}")
    print(f"Saved {label_filename}")

    print(f"Saved {mode}_data_batch{batch_idx}")
    print(f"Saved {mode}_label_batch{batch_idx}")

# Directory where you want to save the NIfTI files
save_dir_train = '/workspace/RibCage/saved_nifti/train'
save_dir_val = '/workspace/RibCage/saved_nifti/val'

# Iterate through the DataLoader and save first two batches from training
print("\nSaving DataLoader for training data...")
for i, batch in enumerate(train_loader):
    data_tensor = batch['data']
    label_tensor = batch['label']
    
    print(f'Batch {i + 1}: Data shape: {data_tensor.shape}, Label shape: {label_tensor.shape}')
    save_nifti(data_tensor, label_tensor, save_dir_train, i, is_train=True)
    
    if i == 1:  # Save only the first two batches
        break

# Iterate through the DataLoader and save first two batches from validation
print("\nSaving DataLoader for validation data...")
for i, batch in enumerate(val_loader):
    data_tensor = batch['data']
    label_tensor = batch['label']
    
    print(f'Batch {i + 1}: Data shape: {data_tensor.shape}, Label shape: {label_tensor.shape}')
    save_nifti(data_tensor, label_tensor, save_dir_val, i, is_train=False)
    
    if i == 1:  # Save only the first two batches
        break

# Model Definition using UNet
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=1,  
    classes=1,
    activation=None,
    encoder_depth=5,
    decoder_channels=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
    decoder_attention_type=None,
    aux_params=None,
    strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))
)

# Set up device and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")
model = model.to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Optional: Mixed Precision Training with GradScaler
scaler = GradScaler()

# Save and Load Checkpoints
def save_checkpoint(state, filename='checkpoint-defected-encoder.pth.tar'):
    torch.save(state, filename)
    logger.info(f"=> Checkpoint saved to '{filename}'")

def load_checkpoint(model, optimizer, filename='checkpoint-defected-encoder.pth.tar'):
    if os.path.isfile(filename):
        logger.info(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"=> Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
        return True
    else:
        logger.info(f"=> No checkpoint found at '{filename}'")
        return False

def save_instance_data(data, outputs, labels, epoch, is_train, instance_idx):
    mode = 'train' if is_train else 'val'
    output_dir = f'/workspace/RibCage/instances/instance_data_epoch_{epoch}/{mode}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays and cast to float32 (which is supported by NIfTI)
    data_np = data.cpu().numpy().astype(np.float32)
    outputs_np = outputs.cpu().detach().numpy().astype(np.float32)
    labels_np = labels.cpu().numpy().astype(np.float32)

    #logger.debug(f"Saving data shape: {data_np.shape}")
    #logger.debug(f"Saving output shape: {outputs_np.shape}")
    #logger.debug(f"Saving labels shape: {labels_np.shape}")

    # Check the shape before saving (expecting 256x256x128)
    expected_shape = (256, 256, 128)
    if data_np.shape[1:] != expected_shape:
        raise ValueError(f"Data shape {data_np.shape[1:]} does not match expected shape {expected_shape}")
    if outputs_np.shape[1:] != expected_shape:
        raise ValueError(f"Outputs shape {outputs_np.shape[1:]} does not match expected shape {expected_shape}")
    if labels_np.shape[1:] != expected_shape:
        raise ValueError(f"Labels shape {labels_np.shape[1:]} does not match expected shape {expected_shape}")

    data_np=data_np.squeeze(0)
    outputs_np=outputs_np.squeeze(0)
    labels_np=labels_np.squeeze(0)

    logger.debug(f"Saving data shape: {data_np.shape}")
    logger.debug(f"Saving output shape: {outputs_np.shape}")
    logger.debug(f"Saving labels shape: {labels_np.shape}")

    

    # Save input data
    nib.save(nib.Nifti1Image(data_np, np.eye(4)), f'{output_dir}/instance_{instance_idx}_input.nii.gz')
    
    # Save model outputs
    nib.save(nib.Nifti1Image(outputs_np, np.eye(4)), f'{output_dir}/instance_{instance_idx}_output.nii.gz')
    
    # Save ground truth labels
    nib.save(nib.Nifti1Image(labels_np, np.eye(4)), f'{output_dir}/instance_{instance_idx}_label.nii.gz')


def save_latent_representation(features, epoch, batch_idx, is_train):
    mode = 'train' if is_train else 'val'
    output_dir = f'/workspace/RibCage/latents/latent_representation_epoch_{epoch}/{mode}'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, feature in enumerate(features):
        # Ensure the feature is in float32 before saving
        feature = feature.cpu().detach().numpy().astype(np.float32)
        
        # Save the latent representation as a NIfTI image, no need to resize
        nib.save(nib.Nifti1Image(feature, np.eye(4)), 
                 f'{output_dir}/batch_{batch_idx}_level_{i}.nii.gz')

def val_model(model, dataloader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['data'].to(device)
            labels = batch['label'].to(device)
            
            # Get encoder output (latent representation)
            features = model.encoder(inputs)

            logger.debug(f"Val encoder output shape: {len(features)}")
            
            # Save latent representation
            save_latent_representation(features, epoch, batch_idx, is_train=False)
            
            # Continue with the rest of the forward pass
            decoder_output = model.decoder(*features)

            logger.debug(f"Val decoder output shape: {decoder_output.shape}")
            
            outputs = model.segmentation_head(decoder_output)

            logger.debug(f"Val output shape: {outputs.shape}")
            
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Save data for each instance in the validation set
            for i in range(inputs.size(0)):
                save_instance_data(inputs[i], outputs[i], labels[i], epoch, False, batch_idx * dataloader.batch_size + i)

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_interval=5, early_stopping_patience=5):
    best_loss = float('inf')
    no_improvement_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['data'].to(device)
            labels = batch['label'].to(device)

            logger.debug(f"Training Epoch {epoch}, Batch {batch_idx} - Input shape: {inputs.shape}, Label shape: {labels.shape}")

            optimizer.zero_grad()

            with autocast():  # Mixed precision
                # Get encoder output (latent representation)
                features = model.encoder(inputs)

                logger.debug(f"Training Epoch {epoch}, Batch {batch_idx} - Train Encoder output shapes: {[f.shape for f in features]}")
                
                # Save latent representation
                save_latent_representation(features, epoch, batch_idx, is_train=True)
                
                # Continue with the rest of the forward pass
                decoder_output = model.decoder(*features)
                outputs = model.segmentation_head(decoder_output)

                logger.debug(f"Training Epoch {epoch}, Batch {batch_idx} - Train Output shape: {outputs.shape}")
                
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

            # Save data for each instance in the training set
            #for i in range(inputs.size(0)):
            #    save_instance_data(inputs[i], outputs[i], labels[i], epoch, True, batch_idx * train_loader.batch_size + i)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss = val_model(model, val_loader, criterion, epoch)
        scheduler.step(val_loss)
        
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Save the best model every five intervals
        if ((epoch + 1) // checkpoint_interval) % 5 == 0:
            if val_loss < best_loss:
                best_loss = val_loss
                no_improvement_counter = 0
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(state, f'/workspace/RibCage/defectedrib-encoder-models/checkpoint_best_interval_{(epoch + 1) // checkpoint_interval}.pth.tar')
            else:
                no_improvement_counter += 1

        # Early stopping
        if no_improvement_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    logger.info('Training and Validation finished!')

logger.info('Training Started')
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=100)