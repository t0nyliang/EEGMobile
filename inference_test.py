from models.EEGViT_pretrained import EEGViT_pretrained
from models.EEGViT import EEGViT_raw
from models.ViTBase_pretrained import ViTBase_pretrained
from models.EEGVit_TCNet import EEGVit_TCN
from models.EEGMobileVitV2_TCNet import EEGMobileVit_TCN

from helper_functions import split
from dataset.EEGEyeNet import EEGEyeNetDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import time

# model to test likely: moblevit, mobilevitv2, eegvit_tcn, eegvit (maybe)
model = EEGMobileVit_TCN()
model.load_state_dict(torch.load("weights_cur.pt"), strict=False)
EEGEyeNet = EEGEyeNetDataset('./dataset/Position_task_with_dots_synchronised_min.npz')

batch_size = 64
n_rounds = 10

def inference_test(model, model_name):
	
    torch.cuda.empty_cache()
    train_indices, val_indices, test_indices = split(EEGEyeNet.trainY[:,0],0.7,0.15,0.15)  # indices for the training set
    print('create dataloader...')
    
    train = Subset(EEGEyeNet, indices=train_indices)
    val = Subset(EEGEyeNet,indices=val_indices)
    test = Subset(EEGEyeNet,indices=test_indices)
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    if torch.cuda.is_available():
        gpu_id = 0  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(student_model)  # Wrap the model with DataParallel
    print("HI")

    model = model.to(device)

    model.eval()
    runtimes = []
    for i in range(0, 5):
	start_time = time.time()
        for round in tqdm(range(n_rounds)):
            with torch.no_grad():

                for inputs, targets, index in train_loader:
                    # Move the inputs and targets to the GPU (if available)
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Compute the outputs and loss for the current batch
                    outputs = model(inputs)

                for inputs, targets, index in val_loader:
                    # Move the inputs and targets to the GPU (if available)
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Compute the outputs and loss for the current batch
                    outputs = model(inputs)

                for inputs, targets, index in test_loader:
                    # Move the inputs and targets to the GPU (if available)
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Compute the outputs and loss for the current batch
                    outputs = model(inputs)

        total_runtime = (time.time() - start_time) / 60
        print(total_runtime, "mins")
        runtimes.append(total_runtime)
    print(runtimes, "runtimes")
if __name__ == "__main__":
    inference_test(model, "name")
