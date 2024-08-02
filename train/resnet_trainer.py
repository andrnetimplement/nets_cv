from model.resnet import ResNet, resnet50

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms, models
from torch.nn.parallel import DataParallel

import json
import os

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, filename='resnet_models/training.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 100 == 0:
            logging.info(f'Train Epoch: {epoch} [{i * len(inputs)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
            print(f'Train Epoch: {epoch} [{i * len(inputs)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    logging.info(f'Accuracy on test set: {100 * correct / total:.2f}%')
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    
BATCH_SIZE = 1536
num_epochs = 21
num_workers = 30
# Путь к датасету
data_dir = 'imagenet/ILSVRC/Data/CLS-LOC/train'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform) 
criterion = torch.nn.CrossEntropyLoss()

start_epoch = 0
checkpoint_path = 'resnet_models/model50_checkpoint.pth'


if os.path.exists(checkpoint_path):
    with open('resnet_models/class_to_idx.json', 'r') as f:
        class_to_idx = json.load(f)
    full_dataset.class_to_idx = class_to_idx
    LEARNING_RATE = 0.1 # start from 0.1
    
    model50 = resnet50()
    model50.to(device)
    
    optimizer = torch.optim.Adam(model50.parameters(), lr=0.01)
    model50, optimizer, start_epoch = load_checkpoint(checkpoint_path, model50, optimizer)
    
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model50 = nn.DataParallel(model50)
   
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE
        
    print(f"Resuming from epoch {start_epoch}, lr: {optimizer.param_groups[0]['lr']}")
else:
    
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    
    model50 = resnet50()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model50 = nn.DataParallel(model50)
    model50.to(device)
    optimizer = torch.optim.Adam(model50.parameters(), lr=0.1)
    
    print("Starting training from scratch")
    
with open('resnet_models/class_to_idx.json', 'w') as f:
    json.dump(full_dataset.class_to_idx, f)
    
for epoch in range(start_epoch, num_epochs):
    train(model50, device, train_loader, optimizer, epoch)
    if epoch % 5 == 0:
        save_checkpoint(model50, optimizer, epoch + 1, checkpoint_path) 