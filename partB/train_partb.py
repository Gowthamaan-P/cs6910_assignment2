import argparse
import gc

import torch

from torch import nn

from torch.optim import AdamW

from torch.utils.data import DataLoader, random_split

from torchvision import models, transforms

from torchvision.datasets import ImageFolder


WANDB_PROJECT = "CS6910_AS1"
WANDB_ENTITY = "ed23s037"

network_config = {
    'n_neurons': 1024,
    'n_neurons1': 512,
    'weight_decay': 1e-2,
    'n_classes': 10,
    'learning_rate': 5e-5,
    'batch_size': 32,
    'epochs': 100,
}



parser = argparse.ArgumentParser()
parser.add_argument(
    "-wp",
    "--wandb_project",
    type=str,
    default=WANDB_PROJECT,
    help="Wandb project name",
    required=True,
)
parser.add_argument(
    "-we", "--wandb_entity", type=str, default=WANDB_ENTITY, help="Wandb entity name", required=True
)
parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("-wd", "--weight_decay", type=int, default=1e-2, help="Weight Decay")
parser.add_argument(
    "-nn",
    "--n_neurons",
    type=int,
    default=1024,
    help="Number of neurons in the fully connected layer 1",
)
parser.add_argument(
    "-nn1",
    "--n_neurons1",
    type=int,
    default=512,
    help="Number of neurons in the fully connected layer 2",
)
parser.add_argument("-nc", "--n_classes", type=int, default=10, help="Number of classes")


def train():
    training_loss, training_accuracy, validation_loss, validation_accuracy = [], [], [], []
    gc.collect()
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.fc.parameters(),
        lr=network_config['learning_rate'],
        weight_decay=network_config['weight_decay'],
    )
    for epoch in range(0, network_config['epochs']):
        running_loss, running_accuracy, running_batch = 0, 0, 0
        model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = criterion(pred.logits, y)
            running_loss += loss.item() * x.size()[0]
            running_accuracy += (pred.logits.argmax(1) == y).sum().item()
            running_batch += y.size()[0]
            loss.backward()
            optimizer.step()
            del x
            del y
        training_loss.append(running_loss / len(train_dataloader.dataset))
        training_accuracy.append(100 * (running_accuracy / running_batch))

        running_loss, running_accuracy, running_batch = 0, 0, 0
        with torch.no_grad():
            model.eval()
            for x, y in val_dataloader:
                (x, y) = (x.to(device), y.to(device))
                pred = model(x)
                loss = criterion(pred, y)
                running_loss += loss.item() * x.size()[0]
                running_accuracy += (pred.argmax(1) == y).sum().item()
                running_batch += y.size()[0]
                del x
                del y
        validation_loss.append(running_loss / len(val_dataloader.dataset))
        validation_accuracy.append(100 * (running_accuracy / running_batch))
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Epoch: {epoch+1}/{network_config['epochs']}")
    return training_loss, training_accuracy, validation_loss, validation_accuracy


args = parser.parse_args()
network_config.update(vars(args))
print("Parameters:")
for key, value in network_config.items():
    print(f"{key}: {value}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_path = "inaturalist_12K/train/"
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False),
])

training_data = ImageFolder(root=training_data_path, transform=transform)
train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
train_set, validation_set = random_split(training_data, [train_size, val_size])
train_dataloader = DataLoader(train_set, batch_size=network_config['batch_size'], shuffle=True)
val_dataloader = DataLoader(validation_set, batch_size=network_config['batch_size'], shuffle=False)
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, init_weights=False)
print(model)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, network_config['n_neurons']),
    nn.ReLU(),
    nn.Linear(network_config['n_neurons'], network_config['n_neurons1']),
    nn.ReLU(),
    nn.Linear(network_config['n_neurons1'], network_config['n_classes']),
)

print(model)
for name, param in model.named_parameters():
    if "fc" in name or "Mixed_7c":
        param.requires_grad = True
    else:
        param.requires_grad = False
model = model.to(device)
training_loss, training_accuracy, validation_loss, validation_accuracy = train()
network_config['epochs'] = 5
for name, param in model.named_parameters():
    if "Mixed_7b" in name:
        param.requires_grad = True
training_loss, training_accuracy, validation_loss, validation_accuracy = train()
torch.save(model, "models/model.h5")