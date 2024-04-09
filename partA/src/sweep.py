import gc

import torch
import wandb

from src.network import CNN

from torch import nn

from torch.optim import Adam

from torch.utils.data import DataLoader, random_split

from torchvision import transforms

from torchvision.datasets import ImageFolder


def get_transform(use_augmentation):
    if use_augmentation:
        return transforms.Compose([
            transforms.RandomCrop(50, padding=1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 20)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
        ),])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def wandb_sweep():

    run = wandb.init()
    config = wandb.config
    run.name = f"nf_{config['number_of_filters']}_fs_{config['filter_size']}_nn_{config['n_neurons']}_lr_{config['learning_rate']}_bs_{config['batch_size']}_cact_{config['conv_activation']}"

    training_data = ImageFolder(
        root=training_data_path, transform=get_transform(config['use_augmentation'])
    )
    train_size = int(0.8 * len(training_data))
    val_size = len(training_data) - train_size
    train_set, validation_set = random_split(training_data, [train_size, val_size])
    train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=config['batch_size'], shuffle=False)

    training_loss, training_accuracy, validation_loss, validation_accuracy = [], [], [], []

    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'mish': nn.Mish(),
        'relu6': nn.ReLU6(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
    gc.collect()
    torch.cuda.empty_cache()
    model = CNN(
        input_dimension=(3, 224, 224),
        number_of_filters=config['number_of_filters'],
        filter_size=(config['filter_size'], config['filter_size']),
        stride=config['stride'],
        padding=config['padding'],
        max_pooling_size=(config['max_pooling_size'], config['max_pooling_size']),
        n_neurons=config['n_neurons'],
        n_classes=config['n_classes'],
        conv_activation=activations[config['conv_activation']],
        dense_activation=activations[config['dense_activation']],
        dropout_rate=config['dropout_rate'],
        use_batchnorm=config['use_batchnorm'],
        factor=config['factor'],
        dropout_organisation=config['dropout_organisation'],
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, config["epochs"]):

        running_loss, running_accuracy, running_batch = 0, 0, 0
        model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            loss = criterion(pred, y)
            running_loss += loss.item() * x.size()[0]
            running_accuracy += (pred.argmax(1) == y).sum().item()
            running_batch += y.size()[0]
            del x
            del y
            loss.backward()
            optimizer.step()
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

        wandb.log({
            "epochs": epoch + 1,
            "train_loss": training_loss[-1],
            "train_accuracy": training_accuracy[-1],
            "val_loss": validation_loss[-1],
            "val_accuracy": validation_accuracy[-1],
        })
        print(f"Epoch: {epoch+1}/{config['epochs']}")
    del model
    gc.collect()
    torch.cuda.empty_cache()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_path = "inaturalist_12K/train/"

sweep_config = {
    'method': 'bayes',
    'name': 'PART_A_Q2_SWEEP_1',
    'metric': {
        'name': "val_accuracy",
        'goal': 'maximize',
    },
    'parameters': {
        'number_of_filters': {'values': [16, 32, 64, 128, 256]},
        'filter_size': {'value': 3},
        'stride': {'value': 1},
        'padding': {'value': 1},
        'max_pooling_size': {'value': 2},
        'n_neurons': {'values': [64, 128, 256, 512, 1024]},
        'n_classes': {'value': 10},
        'conv_activation': {'values': ['relu', 'gelu', 'silu', 'mish', 'relu6', 'tanh', 'sigmoid']},
        'dense_activation': {
            'values': ['relu', 'gelu', 'silu', 'mish', 'relu6', 'tanh', 'sigmoid']
        },
        'dropout_rate': {'values': [0.2, 0.3, 0.4, 0.5]},
        'use_batchnorm': {'values': [True, False]},
        'factor': {'values': [1, 2, 3, 0.5]},
        'learning_rate': {'values': [1e-2, 1e-3, 1e-4, 1e-5]},
        'batch_size': {'value': 16},
        "epochs": {'values': [5, 10, 15, 20]},
        'use_augmentation': {'values': [True, False]},
        'dropout_organisation': {'values': [1, 2, 3, 4, 5]},
},}


wandb.login(key='API_KEY')

wandb.init(project="PROJECT", entity='ENTITY')

wandb_id = wandb.sweep(sweep_config, project="PROJECT")
wandb.agent(wandb_id, function=wandb_sweep, count=300)

wandb.finish()