import argparse
import torch
from src.network import CNN
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

WANDB_PROJECT = "CS6910_AS1"
WANDB_ENTITY = "ed23s037"

network_config = {
    'number_of_filters': 16,
    'filter_size': 3,
    'stride': 1,
    'padding': 1,
    'max_pooling_size': 2,
    'n_neurons': 512,
    'n_classes': 10,
    'conv_activation': "relu",
    'dense_activation': "relu6",
    'dropout_rate': 0.2,
    'use_batchnorm': True,
    'factor': 1,
    'learning_rate': 1e-4,
    'batch_size': 16,
    'epochs': 10,
    'use_augmentation': False,
    'dropout_organisation': 3,
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
parser.add_argument(
    "-nf", "--number_of_filters", type=int, default=16, help="Number of filters in the first layer"
)
parser.add_argument(
    "-fs",
    "--filter_size",
    type=int,
    default=3,
    help="Size of the filter (Assuming squared filter fs x fs)",
)
parser.add_argument(
    "-ps",
    "--max_pooling_size",
    type=int,
    default=2,
    help="Size of the max pooling (Assuming squared kernel ps x ps)",
)
parser.add_argument("-stride", "--stride", type=int, default=1, help="Stride")
parser.add_argument("-padding", "--padding", type=float, default=1e-8, help="Padding")
parser.add_argument("-dr", "--dropout_rate", type=float, default=0.2, help="Dropout rate")
parser.add_argument(
    "-ca",
    "--conv_activation",
    type=str,
    default="relu",
    help="Activation in Convolution Block=['relu','gelu','silu','mish','relu6','tanh','sigmoid']",
)
parser.add_argument(
    "-da",
    "--dense_activation",
    type=str,
    default="relu6",
    help="Activation in Dense/FC Block=['relu','gelu','silu','mish','relu6','tanh','sigmoid']",
)
parser.add_argument(
    "-nn",
    "--n_neurons",
    type=int,
    default=512,
    help="Number of neurons in the fully connected layer",
)
parser.add_argument("-nc", "--n_classes", type=int, default=10, help="Number of classes")
parser.add_argument(
    "-fo",
    "--factor",
    type=int,
    default=1,
    help="Filter organisation (Doubling, tripling or havling in every layer)",
)

parser.add_argument(
    "-do", "--dropout_organisation", type=int, default=3, help="Number of dropout layer [1,2,3,4,5]"
)

parser.add_argument(
    "-us", "--use_batchnorm", type=bool, default=True, help="Use Batch Normalization"
)

parser.add_argument(
    "-ua", "--use_augmentation", type=bool, default=False, help="Use Data Augmentation"
)


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


def train(config):

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
        "relu": nn.ReLU(),
        'gelu': nn.GELU(),
        'silu': nn.SiLU(),
        'mish': nn.Mish(),
        "relu6": nn.ReLU6(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }
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

    for epoch in range(0, config['epochs']):

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
        validation_loss.append(running_loss / len(val_dataloader.dataset))
        validation_accuracy.append(100 * (running_accuracy / running_batch))

        print(f"Epoch: {epoch+1}/{config['epochs']}")
        print(validation_loss)
        print(validation_accuracy)
        torch.cuda.empty_cache()
    return model, training_loss, training_accuracy, validation_loss, validation_accuracy



args = parser.parse_args()
network_config.update(vars(args))

# Print the parameters
print("Parameters:")
for key, value in network_config.items():
    print(f"{key}: {value}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_data_path = "inaturalist_12K/train/"

model, training_loss, training_accuracy, validation_loss, validation_accuracy = train(
    network_config
)


torch.save(model, "models/model.h5")