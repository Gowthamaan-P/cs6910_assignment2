import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np
from src.network import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data_path = "inaturalist_12K/val/"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            inplace=False),
])

test_data = ImageFolder(root=test_data_path, transform=transform)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

classes = test_dataloader.dataset.classes

config = {
    'number_of_filters': 32,
    'filter_size': 3,
    'stride': 1,
    'padding': 1,
    'max_pooling_size': 2,
    'n_neurons': 512,
    'n_classes': 10,
    'conv_activation': 'relu',
    'dense_activation': 'relu',
    'dropout_rate': 0.2,
    'use_batchnorm': True,
    'factor':1,
    'learning_rate': 1e-5,
    'batch_size':64,
    'epochs':100,
    'use_augmentation': False,
    'dropout_organisation': 3,
    'weight_decay':1e-2
}

model = CNN(
      input_dimension=(3,224,224),
        number_of_filters=config['number_of_filters'],
        filter_size =(config['filter_size'],config['filter_size']),
        stride=config['stride'],
        padding=config['padding'],
        max_pooling_size=(config['max_pooling_size'],config['max_pooling_size']),
        n_neurons=config['n_neurons'],
        n_classes=config['n_classes'],
        conv_activation=nn.ReLU(),
        dense_activation=nn.ReLU(),
        dropout_rate=config['dropout_rate'],
        use_batchnorm=config['use_batchnorm'],
        factor=config['factor'],
        dropout_organisation=config['dropout_organisation']
    ).to(device)
model = torch.load("models/custom_cnn.pt",  map_location=device)
model.eval()

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in test_dataloader:
        (inputs, labels) = (inputs.to(device), labels.to(device))
        output = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# Count the number of correct predictions
correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
# Calculate the accuracy
accuracy = correct_predictions / len(y_true)
print("Overall accuracy:", accuracy)

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')