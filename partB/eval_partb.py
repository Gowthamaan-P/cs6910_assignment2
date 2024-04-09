import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data_path = "inaturalist_12K/val/"

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                            inplace=False),
])

test_data = ImageFolder(root=test_data_path, transform=transform)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

classes = test_dataloader.dataset.classes

model = models.inception_v3(weights=None, init_weights=False).to(device)
model = torch.load("models/inceptionv3.pt",  map_location=device)
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