import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.fc1 = nn.Linear(in_features=800, out_features=500)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(in_features=500, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        pass