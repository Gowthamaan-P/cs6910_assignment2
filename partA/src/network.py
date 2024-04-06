from torch import nn

class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        number_of_filters: int,
        filter_size: tuple,
        stride: int,
        padding: int,
        max_pooling_size: tuple,
        n_neurons: tuple,
        n_classes: int,
        conv_activation: nn.Module,
        dense_activation: nn.Module,
    ):
        super().__init__()      
        
        self.conv1 = nn.Conv2d(
            in_channels, number_of_filters, kernel_size=filter_size, stride=stride, padding=padding
        )
        self.c_act1 = conv_activation
        self.pool1 = nn.MaxPool2d(kernel_size=max_pooling_size)

        self.conv2 = nn.Conv2d(
            number_of_filters,
            number_of_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
        )
        self.c_act2 = conv_activation
        self.pool2 = nn.MaxPool2d(kernel_size=max_pooling_size)

        self.conv3 = nn.Conv2d(
            number_of_filters,
            number_of_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
        )
        self.c_act3 = conv_activation
        self.pool3 = nn.MaxPool2d(kernel_size=max_pooling_size)

        self.conv4 = nn.Conv2d(
            number_of_filters,
            number_of_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
        )
        self.c_act4 = conv_activation
        self.pool4 = nn.MaxPool2d(kernel_size=max_pooling_size)

        self.conv5 = nn.Conv2d(
            number_of_filters,
            number_of_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
        )
        self.c_act5 = conv_activation
        self.pool5 = nn.MaxPool2d(kernel_size=max_pooling_size)

        self.fc1 = nn.Linear(in_features=n_neurons[0], out_features=n_neurons[1])
        self.d_act1 = dense_activation

        self.fc2 = nn.Linear(in_features=n_neurons[1], out_features=n_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    @staticmethod
    def forward(self, x):
        r = self.conv1(x)
        
        r = self.c_act1(r)
        r = self.pool1(r)

        r = self.conv2(r)
        r = self.c_act2(r)
        r = self.pool2(r)

        r = self.conv3(r)
        r = self.c_act3(r)
        r = self.pool3(r)
        
        r = self.conv4(r)
        r = self.c_act4(r)
        r = self.pool4(r)

        r = self.conv5(r)
        r = self.c_act5(r)
        r = self.pool5(r)
        
        r = self.fc1(r)
        r = self.d_act1(r)

        r = self.fc2(r)
        return self.logSoftmax(r)