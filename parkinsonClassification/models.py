import torch.nn as nn
from torch.nn import init

class Model0(nn.Module):
    # ------------------------
    # Arquitectura 0. Se ha probado y hay overfitting
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 8, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=32, out_features=16),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=16, out_features=8),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=8, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

# ------------- 3 capas de convolución -----------------
class Model1(nn.Module):
    # ------------------------
    # Arquitectura 1. Out_channel1 = 2^4
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 16, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(2)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(2)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(2),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=64, out_features=32),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=32, out_features=16),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=16, out_features=2),
               nn.Softmax(dim=1)]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

class Model2(nn.Module):
    # ------------------------
    # Arquitectura 2. Out_channel1 = 2^5
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 32, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=128, out_features=64),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=64, out_features=32),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=32, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

class Model3(nn.Module):
    # ------------------------
    # Arquitectura 3. Out_channel1 = 2^6
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 64, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=256, out_features=128),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=128, out_features=64),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=64, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

class Model4(nn.Module):
    # ------------------------
    # Arquitectura 3. Out_channel1 = 2^7
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 128, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=512, out_features=256),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=256, out_features=128),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=128, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

# ------------- 4 capas de convolución -----------------
class Model5(nn.Module):
    # ------------------------
    # Arquitectura 5. Out_channel1 = 2^4
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 16, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        conv4 = [nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv4[0].weight, a=0.1)

        self.conv4 = nn.Sequential(*conv4)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=128, out_features=64),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=64, out_features=32),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=32, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

class Model6(nn.Module):
    # ------------------------
    # Arquitectura 6. Out_channel1 = 2^5
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 32, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        conv4 = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv4[0].weight, a=0.1)

        self.conv4 = nn.Sequential(*conv4)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=256, out_features=128),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=128, out_features=64),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=64, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

class Model7(nn.Module):
    # ------------------------
    # Arquitectura 7. Out_channel1 = 2^7
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 128, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        conv4 = [nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv4[0].weight, a=0.1)

        self.conv4 = nn.Sequential(*conv4)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=1024, out_features=512),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=256, out_features=124),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=124, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

# ------------- 5 capas de convolución -----------------
class Model8(nn.Module):
    # ------------------------
    # Arquitectura 8. Out_channel1 = 2^4
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 16, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        conv4 = [nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv4[0].weight, a=0.1)

        self.conv4 = nn.Sequential(*conv4)

        conv5 = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv5[0].weight, a=0.1)

        self.conv5 = nn.Sequential(*conv4)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=128, out_features=64),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=64, out_features=32),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=32, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

class Model9(nn.Module):
    # ------------------------
    # Arquitectura 9. Out_channel1 = 2^5
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 32, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        conv4 = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv4[0].weight, a=0.1)

        self.conv4 = nn.Sequential(*conv4)

        conv5 = [nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv5[0].weight, a=0.1)

        self.conv5 = nn.Sequential(*conv4)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=512, out_features=256),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=256, out_features=128),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=128, out_features=2),
               nn.Softmax()]

        init.kaiming_normal_(mlp[0].weight, a=0.1)
        init.kaiming_normal_(mlp[3].weight, a=0.1)
        init.kaiming_normal_(mlp[6].weight, a=0.1)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x

class Model10(nn.Module):
    # ------------------------
    # Arquitectura 10. Out_channel1 = 2^6
    # ------------------------
    def __init__(self):
        super().__init__()

        conv1 = [nn.Conv2d(2, 64, kernel_size=(7, 5), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv1[0].weight, a=0.1)

        self.conv1 = nn.Sequential(*conv1)

        conv2 = [nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3)]
        init.kaiming_normal_(conv2[0].weight, a=0.1)

        self.conv2 = nn.Sequential(*conv2)

        conv3 = [nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv3[0].weight, a=0.1)

        self.conv3 = nn.Sequential(*conv3)

        conv4 = [nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv4[0].weight, a=0.1)

        self.conv4 = nn.Sequential(*conv4)

        conv5 = [nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)),
                 nn.ReLU(),
                 nn.MaxPool2d(3),
                 nn.Dropout(0.5)]
        init.kaiming_normal_(conv5[0].weight, a=0.1)

        self.conv5 = nn.Sequential(*conv4)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        mlp = [nn.Linear(in_features=1024, out_features=512),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=512, out_features=256),
               nn.Tanh(),
               nn.Dropout(0.5),
               nn.Linear(in_features=256, out_features=2),
               nn.Softmax()]

        init.xavier_uniform_(mlp[0].weight, gain=5/3)
        init.xavier_uniform_(mlp[3].weight, gain=5/3)
        init.xavier_normal_(mlp[6].weight)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        # El dato x recorre toda la arquitectura
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)

        return x


# Esta es la arquitectura con adición del sexo. Hay que modificar
# train, test y val si se quiere usar.
"""   
    def forward(self, x1, x2):
      # Ejecuta los bloques de convolución sobre
      # la imagen
      # Ejecuta los bloques de convolución
      x1 = self.conv(x1)

      # Mezcla adaptativa y combinación para entrada del clasificador lineal
      x1 = self.ap(x1)

      # Combinamos los datos
      merged = torch.cat((x1.view(x1.size(0), -1),
                          x2.view(x2.size(0), -1)), dim=1)


      # MLP
      x = nn.GELU()(self.inputMLP(merged))
      x = nn.GELU()(self.hiddenMLP(x))
      x = self.outputMLP(x)

      # Salida final
      return x
"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
