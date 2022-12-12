import torch.nn as nn


class AlexNet(nn.Module):
    """_summary_
    input size: 227*227
    """
    def __init__(self, n_classes):
        super(AlexNet, self).__init__()
        
        # conv layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96,  256,  kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256,  384,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,  384,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,  256,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes)
        )
        
    def forward(self, x):
        
        # conv layers
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x       