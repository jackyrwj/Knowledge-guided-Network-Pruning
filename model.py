import torch
import torch.nn as nn
from torch.serialization import load


class Model(nn.Module):
    def __init__(self, num_classes, sample_rate=128):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.sample_rate = sample_rate

        self.conv1 = nn.Conv2d(sample_rate,
                               128,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(128,
                               256,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1)
        self.conv3 = nn.Conv2d(256,
                               256,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)
        self.conv4 = nn.Conv2d(256,
                               512,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1)

        self.selu1 = nn.SELU()
        self.selu2 = nn.SELU()
        self.selu3 = nn.SELU()
        self.selu4 = nn.SELU()

        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(512 * 9, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        # self.soft = nn.Softmax(dim=1)

        self.init()

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        conv1_selu1_out = self.selu1(self.conv1(x))
        conv2_selu2_out = self.selu2(self.conv2(conv1_selu1_out))
        conv3_selu3_out = self.selu3(self.conv3(conv2_selu2_out))
        conv4_selu4_out = self.selu4(self.conv4(conv3_selu3_out))

        x = conv4_selu4_out.view(-1, 512 * 9)

        fc1_out = self.fc1(x)
        fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.dropout(fc2_out)
        fc3_out = self.fc3(fc2_out)
        # aft_soft = self.soft(fc3_out)

        return fc3_out