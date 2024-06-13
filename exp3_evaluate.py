import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data.dataloader import DataLoader

from model import Model
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.trainers import ClassificationTrainer

PARAMS = {
    'IO_PATH': './tmp_out/xai/deap',
    'ROOT_PATH': './tmp_in/data_preprocessed_python',
    'LOG_PATH': './tmp_out/xai_log/',
    'PARAM_PATH': './tmp_out/xai_log/',
    'SPLIT_PATH': './tmp_out/xai/split',
    'NUM_CLASSES': 2,
    'BATCH_SIZE': 256,
    'EPOCH': 50,
    'KFOLD': 10,
    'DEVICE_IDS': [1],
    'WEIGHT_DECAY': 1e-4,
    'LR': 1e-4,
    'LABEL': 'valence'
}

logger = logging.getLogger('XAI')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler(
    os.path.join(PARAMS['LOG_PATH'], f'{timeticks}.log'))
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(PARAMS)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        exp_y = self.fc(y).view(b, c, 1, 1)
        exp_y = exp_y.expand_as(x)
        return x * exp_y, y


class MyModel(nn.Module):
    def __init__(self, num_classes, sample_rate=128):
        super(MyModel, self).__init__()
        self.num_classes = num_classes
        self.sample_rate = sample_rate

        self.conv1 = nn.Conv2d(sample_rate,
                               128,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1)

        # modify here
        self.se_layer = SELayer(128)

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

        # modify here
        conv1_selu1_out, _ = self.se_layer(conv1_selu1_out)

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

    def forward_with_attention(self, x):
        conv1_selu1_out = self.selu1(self.conv1(x))

        # modify here
        conv1_selu1_out, channel_attention = self.se_layer(conv1_selu1_out)

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

        return fc3_out, channel_attention


class MyClassificationTrainer(ClassificationTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)

    def before_test_epoch(self):
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(
            num_classes=PARAMS['NUM_CLASSES']).to(self.device)
        self.test_auroc = torchmetrics.AUROC(
            num_classes=PARAMS['NUM_CLASSES']).to(self.device)
        self.test_confusion_matrix.reset()
        self.test_auroc.reset()
        self.test_loss.reset()
        self.test_accuracy.reset()

    def on_test_step(self, test_batch, batch_id, num_batches):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)
        pred = self.modules['model'](X)

        self.test_loss.update(self.loss_fn(pred, y))
        self.test_accuracy.update(pred.argmax(1), y)
        self.test_confusion_matrix.update(pred.argmax(1), y)
        self.test_auroc.update(pred, y)

    def after_test_epoch(self):
        test_accuracy = 100 * self.test_accuracy.compute()
        test_loss = self.test_loss.compute()
        test_confusion_matrix = self.test_confusion_matrix.compute()
        test_auroc = 100 * self.test_auroc.compute()
        self.log(f"\nloss: {test_loss:>8f}, accuracy: {test_accuracy:>0.1f}%")
        self.log(test_confusion_matrix)
        self.log(f"Area under ROC: {test_auroc:>0.1f}%")


if __name__ == "__main__":
    os.makedirs("./tmp_out/xai", exist_ok=True)

    dataset = DEAPDataset(
        io_path=PARAMS['IO_PATH'],
        root_path=PARAMS['ROOT_PATH'],
        offline_transform=transforms.Compose([
            transforms.BaselineRemoval(),
            transforms.MeanStdNormalize(),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ]),
        online_transform=transforms.Compose([transforms.ToTensor()]),
        label_transform=transforms.Compose(
            [transforms.Select(PARAMS['LABEL']),
             transforms.Binary(5.0)]),
        num_worker=16)

    k_fold = KFoldGroupbyTrial(n_splits=PARAMS['KFOLD'],
                               split_path=PARAMS['SPLIT_PATH'])

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        model = MyModel(num_classes=PARAMS['NUM_CLASSES'])
        trainer = MyClassificationTrainer(model=model,
                                          lr=PARAMS['LR'],
                                          weight_decay=PARAMS['WEIGHT_DECAY'],
                                          device_ids=PARAMS['DEVICE_IDS'])

        train_loader = DataLoader(train_dataset,
                                  batch_size=PARAMS['BATCH_SIZE'],
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=PARAMS['BATCH_SIZE'],
                                shuffle=False)

        trainer.load_state_dict(
            os.path.join(PARAMS['PARAM_PATH'], f'2022-09-28-22-22-03.pt'))
        trainer.test(val_loader)
        break

    # may need to close the ddp

# loss: 0.436341, accuracy: 92.0%
# tensor([[3020,  316],
#         [ 295, 4049]], device='cuda:1')
# Area under ROC: 81.3%