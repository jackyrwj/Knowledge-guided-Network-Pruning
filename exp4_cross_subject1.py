import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from model import Model
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import LeaveOneSubjectOut
from torcheeg.trainers import CORALTrainer

PARAMS = {
    'IO_PATH': './tmp_out/xai/deap',
    'ROOT_PATH': './tmp_in/data_preprocessed_python',
    'LOG_PATH': './tmp_out/xai_log/',
    'PARAM_PATH': './tmp_out/xai_log/',
    'SPLIT_PATH': './tmp_out/xai/loso_split',
    'NUM_CLASSES': 2,
    'BATCH_SIZE': 256,
    'EPOCH': 50,
    'DEVICE_IDS': [0],
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


class MyCORALTrainer(CORALTrainer):
    def print(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)


class Extractor(Model):
    def forward(self, x):
        conv1_selu1_out = self.selu1(self.conv1(x))
        conv2_selu2_out = self.selu2(self.conv2(conv1_selu1_out))
        conv3_selu3_out = self.selu3(self.conv3(conv2_selu2_out))
        conv4_selu4_out = self.selu4(self.conv4(conv3_selu3_out))

        x = conv4_selu4_out.view(-1, 512 * 9)
        return x


class Classifier(Model):
    def forward(self, x):
        fc1_out = self.fc1(x)
        fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.dropout(fc2_out)
        fc3_out = self.fc3(fc2_out)
        aft_soft = self.soft(fc3_out)

        return aft_soft


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

    k_fold = LeaveOneSubjectOut(split_path=PARAMS['SPLIT_PATH'])

    scores = {}

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        extractor = Extractor(num_classes=PARAMS['NUM_CLASSES'])
        classifier = Classifier(num_classes=PARAMS['NUM_CLASSES'])
        trainer = MyCORALTrainer(extractor=extractor,
                                 classifier=classifier,
                                 lr=PARAMS['LR'],
                                 weight_decay=PARAMS['WEIGHT_DECAY'],
                                 device_ids=PARAMS['DEVICE_IDS'])

        train_loader = DataLoader(train_dataset, batch_size=PARAMS['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=PARAMS['BATCH_SIZE'],
                                shuffle=False)
        # trainer.save_state_dict(
        #     os.path.join(PARAMS['PARAM_PATH'], f'{timeticks}.pt'))
        # trainer.load_state_dict(os.path.join(PARAMS['PARAM_PATH'], f'{timeticks}.pt'))
        trainer.fit(train_loader, val_loader, val_loader, num_epochs=PARAMS['EPOCH'])
        scores[i] = trainer.score(val_loader)

        trainer.save_state_dict(
            os.path.join(PARAMS['PARAM_PATH'], f'{timeticks}.pt'))
        print(scores[i])
        # only one fold is okay for temp experiments
        break

    # may need to close the ddp
