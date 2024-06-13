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


class MyClassificationTrainer(ClassificationTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)

    def before_test_epoch(self):
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(
            num_classes=PARAMS['NUM_CLASSES']).to(self.device)
        self.test_auroc = torchmetrics.AUROC(num_classes=PARAMS['NUM_CLASSES']).to(self.device)
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
        model = Model(num_classes=PARAMS['NUM_CLASSES'])
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
            os.path.join(PARAMS['PARAM_PATH'], f'2022-09-26-12-01-07.pt'))
        trainer.test(val_loader)
        break

    # may need to close the ddp

# loss: 0.952343, accuracy: 91.0%
# tensor([[2953,  383],
#         [ 308, 4036]], device='cuda:1')
# Area under ROC: 96.8%