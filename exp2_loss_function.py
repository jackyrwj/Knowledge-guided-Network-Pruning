import logging
import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from model import Model
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.trainers import ClassificationTrainer
from loss_v1 import grad_cam_loss_v1
from loss_v2 import grad_cam_loss_v2

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
    'DEVICE_IDS': [2],
    'WEIGHT_DECAY': 1e-4,
    'LR': 1e-4,
    'LABEL': 'valence',
    'LOSS': 'v1'
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

    def on_training_step(self, train_batch, batch_id,
                         num_batches):
        self.train_accuracy.reset()
        self.train_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        # compute prediction error
        pred = self.modules['model'](X)
        if PARAMS['LOSS'] == 'v1':
            # region-based loss
            loss = self.loss_fn(pred, y) - torch.sigmoid(grad_cam_loss_v1(self.modules['model'], X, y, weight_nroi=0))
        elif PARAMS['LOSS'] == 'v2':
            # sample-based loss
            loss = 10.0 * grad_cam_loss_v2(self.modules['model'], X, y)

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_loss.update(loss)
            self.train_accuracy.update(pred.argmax(1), y)

            train_loss = self.train_loss.compute()
            train_accuracy = 100 * self.train_accuracy.compute()

            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"loss: {train_loss:>8f}, accuracy: {train_accuracy:>0.1f}% [{batch_id:>5d}/{num_batches:>5d}]"
                )


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

        train_loader = DataLoader(train_dataset, batch_size=PARAMS['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=PARAMS['BATCH_SIZE'],
                                shuffle=False)
        # trainer.load_state_dict(os.path.join(PARAMS['PARAM_PATH'], f'2022-09-26-12-01-07.pt'))
        trainer.fit(train_loader, val_loader, num_epochs=PARAMS['EPOCH'])

        trainer.save_state_dict(
            os.path.join(PARAMS['PARAM_PATH'], f'{timeticks}.pt'))
        trainer.test(val_loader)
        # only one fold is okay for temp experiments
        break

    # may need to close the ddp
