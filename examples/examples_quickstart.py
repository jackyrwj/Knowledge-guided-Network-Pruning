import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import DEAPDataset
from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import CCNN
from torcheeg.trainers import ClassificationTrainer

if __name__ == "__main__":
    os.makedirs("./tmp_out/examples_quickstart", exist_ok=True)

    dataset = DEAPDataset(
        io_path=f'./tmp_out/examples_quickstart/deap',
        root_path='./tmp_in/data_preprocessed_python',
        offline_transform=transforms.Compose([
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT,
                              apply_to_baseline=True)
        ]),
        online_transform=transforms.Compose(
            [transforms.BaselineRemoval(),
             transforms.ToTensor()]),
        label_transform=transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0),
        ]),
        num_worker=16)

    k_fold = KFoldGroupbyTrial(
        n_splits=5, split_path=f'./tmp_out/examples_quickstart/split')

    scores = {}

    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):

        model = CCNN()
        trainer = ClassificationTrainer(model=model,
                              lr=1e-4,
                              weight_decay=1e-4,
                              device_ids=[0])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        trainer.fit(train_loader, val_loader, num_epochs=50)
        scores[i] = trainer.score(val_loader)
