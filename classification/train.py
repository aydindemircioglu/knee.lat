#
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import os
import random
import optuna
import dill

import torch
from torchsummary import summary
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as tf
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import torchvision.models as models
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from torchsummary import summary
from pytorch_lightning import seed_everything

from datasets import *
from model import KneeModel
from utils import recreatePath


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # only parameters for tuning
        self.parser.add_argument('--lr', type=float, default=1e-20, help='initial learning rate for adam')
        self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
        self.parser.add_argument('--loss', type=str, default="CE", help='CE or focal')

        # CV
        self.parser.add_argument('--fold', type=str, default="val", help='fold for CV')
        self.parser.add_argument('--cv', type=int, default=10, help='n for CV')

        # input/output sizes
        self.parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
        self.parser.add_argument('--gpus', type=str, default="0", help='list of gpus to run on')
        self.parser.add_argument('--model', type=str, default="resnet18", help='model for training')

        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)

        if self.opt.fold == "all":
            self.opt.fold = "all"
        else:
            try:
                self.opt.fold = int(self.opt.fold)
            except:
                raise Exception ("Fold must be numeric.")
            # just for here
            if self.opt.fold < 0 or self.opt.fold > self.opt.cv:
                raise Exception (str(self.opt.cv) + "-fold CV needs fold between 0 and " + str(self.opt.cv-1))


        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt



if __name__ == '__main__':
    # parse args
    opt = BaseOptions().parse()

    # init model
    seed_everything(571)

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    early_stop_callback = EarlyStopping(
       monitor='val_loss',
       min_delta=0.01,
       patience=25,
       verbose=False,
       mode='min'
    )

    model =opt.model
    basePath = os.path.join('/data/uke/data/knee.lat/checkpoints/' , str(model), str(opt.lr), str(opt.loss))
    foldPath = os.path.join(basePath, str(opt.fold))
    recreatePath (foldPath)

    # for final training we only need the best model
    top_k = 5
    if opt.fold == "all":
        top_k = 1

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= foldPath,
        filename='{epoch:02d}-{val_loss:.2f}' + "_lr_" + str(opt.lr) + "_" + str(opt.loss),
        save_top_k=top_k,
        mode='min',
    )

    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=opt.gpus, max_epochs = 30,
                                        accumulate_grad_batches=5,
                                        stochastic_weight_avg=True,
                                        deterministic=True)

    # Run lr finder
    model = KneeModel(learning_rate = float(opt.lr), model = model, loss = opt.loss)
    dm = KneeDataModule (batch_size = 32, subset = False, fold = opt.fold, nCV = opt.cv)
    trainer.fit(model, dm)


#
