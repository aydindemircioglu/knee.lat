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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchsummary import summary
from pytorch_lightning import seed_everything

from datasets import *
from losses import FocalLoss


class KneeModel (pl.LightningModule):
    def __init__(self, learning_rate = 3e-4, imgSize = 224, model = "resnet", loss = "CE"):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = loss
        if loss != "CE" and loss != "focal":
            raise Exception ("Unknown loss function")

        print ("Resnet:", model)
        backbone = eval("models."+model+"(pretrained=True)")
        num_filters = backbone.fc.in_features
        backbone_layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*backbone_layers)

        try:
            doSummary = False
            if doSummary == True:
                summary(self.feature_extractor.cuda(), input_size=(3, imgSize, imgSize))
        except Exception as e:
            print(e)

        num_target_classes = 2
        self.classifier = nn.Sequential( nn.Dropout(0.1),
            nn.Linear(num_filters, num_filters//4),
            nn.ReLU(),
            nn.Linear(num_filters//4, num_target_classes))



    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x)
            representations = representations.flatten(1)
            o = self.classifier(representations)
        return o


    def training_step(self, batch, batch_idx):
        x, y, imgPath = batch
        z = self.feature_extractor(x).flatten(1)
        y_hat = self.classifier(z)

        if self.loss == "focal":
            fl = FocalLoss ()
        if self.loss == "CE":
            fl =  nn.CrossEntropyLoss()
        loss = fl (y_hat, y)

        # Logging to TensorBoard by default
        self.log('train_loss', loss, prog_bar = True)
        loss.requres_grad = True
        return loss


    def validation_step(self, batch, batch_idx):
        from losses import FocalLoss
        x, y, imgPath = batch
        y_hat = self.forward(x)

        if self.loss == "focal":
            fl = FocalLoss ()
        if self.loss == "CE":
            fl =  nn.CrossEntropyLoss()

        loss = fl (y_hat, y)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        print ("Setting learning rate to", self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer,  "monitor": "val_loss"}


#
