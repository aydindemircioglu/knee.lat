import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
import time
from typing import Optional
from sklearn.model_selection import KFold

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


class KneeDataset (Dataset):
    def __init__(self, split, csvFile = None, transform = None, subset = True):
        self.split = split
        if csvFile is None:
            raise Exception ("CSV File needed")
        self.csvFile = csvFile
        df = pd.read_csv(self.csvFile)

        self.y = np.asarray(df["Laterality"] == "L", dtype = np.int64)
        self.img_paths = df['ImagePath']

        # dummy for now
        self.transform = transform
        self.df = df

    def __len__(self):
       return len(self.df)


    def __getitem__(self, index):
        imgPath = str(self.img_paths[index])
        img = cv2.imread(imgPath)

        if self.transform is not None:
            img = self.transform(image=img)
            img = img["image"]

        label = self.y[index]

        return img, label, imgPath



class KneeDataModule (pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers = 16, subset = True, fold = "all", nCV = 10):
        super().__init__()

        self.nCV = nCV
        self.batch_size = batch_size
        self.subset = subset
        self.num_workers = num_workers
        self.rSize = 256
        self.size = self.rSize - self.rSize//8

        self.train_transforms = A.Compose([
            A.Resize (self.rSize+self.rSize//2, self.rSize),
            A.RandomCrop(self.size+self.size//2, self.size),
        # Pixels
            A.OneOf([
                A.CoarseDropout(max_height = self.rSize//10, max_width = self.rSize//10),
                A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1.0),
                A.RandomGamma(p=1.0, gamma_limit = (70, 130)),
                A.RandomBrightnessContrast(brightness_by_max = False, p=1.0),
                A.IAASharpen(p=1.0),
                A.Blur(p=1.0),
            ], p=0.5),

            # Affine
            A.ElasticTransform(alpha_affine = 0, p=0.2),
            A.ShiftScaleRotate(shift_limit = 0, rotate_limit = 22, p=0.2, border_mode = cv2.BORDER_CONSTANT),

            A.Normalize(p=1.0),
            ToTensorV2()
        ])

        self.val_transforms = A.Compose([
            A.Resize (self.rSize, self.rSize),
            A.CenterCrop(self.size, self.size),
            A.Normalize(always_apply=True),
            ToTensorV2()
        ])

        # prepare folds now
        print ("FOLD ARG:",fold)
        if fold == "all" or fold == -1:
            self.fold = -1
        else:
            try:
                self.fold = int(fold)
            except:
                raise Exception ("Fold must be numeric.")
            # just for here
            if self.fold < 0 or self.fold > nCV-1:
                raise Exception (str(nCV) + "-fold CV needs fold between 0 and " + str(nCV-1))
            foldFile = "./data/folds_" + str(nCV) + "/" + str(self.fold) + "_train.csv"
            if os.path.exists(foldFile) == False:
                print ("Recreating folds")
                np.random.seed (667)
                # read csv
                df = pd.read_csv( "./data/train_final.csv")
                df = df[np.isnan(df["age"]) == False]
                # split
                kf = KFold(n_splits = nCV, shuffle = True, random_state = 2)
                os.makedirs("./data/folds_" + str(nCV), exist_ok = True)
                for j, (train_index, test_index) in enumerate(kf.split(df)):
                    X_train, X_test = df.loc[train_index,], df.loc[test_index, ]
                    # dump them
                    X_train.to_csv("./data/folds_" + str(nCV) +"/" + str(j) + "_train.csv", index = False)
                    X_test.to_csv("./data/folds_" + str(nCV) + "/" + str(j) + "_val.csv", index = False)
                pass
            pass


    def prepare_data (self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "ival":
            print ("Loading internal validation")
            self.testData =  KneeDataset("ival",  csvFile = "./data/ival_final.csv", transform = self.val_transforms, subset = self.subset)
            return (None)
        if stage == "eli.2021A":
            print ("Loading external eli.2021A")
            self.testData =  KneeDataset("eli.2021A",  csvFile = "./data/eli.2021A_final.csv", transform = self.val_transforms, subset = self.subset)
            return (None)
        # final train?
        if self.fold == "all" or self.fold == -1:
            print ("Loading  data for final training -- stage:", stage)
            self.trainData =  KneeDataset("train", csvFile = "./data/train_final.csv",  transform = self.train_transforms, subset = self.subset)
            self.valData =  KneeDataset("train", csvFile = "./data/train_final.csv", transform = self.val_transforms, subset = self.subset )
            self.testData = None
            return(None)
        # now it must be a fold
        print ("Loading fold ", self.fold, "-- stage:", stage)
        self.trainData =  KneeDataset("train", csvFile = "./data/folds_" +str(self.nCV) + "/" + str(self.fold) + "_train.csv",  transform = self.train_transforms, subset = self.subset)
        self.valData =  KneeDataset("val", csvFile = "./data/folds_" +str(self.nCV) + "/" + str(self.fold) + "_val.csv", transform = self.val_transforms, subset = self.subset )
        self.testData =  KneeDataset("test", csvFile = "./data/folds_" +str(self.nCV) + "/" + str(self.fold) + "_val.csv", transform = self.val_transforms, subset = self.subset )
        pass


    def train_dataloader(self):
        print ("Training on train")
        return DataLoader(self.trainData, batch_size=self.batch_size,
            num_workers = self.num_workers)

    def val_dataloader(self):
        print ("Validating on val")
        return DataLoader(self.valData, batch_size=self.batch_size,
            num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=self.batch_size,
            num_workers = self.num_workers)


#
