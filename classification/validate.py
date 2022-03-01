#
import argparse
from pprint import pprint
import numpy as np
from glob import glob
import pandas as pd
import os
import random
import dill
import progressbar
import numpy as np
import torch
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import scipy
from datasets import *
from model import KneeModel
from utils import recreatePath
import shutil


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # only parameters for tuning
        self.parser.add_argument('--lr', type=float, default=None, help='initial learning rate for adam')

        # CV
        self.parser.add_argument('--cv', type=int, default=5, help='n for CV')
        self.parser.add_argument('-f', type=str, default=None, help='dummy, make hydrogen happy')
        self.parser.add_argument('--loss', type=str, default="CE", help='CE or focal')

        # input/output sizes
        self.parser.add_argument('--cohort', type=str, default="fit", help='cohort to evaluate')
        self.parser.add_argument('--gpus', type=str, default="0", help='list of gpus to run on')
        self.parser.add_argument('--model', type=str, default="resnet34", help='model for training')

        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)


        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt





# init model
def getResults (opt, ckptName, cohort, fold):
    seed_everything(571)
    model = KneeModel.load_from_checkpoint(ckptName, model = opt.model, subset = False)
    model.cuda()
    model.eval()

    dm = KneeDataModule(subset = False, fold = fold, nCV = opt.cv)
    dm.setup(stage = cohort)

    targets = []
    preds = []
    imgpaths = []
    for batch in dm.test_dataloader():
        y = batch[1].cuda()
        imgpath = batch[2]
        imgpaths.extend(imgpath)

        y_pred = model(batch[0].cuda())
        preds.extend(y_pred.detach().cpu().numpy())
        targets.extend(y.detach().cpu().numpy())
    return targets, preds, imgpaths


def getResultsUsingModel (opt, f, cohort):
    targets = []
    preds = []
    imgpaths = []
    modelList = glob (os.path.join("/data/uke/data/knee.lat/checkpoints", str(opt.model), str(opt.lr), str(opt.loss), str(f), "*"))

    print ("Have", len(modelList), "models")
    accList = [(float(f.split("val_loss=")[1].replace(".ckpt", "").split("_")[0] ), f) for f in modelList]
    accList = sorted(accList)
    pprint (accList)

    # just in case take only the last model, the last one will be the best, as all epochs are 0.0, stupid..
    accList = [accList[-1]]
    for m in accList:
        print ("Using model ", m, " to predict on ", cohort, " setting fold to ", f)
        t, p, ipath = getResults (opt, m[1], cohort, f)
        targets.append(t)
        preds.append(p)
        imgpaths.append(ipath)
    return targets, preds, imgpaths





if __name__ == '__main__':
    opt = BaseOptions().parse()

    results = {}
    # we use all models we trained using CV
    # first query the results. here the results will be over the left-out folds in case of CV
    # and the preds on the int/ext validation resp
    if opt.cohort == "fit":
        nModels = 1 # only best model is necessary here, no ensembling like in the other project
        for f in range(opt.cv):
            print ("Predicting using model ", f, " on fold", f)
            results["Fold_" + str(f)] = getResultsUsingModel (opt, f, opt.cohort)
    else:
        nModels = 1
        results["Model_0"] = getResultsUsingModel (opt, "all", opt.cohort)



    # also need for age things-- revision
    if opt.cohort == "fit":
        trData = pd.read_csv("./data/train_final.csv")
    else:
        trData = pd.read_csv("./data/" + opt.cohort + "_final.csv")

    # if its fit then we have in results for each fold the preds of the 5 models kept during train
    # if its validation then we have in results the preds of the mdoel trained over  each  fold for that one test set

    fTable = []
    for k in sorted(list(results.keys())):
        t, p, d = results[k]
        rTable = {}
        # have 5 models predictions for same data, so target,  imagepath are shared, so take 1st one
        rTable["Target"] = t[0]
        rTable["ImagePath"] = d[0]
        rTable["Fold"] = k
        for j in range(nModels): # top 5 models
            rTable["Model_" + str(j)] = p[j]

        # revision

        ages = []
        for ipath in d[0]:
            ires = trData.query("ImagePath == @ipath")
            if ires.shape[0] != 1:
                print(ires)
                print ("OFFENDING PATH", ipath)
                raise Exception ("### BROKEN:")
            ages.append(ires.iloc[0]["age"])
        rTable["Age"] = ages

        pd.DataFrame(rTable)
        fTable.append(pd.DataFrame(rTable))
    rTable = pd.concat(fTable).reset_index()

    # first convert logit outputs to binary -- for now overwrite, lossing "confidence"
    for i, (idx, row) in enumerate(rTable.iterrows()):
        for z in ["Model_"+str(_) for _ in range(nModels)]:
            rTable.at[idx, z] = np.argmax (row[z])


    def evaluate (rTable, model):
        t = np.asarray(rTable["Target"].copy())
        p = np.asarray(rTable[model].copy())
        # safety
        t = np.asarray(t, dtype = np.float64)
        p = np.asarray(p, dtype = np.float64)
        error = np.mean(np.abs(t-p))
        acc = 1 - error
        print ("ACCURACY:", acc)
        # also ages --revision
        eTable = rTable[rTable["Target"] != rTable[model]]
        print (sorted(np.round(e,) for e in eTable["Age"]))
        return acc, error

    # take each fold
    accs = []
    for k in rTable["Fold"].unique():
        sTable = rTable [ rTable["Fold"] == k].copy()
        acc, error = evaluate(sTable, "Model_0")
        accs.append(acc)
    print ("CV mean:", np.mean(accs), "+/-",  "; SE:", scipy.stats.sem (accs))

    # overall
    _ = evaluate(rTable, "Model_0")

    fname = "eval_" + str(opt.cohort) + "_lr_" + str(opt.lr) + "_"+ str(opt.loss) + ".csv"
    rTable.reset_index(drop = True)
    rTable.to_csv(os.path.join("results", fname))

    # also write out the error images for later use
    ePath = os.path.join("./results", "errors", str(opt.cohort) + "_lr_" + str(opt.lr) + "_"+ str(opt.loss))
    recreatePath (ePath)
    for k in range(rTable.shape[0]):
        if rTable.iloc[k]["Target"] != rTable.iloc[k]["Model_0"]:
            src = rTable.iloc[k]["ImagePath"]
            target = os.path.join(ePath, os.path.basename(src))
            shutil.copyfile (src, target)

#
