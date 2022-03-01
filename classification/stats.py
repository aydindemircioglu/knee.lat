#
import argparse
import pycm
import statsmodels
import statsmodels.stats.weightstats
from pprint import pprint
from scipy import stats
import numpy as np
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import dill
import progressbar
import numpy as np
import torch
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt

from datasets import *
from model import KneeModel



class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')

        # only parameters for tuning
        self.parser.add_argument('--lr', type=float, default=None, help='initial learning rate for adam')
        self.parser.add_argument('--loss', type=str, default="CE", help='CE or focal')

        # CV
        #self.parser.add_argument('--cv', type=int, default=5, help='n for CV')
        #self.parser.add_argument('--fold', type=int, default=0, help='fold for CV')
        self.parser.add_argument('-f', type=str, default=None, help='dummy, make hydrogen happy')

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




def evaluate (targets, preds):
    t = np.array(targets)
    p = np.array(preds)
    mae = np.mean(np.abs(t-p))
    #print (preds)

    cm = confusion_matrix(t, p)
    print(cm)
    return cm



def getStats (t, p):
    t = np.array(t)
    p = np.array(p)
    p = np.asarray(p, dtype = np.uint8)

    cm = pycm.ConfusionMatrix(t,p)
    print(cm)
    acc = cm.class_stat["ACC"][0]
    print ("\tAccuracy:",acc)
    pass



def getStatsFromModel (rTable, modelName):
    getStats (rTable["Target"], rTable[modelName])



def evalModel (m):
    print ("\n\n## ", m)
    rTable = pd.read_csv(m)

    print ("Model_0")
    getStatsFromModel (rTable, "Model_0")
    pass



# scatter plot
def createHistogram (rTable):
    fig, axs = plt.subplots(ncols=1, figsize = (10,10))

    diffs = rTable["Target"] - rTable["Ensemble Model"]
    sns.set_style("whitegrid")
    sns.histplot(diffs,  color="#333333", ax=axs, stat = "count", binwidth = 1/6)
    from matplotlib.ticker import MaxNLocator
    axs.yaxis.set_major_locator(MaxNLocator(integer=True))
    axs.set_ylabel("Count", fontsize= 20)
    axs.set_xlabel("Prediction Error \n(Training cohort)", fontsize= 20)
    axs.tick_params(axis='x', labelsize=14 )
    axs.tick_params(axis='y', labelsize=14 )
    return axs



# scatter plot
def createPlot (rTable, msize):
    fig, axs = plt.subplots(ncols=1, figsize = (10,10))

    u = rTable["Target_class"].unique()
    scolor=plt.get_cmap("Spectral")
    color = scolor  (np.linspace(.1,.8, len(u)))
    medianprops = dict(color='firebrick', linewidth=2.0)
    for c, (name, group) in zip(color, rTable.groupby("Target_class")):
        bp = axs.boxplot(group["Ensemble Model"].values, positions=[name], widths=0.8, patch_artist=True, medianprops=medianprops)
        c = (0.9,0.9, 0.9)
        bp['boxes'][0].set_facecolor(c)

    sns.set_style("whitegrid")
    #sns.set(style="whitegrid",  {"axes.facecolor": ".4"})

    axs.set_xlim(0,21)
    sns.lineplot(x= [0,  21], y=[0.5, 21.5], color = "black", linewidth = 1.5, ax =axs) # shifted by 0.5, as ageclass 0-1 should have prediction around 0.5
    #sns.boxplot(x="Target_class", y="Ensemble_Model", data=rTable, showfliers = False, order=range(0,21), color = "#aaffcc", ax = axs)
    sns.swarmplot(x="Target_class", y="Ensemble Model", data=rTable, color=".25", size = msize, ax = axs)
    axs.set_ylabel("Predictions of Ensemble Model", fontsize= 20)
    axs.set_xlabel("True Chronological Age Class", fontsize= 20)
    axs.set_xticks(range(0,22))
    axs.set_xticklabels([str(k)+"-"+str(k+1) for k in range(0,21)] + [''])
    axs.yaxis.set_ticks(np.arange(0, 22, 1))
    axs.grid(b=True, which='both', color='black', linewidth=0.2)
    axs.tick_params(axis='x', labelsize=14 )
    axs.tick_params(axis='y', labelsize=14 )
    for label in axs.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(45)

    axs.get_figure()
    return axs


if __name__ == '__main__':
    opt = BaseOptions().parse()

    # only analyse mae, but no plots
    if opt.cohort == "cvmae":
        # for all LRs
        modelList = glob("./results/eval*csv")
        for m in modelList:
            evalModel (m)
        exit(-1)


    fname = "eval_" + str(opt.cohort) + "_lr_" + str(opt.lr) + "_" + str(opt.loss) + ".csv"
    evalModel ("results/" + fname)
    rTable = pd.read_csv(os.path.join("results", fname))


    # find bad ones
    print ("ERRORS")
    ePath = os.path.join("./results/errors/", opt.cohort)
    os.makedirs(ePath, exist_ok = True)
    subdata = rTable.copy()

    useModel = "Model_0"

    subdata["Diffs"] = np.abs(subdata["Model_0"] != subdata["Target"])
    bdata = subdata[subdata["Diffs"] == 1].copy()

    doLabel = False
    if doLabel == True:
        for i, (idx, row) in enumerate(bdata.iterrows()):
            img = cv2.imread(row["ImagePath"], 3)
            # write age to it
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 0.4
            fontColor              = (255,0,0)
            lineType               = 1
            _ = cv2.putText(img,"target:" + str(row["Target"]) + " -- pred:" + str(row["Model_0"]) ,  (50,50), font, fontScale, fontColor, lineType)
            #Image.fromarray(img)
            cv2.imwrite(os.path.join(ePath, "error_" + os.path.basename(row["ImagePath"]) ), img)
    print ("DONE")

    ptitle = None
    pname = None
    msize = 2
    if opt.cohort == "eli.2021A":
        ptitle = "external validation cohort"
        pname = "cal_curve_ext_val"
    elif opt.cohort == "ival":
        ptitle = "internal validation cohort"
        pname = "cal_curve_int_val"
    else:
        ptitle = "cross-validation folds"
        pname = "cal_curve_cv_fit"
        msize = 1
 

#
