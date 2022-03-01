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
import torch.nn.functional as F

from datasets import *
from model import KneeModel

from torch import nn
from PIL import ImageDraw, ImageFont
import PIL

from captum.attr import visualization as viz
from captum.attr import Occlusion



def addText (finalImage, text = '', org = (0,0), fontFace = 'Arial', fontSize = 12, color = (255,255,255)):
     # Convert the image to RGB (OpenCV uses BGR)
     tmpImg = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
     pil_im = Image.fromarray(tmpImg)
     draw = ImageDraw.Draw(pil_im)
     font = ImageFont.truetype(fontFace + ".ttf", fontSize)
     draw.text(org, text, font=font)
     tmpImg = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
     return (tmpImg.copy())



if __name__ == '__main__':
    #/data/uke/data/knee.lat/checkpoints/resnet34/0.0003/CE/all/epoch=30-val_loss=0.00_lr_0.0003_CE.ckpt
    modelList = glob (os.path.join("/data/uke/data/knee.lat/checkpoints/resnet34/0.0003/CE/all/*"))
    print ("Have", len(modelList), "models")

    seed_everything(571)

    model = KneeModel.load_from_checkpoint(modelList[0], model = "resnet34", subset = False)
    _ = model.cuda()
    _ = model.eval()

    vizList = {}
    for stage in ["ival", "eli.2021A"]:
        dm = KneeDataModule(subset = False, batch_size = 1, fold = "all", nCV = -1)
        dm.setup(stage = "ival")


        # our forward has no_grad-- have to overwrite this
        class MyModel  (nn.Module):
            def __init__(self, basemodel):
                super(MyModel, self).__init__()

                # Here you get the bottleneck/feature extractor
                self.feature_extractor = nn.Sequential(*list(basemodel.children())[:-1])
                self.classifier = nn.Sequential(*list(basemodel.children())[-1:])

            # Set your own forward pass
            def forward(self, x):
                self.feature_extractor.eval()
                representations = self.feature_extractor(x)
                representations = representations.flatten(1)
                o = self.classifier(representations)
                return o

            def forward(self, x):
                return self.classifier(self.feature_extractor(x).flatten(1))


        gradmodel = MyModel(model)
        _ = gradmodel.eval()


        images = []

        figImages = random.sample(range(len(dm.test_dataloader())), 4)
        for j, batch in enumerate(dm.test_dataloader()):
            if j not in figImages:
                continue

            y = batch[1].cuda()
            torch_img = batch[0].cuda()
            # pil_img = PIL.Image.open(batch[2][0]).convert("RGB")

            transformed_img = (torch_img - torch.min(torch_img))/(torch.max(torch_img) - torch.min(torch_img))


            output = gradmodel(torch_img)
            output = F.softmax(output, dim=1)
            prediction_score, pred_label_idx = torch.topk(output, 1)

            pred_label_idx.squeeze_()
            predicted_label = ["R", "L"][pred_label_idx]
            true_label = ["R", "L"][y]
            print('True', true_label, 'Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

            occlusion = Occlusion(gradmodel)
            attributions_occ = occlusion.attribute(torch_img,
                                                   strides = (3, 8, 8),
                                                   target=pred_label_idx,
                                                   sliding_window_shapes=(3, 48, 48),
                                                   baselines=0)

            rt = np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0))
            rt = rt/np.max(rt)*255
            rtA = np.asarray(rt, dtype = np.uint8)

            rt = np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0))
            rt = (rt - np.min(rt))/(np.max(rt) - np.min(rt))
            rt = rt/np.max(rt)*255
            rtB = np.asarray(rt, dtype = np.uint8)

            #rtB = cv2.applyColorMap(rtB, cv2.COLORMAP_MAGMA)
            rtB = cv2.applyColorMap(rtB, cv2.COLORMAP_VIRIDIS)

            rtC = 1.0*rtA + rtB*0.5
            rtC = rtC/(np.max(rtC))
            rtC = np.asarray(255*rtC, dtype = np.uint8)
            # rtC = addText (rtC, text="("+str(j) + ")", fontSize = 33)

            rImg = np.hstack([rtA,rtB,rtC])
            images.append (rImg)

        vizList[stage] = images
    vizList["black"] = [z[:,0:48,:]*0+255 for z in vizList["ival"]]
    z = np.hstack([np.vstack(vizList[stage]) for stage in ["ival", "black", "eli.2021A"]])
    z = cv2.resize(z, (0,0), fx=2.5, fy = 2.5)
    cv2.imwrite("../results/Figure_5.png", z)




#
