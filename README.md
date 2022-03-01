
# Determining the Anatomical Site in Knee Radiographs Using Deep Learning

This is the code for our paper, published in Scientific Reports. The code only contains the code for detection and classification. The custom-tailored annotation tool is not part of this repo, the same is true for all data. The final model is included, albeit it might perform suboptimal when given full views of the knee, as it was trained on images cropped to the main knee area (and avoiding all side markers in the images).


## Detection

Detection of the knee was performed using MMDetection. The codes can be found in ./mmdetect. Follow the README.md there.


## Classification

Classifiction was performed using pyTorch Lightning. The code can be found in ./classificaiton.  Follow the README.md there. 


#
