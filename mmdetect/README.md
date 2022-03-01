
# Detection of the main knee area

## Prerequisites

All DICOM x-ray knee files must be first converted to 8-bit png.
The png files must be put into a folder and the path must
be changed accordingly in the ./createDataset.py file

`annDir = "/data/uke/data/knee.lat/asm_detect/train"``

In addition, each png must be accompanied with an annotation.
The annotations must be put into ../annotations.
An example annotation can be found there.


## Training

- Annotate the knees, putting their annotations into ../annotations

- Create from those annotations (in ../annotations) a data set suitable for MMDetection. Note that not all fields of the annotation are used for this. Refer to ./createDatasets.py to see which fields are necessary.
`./createDatasets.py`

- Now train the Cascade RCNN by calling. Change the checkpoints path in the ./run.sh file. Adapt the pathes in the ./configs/cascade_rcnn_*.py paths accordingly. A copy of MMDetection is needed.
`./run.sh`

- Training is not using any validation set. Overfitting could occur, but the risk is much lower for the Cascade-RCNN than for fully connected networks etc.

- The trained model can then be used in the evaluation.


#
