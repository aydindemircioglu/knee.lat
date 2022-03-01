# Classification of the Anatomical Side

This is the code for the training and prediction of the anatomical side of the knee. Training basically proceeds by calling `./train.sh`. This will evaluate the learning rate and the type of loss to use. After training is complete, the best parameters must be jotted down in the `./eval.sh` file. This file is then used to retrain the final model and to evaluate it on the two validation sets, the internal one (ival) and the one from the external hospital (eli2021).

- For training, corresponding .CSV files must be placed into ./data. These have the following structure:

`pat_ckey,study_instance_uid,patient_flag,exam_uid,lta_stat,dttm,image_cnt,ris_exam_id,exam_ckey,identifier,sts_1_stat,procedure_code,age,Laterality,ASM_x,ASM_y,ASM_Width,ASM_Height,Invalid,series_instance_uid,AccessionNumber,series_number,series_description,View,DCM_File,year,ImagePath`

Not all fields are used for training, refer to the train.py file.

- The folds for the 5-fold CV will be created automatically, if they are not available at `./data/folds`.

- All data and checkpoint paths must be adapted in all files.

- After training, the `visualize.py` script can be used to create occlusion maps.
#
