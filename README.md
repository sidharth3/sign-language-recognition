# 50.039 Deep Learning Sign Language Recognition

As part of our Big Project of 50.039 Deep Learning, our team decided to tackle isolated sign language recognition from signer-independent videos involving a large number of sign categories. Below is a brief overview of the project:

### Dataset

We used the WLASL Dataset, which is the largest video dataset for Word-Level American Sign Language (ASL) recognition, which features 2,000 common different words in ASL. You can download the dataset at this link: https://drive.google.com/drive/folders/1ekKwXXHfoxTkyysuWKwvNRQc1Le73og9?usp=sharing 

### Architecture

The model architecture used is a Graph Convolutional Network (GCN) with multiple stacks of GCN and BatchNorm layers. The final layer is a linear fully-classified layer for classification purposes. 

### Inputs and Outputs

- Inputs - 2-3s Videos of hands showing sign languages
- Output - Classification of sign language into its corresponding labels

### To run the code:

Run `python train.py`

Files:

- `asl100.ini` - configuration file containing parameters for training, optimizer as well as parameters for our GCN model
- `configs.py` - configuration file
- `dataloader.py` - file containing our Dataloader
- `model_final.py` - file containing our GCN model
- `train.py` - file containing training and validation fuctions
- `utils.py` - file containing one-hot encoding functions and relevant functions for plotting functions

### To load trained model weights
Please find saved model weights in the  ‘weights’ folder. To load a model, run the first two cells of the submission notebook, and then run the last cell. The function takes in the path of any saved weight and loads the model. This then runs the model on evaluation mode and calls the validation function, which reproduces the accuracy we obtained.

### Group Members:

Suhas Sahu (1003370) <br>
Ong Li-Chang (1003328) <br>
Pung Tuck Wei (1003523) <br>
Noorbakht Khan (1003827) <br>
Sidharth Praveenkumar (1003647)

### Please refer to the Report PDF for a more detailed breakdown of the project.
