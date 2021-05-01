# 50.039 Deep Learning Sign Language Recognition

As part of our Big Project of 50.039 Deep Learning, our team decided to tackle isolated sign language recognition from signer-independent videos involving a large number of sign categories.

### Dataset

We used the WLASL Dataset, which is the largest video dataset for Word-Level American Sign Language (ASL recognition, which features 2,000 common different words in ASL.

### Architecture

As for our model architecture, we implemented a Graph Convolutional Network (GCN) with multiple stacks of GCN and BatchNorm layers. Our GCN is made of a fully-connected linear layer with ReLu as its activation function.

### Inputs and Outputs

- Inputs - 2-3s Videos of hands showing sign languages
- Output - Classification of sign language into its corresponding labels

### To run the code:

Run `python train.py`

Files:

- `asl100.ini` - configuration file containing parameters for training, optimizer and paramters for our GCN model
- `configs.py` - configuration file
- `dataloader.py` - file containing our Dataloader
- `model_final.py` - file containing our GCN model
- `train.py` - file containing training and validation fuctions
- `utils.py` - file containing one-hot encoding functions and relevant functions for plotting functions

### Group Members:

Suhas Sahu (1003370) <br>
Ong Li-Chang (1003328) <br>
Pung Tuck Wei (1003523) <br>
Noorbakht Khan (1003827) <br>
Sidharth Praveenkumar (1003647)
