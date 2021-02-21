# AIDEA mango defect classification Challenge

##  Environment
- PyTorch 1.5.1  & torchvision 0.5.1 <br>
- CUDA 10.2 <br>
- Python 3.6. <br>
- GPU : Nvidia Tesla V100 <br>

## Data Description & Statistics

##  Analyze
- There might be more than one defects in a mango image.<br>
- Unlike normal multi class classification, each image with one only one class, the prediction of the image sometimes will be more than one class, which means this is a multi label multi class classification tasks. <br>
- So the final activation layer of the model can not be Softmax, we use Sigmoid instead. <br>
- The class ouputs of model is still 5.
- Loss function here we choose BinaryCrossEntropy.

## Training Detail
- 
## Results

## Ranking

## Prediction Visualize
