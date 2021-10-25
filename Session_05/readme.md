# EVA - Assignment 5 - MNIST Classification problem

By:
  - Vignesh Babu P J -> vigneshbabupj@gmail.com

Convolution neural network designed to clasify the images of the MNIST dataset in Pytorch

> The MNIST dataset (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems.

# Problem Statement
Target:
* 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
* Less than or equal to 15 Epochs
* Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
* Do this in exactly 4 steps


# Architecture 
## Step -1 

    Target:
    - Set up code
    
    Results:
    - Params: 21562
    - Train Accuracy: 99.6
    - Test Accuracy : 98.9
    
    Analysis:
    - High no of parameters
    - Very Big kernel is used - Can be replaced by GAP
    - There is scope of accuracy improvement

    File : ![Link](https://github.com/vigneshbabupj/EVA07/blob/main/Session_05/EVA_S5_Step_0.ipynb)

## Step -2

    Target:
    - Reduce Parameters by introducing GAP
    - Add a conv layer after GAP as output layer (no Layer Change after this)
    
    Results:
    - Params: 10362
    - Train Accuracy: 99.09
    - Test Accuracy : 98.7
    
    Analysis:
    - no overfitting
    - model can be improved
    - No of parameters is still above 10K mark

    File : ![Link](https://github.com/vigneshbabupj/EVA07/blob/main/Session_05/EVA_S5_Step_1.ipynb)

## Step -3

    Target :
     - Add batch norm to improve model 
     -reduce channel size in last layer from 32 to 16 to bring params less than 10K
    
    Results:
    - Params: 8066
    - Train Accuracy: 99.4
    - Test Accuracy : 99.4 ( 9th and 13th Epoch )
    
    Analysis:
    - no overfitting - slight underfitting
    - model can be improved
    - Accuracy is not consistent ( keep changing drastically for each Epoch) - Will introduct LR scheduler to control this

    File : ![Link](https://github.com/vigneshbabupj/EVA07/blob/main/Session_05/EVA_S5_Step_2.ipynb)
    
## Step -4

    Target :
     - Add Lr schuler - Step size 10 ( why 10 ? - bcoz we reached target accuracy at 9th/10th epoch, so we want it to consistent post that )
    
    
     Result:
    - Params: 8066
    - Train Accuracy: 99.6 (from 11th till 14th Epoch)
    - Test Accuracy : 99.4 (from 9th and 14th Epoch )
    
    Analysis:
    - Still no overfitting
    - Getting target accuracy consistently
    - next try to reduce params below 8k

    File : ![Link](https://github.com/vigneshbabupj/EVA07/blob/main/Session_05/EVA_S5_Step_3.ipynb)

