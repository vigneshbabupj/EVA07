# EVA07 Session 4 Assignment

### Submission by:
1. Sachin Dangayach (sachin.dangayach@gmail.com)
2. Vignesh Babu P J (vigneshbabupj@gmail.com)
3. Sherine 
4. Malathi M

# PART A

## BackProp, Embeddings and Language Models

**Assignment**:
### 1. Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github.:
  - Use exactly the same values for all variables as used in the class
  - Take a screenshot, and show that screenshot in the readme file

## Solution:
  - Screenshot of excel -

  ![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/Excel_NN.JPG)

### Excel file must be there for us to cross-check the image shown on readme (no image = no score)

## Solution:  
**[Link to excel](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/NN%20Training.xlsx)**

### Explain each major step

## Solution:  
![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/NN.png)
***NN parameters, activation function output and loss calculation***  
h1 = w1 * i1+w2 * i2  
h2 = w3 * i1+w4 * i2  
a_h1 = σ(h1) = 1/(1+exp(-h1))  
a_h2 = σ(h2)  
o1 = w5 * a_h1+w6 * a_h2  
o2 = w7 * a_h1+w8 * a_h2  
a_o1 = σ(o1)  
a_o2 = σ(o2)  
E1 = ½ * (t1-a_o1)²  
E2 = ½ * (t2-a_o2)²  

***Partial Derivation on total loss wrt Weights of network***
1.	∂E_T/∂w5 = ∂(E1+E2)/∂w5 = ∂E1/∂w5 = (∂E1/∂a_o1) * (∂a_o1/∂o1) * (∂o1/∂w5)  ---- (1)  
Now, ∂E1/∂a_o1 = (t1 - a_o1) * (-1) = a_o1 – t1   ---- x  
∂a_o1/∂o1 = ∂(σ(o1))/ ∂o1 = σ(o1) * (1 - σ(o1)) = a_o1 * (1 – a_o1)   ---- y  
∂o1/∂w5 = ∂(w5 * a_h1+w6 * a_h2)/ ∂w5 = a_h1  ---- z  
Substituting x, y and z in (1)  
***∂E_T/∂w5 = (a_o1 – t1) * (a_o1 * (1 – a_o1)) * (a_h1)***  
Similarly -  
2.	***∂E_T/∂w6 = (a_o1 – t1) * (a_o1 * (1 – a_o1)) * (a_h2)***  
3.	***∂E_T/∂w7 = (a_o2 – t2) * (a_o2 * (1 – a_o2)) * (a_h2)***  
4.	***∂E_T/∂w8 = (a_o2 – t2) * (a_o2 * (1 – a_o2)) * (a_h1)***  
5.	∂E_T/∂a_h1 = ∂(E1+E2)/∂a_h1 = ∂E1/∂a_h1 + ∂E2/∂a_h1  ---- (1)  
∂E1/∂a_h1 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1  ----l  
∂E1/∂a_o1 = a_o1 – t1   ---- m  
∂a_o1/∂o1 = a_o1 * (1 – a_o1)   ---- n  
∂o1/∂a_h1 = ∂(w5 * a_h1+w6 * a_h2) /∂a_h1 = w5   ----o  
Substituting m, n, o in l  
∂E1/∂a_h1 = (a_o1 – t1) * (a_o1 * (1 – a_o1)) * w5 --- x  
Similarly -    
∂E2/∂a_h1 = (a_o2 – t2) * (a_o2 * (1 – a_o2)) * w7  ---- y  
Substituting x, y in (1)  
***∂E_T/∂a_h1 = (a_o1 – t1) * (a_o1 * (1 – a_o1)) * w5 + (a_o2 – t2) * (a_o2 * (1 – a_o2)) * w7***  
6.	***∂E_T/∂a_h2 = (a_o2 – t2) * (a_o2 * (1 – a_o2)) * w8 + (a_o1 – t1) * (a_o1 * (1 – a_o1)) * w6***  
7.	∂E_T/∂w1 = ∂E_T/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1 ---- (1)  
∂a_h1/∂h1 = a_h1 * ( 1 – a_h1)  -----x  
∂h1/∂w1 = i1 ---- y  
Substituting x, y in (1)  
∂E_T/∂w1 = ((a_o1 – t1) * (a_o1 * (1 – a_o1)) * w5 + (a_o2 – t2) * (a_o2 * (1 – a_o2)) * w7) *   (a_h1 * ( 1 – a_h1)  ) * i1  
Or  
***∂E_T/∂w1 = ∂E_T/∂a_h1 * (a_h1 * ( 1 – a_h1)  ) * i1***    
Similarly -    
8.	***∂E_T/∂w2 = ∂E_T/∂a_h1 * (a_h1 * ( 1 – a_h1)  ) * i2***      
9.	***∂E_T/∂w3 = ∂E_T/∂a_h2 * (a_h2 * ( 1 – a_h2)  ) * i1***    
10.	***∂E_T/∂w4 = ∂E_T/∂a_h2 * (a_h2 * ( 1 – a_h2)  ) * i2***    

***Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0]***  

## Solution:  
***Total Loss per epoch for Learning Rate = 0.1***

![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/lr_01.JPG)

***Total Loss per epoch for Learning Rate = 0.2***

![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/lr_02.JPG)

***Total Loss per epoch for Learning Rate = 0.5***

![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/LR_05.JPG)

***Total Loss per epoch for Learning Rate = 0.8***

![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/LR_08.JPG)

***Total Loss per epoch for Learning Rate = 1***

![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/LR_1.JPG)

***Total Loss per epoch for Learning Rate = 2***

![alt](https://github.com/vigneshbabupj/EVA07/blob/main/Session_04/Images/LR_2.JPG)


# PART B
**Assignment**:
In this assignment we have to train a neural network on MNIST dataset of handwritten digits with following constraints-
1. We have to achieve minimum 99.4% validation accuracy
2. We can use less than 20k Parameters only
3. Less than 20 Epochs should be used
4. No fully connected layer

# We have trained the network and acheived 99.41% accuracy in 17 Epoch.

## Model Details
### 1. Number of parameters:- Total params: 10,066 (Learnt during back propagation)
Trainable params: 10,066 (Learnt during back propagation)
### 2. Batch Size:-64
Used for deciding number of images feed to the network to calculate the loss, before back propagation happens to adjust the weights and reduce the loss
### 3. Learning Rate:- 0.1
Multiplied with gardient to decide the value to update the weight
### 4. Kernel Size:- 3*3
Kernel help to extract the features. Acts as both filter and gate.
### 5. Optimizer:- SGD
One of the aglo. used for updating the weights
### 6. Activation Function:- Relu
Used to act as a gate to let values passed to the next layer or not

### 7. Model Summary
|  Layer (type) |Output Shape   |  Param #  |  Comments |
|---|---|---|---|
| Conv2d-1  |[-1, 8, 26, 26]   | 80  | 3*3 kernel is convolved or 28 * 28 * 1 to extract features and generate 8 Channels  |
| BatchNorm2d-2  |[-1, 8, 26, 26]   |16   |Batch Normalization to sharpen the extracted features   |
| Dropout-3  |[-1, 8, 26, 26]   |0   | Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-4 |[-1, 16, 24, 24]   | 1168  | 3*3 kernel is convolved or 26 * 26 * 1 to extract features and generate 16 Channels  |
|BatchNorm2d-5  |[-1, 16, 24, 24]   |32   |Batch Normalization to sharpen the extracted features    |
|Dropout-6  |[-1, 16, 24, 24]   |0   |Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-7 | [-1, 16, 22, 22]  |2320   |3*3 kernel is convolved or 24 * 24 * 1 to extract features and generate new 16 Channels |
|BatchNorm2d-8  |[-1, 16, 22, 22]   |32   | Batch Normalization to sharpen the extracted features   |
|Dropout-9 |[-1, 16, 22, 22]   |0   | Dropout layer added for regularization. To force all neurons to learn   |
|MaxPool2d-10  | [-1, 16, 11, 11]   |0   |Max Pool layer is used to reduce the number of layers in network (22 * 22 * 16     - 11 * 11 * 16)  |
|BatchNorm2d-11  |  [-1, 16, 11, 11]  |32   |Batch Normalization to sharpen the extracted features    |
|Dropout-12  | [-1, 16, 11, 11]   |0   |  Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-13 |[-1, 8, 11, 11]   |36   |1*1 kernel is used to mix the channels as DJ to reduce the number of channels from 16 to 8. It acts as a filter also to drop unwated features  (convolved or 11 * 11 * 16 to result 11 * 11 * 8    |
|BatchNorm2d-14   |[-1, 8, 11, 11]   |16   | Batch Normalization to sharpen the extracted features   |
|Dropout-15   |[-1, 8, 11, 11]   |0   | Dropout layer added for regularization. To force all neurons to learn   |
|Conv2d-16  |  [-1, 16, 9, 9]  |1168   |3*3 kernel is convolved or 11 * 11 * 8 to extract features to generate new 16 Channels  with output as 9 * 9 * 16 |
|BatchNorm2d-17  |  [-1, 16, 9, 9]  |32   |Batch Normalization to sharpen the extracted features    |
|Dropout-18 | [-1, 16, 9, 9]   |0   |  Dropout layer added for regularization. To force all neurons to learn  |
|Conv2d-19   |[-1, 32, 7, 7]   | 4640  | 3*3 kernel is convolved or 9 * 9 * 16 to extract features to generate new 32 Channels  with output as 7 * 7 * 32 |
|BatchNorm2d-20   | [-1, 32, 7, 7]  |64   |Batch Normalization to sharpen the extracted features    |
|Dropout-21   |[-1, 32, 7, 7]   |0   | Dropout layer added for regularization. To force all neurons to learn   |
|AvgPool2d-22   | [-1, 32, 1, 1]   |0   | Global Average pooling is used to reduce the number of laters from 7 * 7 to 1 * 1   |
|Conv2d-23   | [-1, 32, 1, 1]   |330   | Ten 1 * 1 * 32 kernels used to mix and generate the flatten output to feed Log_softmax   |

# EVA5_99_41.pynp file with code and logs is placed in repo
