# Machine Learning Project - Image Classification - Convolutional Neural Network

Image Classification on CIFAR-10 dataset. This data set is a part of Keras library. 
CIFAR is a dataset from Canadian Institute of Advanced Research, hence the name.
The number 10 indicates the number of classes. Each image is a color image with 32by32 pixel size. The dataset is balanced meaning that each class has same number of images.

Some More Background on Digital Representation of Images

A greyscale image is a system of 256 tones with values 0-256, where 0 represents black and 256 represents white.

CNN Layer --> Kernel/Feature Detectors (Convolution Layer, Sharpens the Image, Apply Activation Function RELU)  -->  Pooling Filter/Layer --> Downsampling (Reduce size, compress to reduce computational complexity) --> Flatenning --> Output Layer

Convolution uses a kernel matric to scan a given image and apply a filter to obtain certain effect. It generates feature map, like different variation of an image example sharpen, blur, rotate etc.

Kernel is used for feature extraction to select most important feature in an image.

Convolution preserves the spatial relationship between pixels.

RELU - Rectified Linear Unit works in a way that if any value in feature map is less than zero, then it is set to zero.

Pooling is used to reduce feature map dimensionality. It impoves computational efficiency while preserving the features. Max Pooling works by retaining the maximum feature response within a given sample size in a feature map.


Dropout is a regularization technique for reducing overfitting in neural network. Dropout refers to dropping out units in a neural network.

Confusion Matrix is a way to represent result of Classification.


Key Performance Indicators (KPI) :

Classification Accuracy = (TP + TN)/(TP + TN + FP + FN)
Misclassification or Error Rate = (FP + FN)/(TP + TN + FP + FN)
Precision = TP / (TP + FP) = TP / Total True
Recall = TP / (TP + FN) = TP / Actual True

Steps :

## 1) Import the libraries Pandas, Numpy, Pyplot, Seaborn

## 2) Load dataset CIFAR 10 from Keras dataset and split into Train and Test dataset

## 3) Examine and Visualize the data

## 4) Visualize matrix of images using subplots

## 5) Convert Train & Test data into Float type

## 6) Convert the Y part into Binary form (Categorical)

## 7) Normalization of X

## 8) Train the CNN model using Keras

## 9) Build the model in sequential manner

## 10) Compile the model and Train the Epochs

## 11) Evaluate the Model

## 12) Predict the model on Test dataset

## 13) Confusion Matrix, Heatmap from Seaborn

## 14) Save the model using os.path

## 15) Try Image Augmentation for better Training

## 16) Model Training using Augmented dataset
