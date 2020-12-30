---
title: "Practical Machine Learning Project"
author: "Jenna Preston"
date: "12/29/2020"
output: 
  html_document: 
    keep_md: yes
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, we use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants in order to create a model that predicts the manner in which the participants did the exercise.  They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

## Data

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

## Processing the Data


```r
library(rpart)
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
training <- read.csv('pml-training.csv', header = TRUE)
testing <- read.csv('pml-testing.csv', header = TRUE)

dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

First I want to get rid of the columns that are missing and non-numeric (except for the classe column).  Then I can also remove the first 4 columns that contain user name, timestamp, etc. that are not relevant for the model.  


```r
training <- training[,colSums(is.na(training))==0]
testing <- testing[,colSums(is.na(testing))==0]

train_numericColumns <- unlist(lapply(training, is.numeric))
classe <- training$classe
training2 <- training[,train_numericColumns]
training2$classe <- classe
training2$classe <- as.factor(training2$classe)

test_numericColumns <- unlist(lapply(testing, is.numeric))
testing2 <- testing[,test_numericColumns]

training3 <- training2[,-c(1:4)]
testing3 <- testing2[,-c(1:4)]

dim(training3)
```

```
## [1] 19622    53
```

```r
dim(testing3)
```

```
## [1] 20 53
```

As we can see the total number of columns of data was reduced to 53 (from 160) in the training and testing data sets.

Next, I will partition the training data set into a portion that is purely the training set and then a portion that will be used for testing and cross validation.  I will use 70% of the training data set as purely training data and 30% of the training data set as the validation of the model.


```r
set.seed(1234)
inTrain <- createDataPartition(training3$classe, p=0.7, list= FALSE)
training3_trainingData <- training3[inTrain,]
training3_testingData <- training3[-inTrain,]
dim(training3_trainingData)
```

```
## [1] 13737    53
```

```r
dim(training3_testingData)
```

```
## [1] 5885   53
```

## Creating the Model

Now I will use the partition of the training data set that is purely for the training set to create the model.  For this project I will use the robust method of Random Forest.  I will use the train control with the number of folds as 5 to control the parameters for the model.  I also used parallel implementation to increase the performance of the random forest model.


```r
set.seed(1234)
library(parallel)
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```r
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

RFControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
RFModel <- train(classe ~ ., data=training3_trainingData, method = "rf", trControl = RFControl)
RFModel
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10991, 10988, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9914101  0.9891331
##   27    0.9922840  0.9902392
##   52    0.9845675  0.9804762
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

```r
stopCluster(cluster)
registerDoSEQ()
```

Now I use the portion of the training data set that I set to be the testing data as a cross-validation of the model.


```r
RFPrediction <- predict(RFModel, training3_testingData)
confusionMatrix(training3_testingData$classe,RFPrediction)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    1    0    0
##          B    5 1130    4    0    0
##          C    0   13 1011    2    0
##          D    0    0    8  955    1
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9942         
##                  95% CI : (0.9919, 0.996)
##     No Information Rate : 0.2851         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9927         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9886   0.9873   0.9979   0.9991
## Specificity            0.9998   0.9981   0.9969   0.9982   1.0000
## Pos Pred Value         0.9994   0.9921   0.9854   0.9907   1.0000
## Neg Pred Value         0.9988   0.9973   0.9973   0.9996   0.9998
## Prevalence             0.2851   0.1942   0.1740   0.1626   0.1840
## Detection Rate         0.2843   0.1920   0.1718   0.1623   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9984   0.9934   0.9921   0.9980   0.9995
```

From this cross-validation, we can see that the accuracy is 0.994 or about 99.4%.  Therefore the out-of-sample error is .006 or about 0.6%.

## Using the Model to Predict Test Data

Now, I can use the model in order to predict the outcomes of the testing data.


```r
TestingPrediction <- predict(RFModel,testing3)
TestingPrediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


