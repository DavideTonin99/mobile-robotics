# Mobile Robotics Anomaly Detection

In this project, we introduce a way to manage a dataset of time series, focused on data of a mobile robot, with the goal of perform anomaly detection, eventually in real-time, on the working robot.

The dataset is composed of 10 time series with anomalies, and 7 nominal without anomalies, divided in the following way:

*Mettere qui la tabella del dataset.*

Our goal was to create a structure in the code that would allow us to apply easily some pre-processing operations and several methods of anomaly detection (both models and identification of the threshold).

The code is structured in the following way:
- TimeSeries class
- Dataset class
- main.py
- main_pca
- ...

Pre-processing operations:
- sliding window
- normalization
- moving average

We tried these methods:
- pca
- ...

Experiments and results:
*the best method*
*un par di grafici*


    Course          Mobile Robotics
    Davide Tonin    VR480503
    Emanuele Feola  VR474837


--- 
versione feola

## Goal
The aim of this project is to perform offline anomaly detection on a mobile robot.
The code is able to process a time series dataset by easily applying pre-processing operations, several anomaly detection algorithms (PCA, NN, SVM, etc) and several threshold identification methods (quantile, one class svm, etc).
The code is structured in a way such that online anomaly detection can be easily performed, once an real-time source of data is provided.

## Code Structure
The code is structured in the following way:
- TimeSeries class
	- breve spiegazione (cosa fa)
- Dataset class
	- breve spiegazione 
- main.py
- main_pca
- ...

## Pre-processing
Pre-processing operations:
- sliding window
	- parametri: size, stride, ...
- normalization
- moving average

## Pipeline
- training
- evluation
- testing

## Anomaly Detection algorithms
The following methods have been tested and executed:
- pca: applicazione della pca e ricostruzione del segnale, confronto degli errori di ricostruzione e identificazione delle anomalie basate su​  
	- threshold calcolata con quantili​
	- one class svm​
	- local outlier factor​​
- neural network "linear regression": training su traiettorie nominali e test su traiettorie nominali ed anomale (modello preso dal notebook colab)​
- one class svm: training su traiettorie nominali e anomale, test su traiettorie nominali ed anomale (non presenti nel training set).
	- Sia con e senza la pca per ridurre la dimensione dei segnali​
- local outlier factor: training su traiettorie nominali e test su traiettorie nominali ed anomale​ ​

## Metrics and scoring: quantifying the quality of predictions
If at least one time window in the trajectory is predicted as anomalous, then the entire trajectory is labeled "anomalous".
If none of the time windows are predicted as anomalous, then the trajectory is labeled "nominal".
If the trajectory 

## Experiments and results:
Among the different anomaly detection algorithms, the PCA is the one with the best results. 
*the best method*
*un par di grafici*

## Authors
    Course          Mobile Robotics
    Academic year   2022/2023
    Davide Tonin    VR480503
    Emanuele Feola  VR474837

