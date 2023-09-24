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