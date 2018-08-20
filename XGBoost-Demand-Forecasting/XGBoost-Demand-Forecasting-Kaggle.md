## Store Item Demand using XGBoost

In this code we used, the **XGBoost** technique which is a tuned implementation of a gradient boosting method (https://hackernoon.com/gradient-boosting-and-xgboost-90862daa6c77) , and the parameters were tuned previously using GridSearchCV, to improve the SMAPE metric, there was 3 attempt to improve the result but the last one doesn't represent an improvement in the test set.

This technique was near to DNN and with the learning rate that was used, was even faster than DNN code version.

I think sometimes the gradien boosting methods could be used, with faster running times and simmilar accuracies to DNN, at the moment of writing this, some better results in the Kaggle contest are using
LigthGBM methods which is similar to XGBoost method.

Some of the time variables was taken from antoher kernels, this code is more an attempt to have a comparison code for our DNN Time series version, we didn't spend much time adding new variables or adjusting the parameters.

