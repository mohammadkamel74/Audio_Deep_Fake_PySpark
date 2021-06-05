# Audio_Deep_Fake_PySpark

In this project an attempt is made to train models that are capable to detect fake or real audio files. In this project Logistic Regression, Support Vector Machine, Naive Bayes and Multilayer Perceptron are used as models. This project is implemented in pyspark framework and could be a very good starting point for those who want to learn big data frameworks like pyspark.
 
## Installation

#### Install the required Python packages:
```
pip install -r requirements.txt
apt install openjdk-8-jdk-headless -qq
```
## Feature extraction from audio files

In this project an attempt is made to compute the constant-Q transform of audio signals and use these CQT as the input of our models.

Please see the following links for extracting audio features:

[CQT](http://librosa.org/doc/main/generated/librosa.cqt.html)

## Data

A very light weight version of data is provided in    folder that guide you have to use CQT to make your dataset and use it for trianing. It worth mentioning that the provoded data is just sample files to test the approach. For investigation you must build your own rich dataset.
