import matplotlib.pyplot as plt
%matplotlib inline
import librosa.display
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import glob
import math 
import os
JAVA_HOME = "/usr/lib/jvm/java-8-openjdk-amd64"
from handyspark import *
os.environ["JAVA_HOME"] = JAVA_HOME
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from handyspark import *
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# Create spark context
conf = SparkConf().set("spark.ui.port", "4050").set('spark.executor.memory', '4G').set('spark.driver.memory', '45G').set('spark.driver.maxResultSize', '10G')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
sql_context = SQLContext(sc)
################################################################################
# Load train data
df_train = sql_context.read.csv('Sample_Data/CQT_train.csv',
                    header=True,
                    inferSchema=True)
print("The shape of the train dataset is {:d} rows by {:d} columns".format(df_train.count(), len(df_train.columns)))
# Merge all feature columns
columns = ['pixel{:d}'.format(k) for k in range(7308)]
assembler = VectorAssembler(inputCols=columns, 
                            outputCol="features")
Combine_df = assembler.transform(df_train)
Combine_df = Combine_df.select("features", "label")
# Shuffle data
df_transform_fin = Combine_df.orderBy(rand())
train_data=df_transform_fin
print("The shape of the dataset is {:d} rows by {:d} columns".format(train_data.count(), len(train_data.columns)))
################################################################################
# Load test data
df_test = sql_context.read.csv('Sample_Data/CQT_test.csv',
                    header=True,
                    inferSchema=True)
print("The shape of the train dataset is {:d} rows by {:d} columns".format(df_test.count(), len(df_test.columns)))
# Merge all feature columns
columns = ['pixel{:d}'.format(k) for k in range(7308)]
assembler = VectorAssembler(inputCols=columns, 
                            outputCol="features")
Combine_df_test = assembler.transform(df_test)
Combine_df_test = Combine_df_test.select("features", "label")
test_data=Combine_df_test
print("The shape of the test dataset is {:d} rows by {:d} columns".format(test_data.count(), len(test_data.columns)))

################################################################################
                                                  # Logostic Regression
# Build the model
LRA = LogisticRegression(labelCol="label", featuresCol="features", maxIter=100)
LRAModel = LRA.fit(train_data)
# Evaluation on train data
trainingSummary=LRAModel.summary
print("Area Under ROC curve for train set(LR): " + str(trainingSummary.areaUnderROC))
print("Train accuracy(LR): " + str(trainingSummary.accuracy))
# Evaluation on test data
predictionsLRA = LRAModel.transform(test_data)
evaluatorLRA = BinaryClassificationEvaluator()
print('Area Under ROC curve for test set(LR)', evaluatorLRA.evaluate(predictionsLRA))
auroc = evaluatorLRA.evaluate(predictionsLRA, {evaluatorLRA.metricName: "areaUnderROC"})
auprc = evaluatorLRA.evaluate(predictionsLRA, {evaluatorLRA.metricName: "areaUnderPR"})
evaluatorLRA.metricName=""
evaluator14 = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy14 = evaluator14.evaluate(predictionsLRA)
print("Area Under PR curve for test set(LR): {:.4f}".format(auprc)) 
print("Test accuracy(LR) = %g" % (accuracy14))
################################################################################
                                                  # Support Vector machine
# Build the model
lsvc = LinearSVC(labelCol="label", featuresCol="features", maxIter=100)
lsvcModel = lsvc.fit(train_data)
# Evaluation on test data
predictionslsvc = lsvcModel.transform(test_data)
evaluatorlsvc = BinaryClassificationEvaluator(labelCol='label')
auroc = evaluatorlsvc.evaluate(predictionslsvc, {evaluatorlsvc.metricName: "areaUnderROC"})
auprc = evaluatorlsvc.evaluate(predictionslsvc, {evaluatorlsvc.metricName: "areaUnderPR"})
evaluatorlsvc.metricName=""
print("Area under ROC Curve for test set(SVM): {:.4f}".format(auroc))
print("Area under PR Curve for test set(SVM): {:.4f}".format(auprc))
evaluator12 = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy12 = evaluator12.evaluate(predictionslsvc)
print("Test accuracy(SVM) = %g" % (accuracy12))
################################################################################
                                                  # Naive Bayes
# Build the model
nb = NaiveBayes(modelType="multinomial")
model_nb = nb.fit(train_data)
# Evaluation on test data
predictions_nb = model_nb.transform(test_data)
evaluator13 = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy13 = evaluator13.evaluate(predictions_nb)
print("Test accuracy(NB) = %g" % (accuracy13))
predictions_nb.toHandy().cols[['probability', 'prediction', 'label']][:1088]
bcmnb = BinaryClassificationMetrics(predictions_nb, scoreCol='probability', labelCol='label')
print("Area under ROC Curve for test set(NB): {:.4f}".format(bcmnb.areaUnderROC))
print("Area under PR Curve for test set(NB): {:.4f}".format(bcmnb.areaUnderPR))
################################################################################
                                                  # Multi Layer Perceptron
# Extraction of the number of classes and input dimensions
nb_classes = train_data.select("label").distinct().count()
input_dim = len(train_data.select("features").first()[0])
# Build the model
layers = [input_dim,40, 2]
FNN = MultilayerPerceptronClassifier(labelCol="label", featuresCol="features",maxIter=100, layers=layers, blockSize=32, seed=1234)     
modelFNN = FNN.fit(train_data)
# Evaluation on test data
predictionsFNN = modelFNN.transform(test_data)
print("number of trainable weights in the network=" + str(len(modelFNN.weights)))
evaluatorFNN = BinaryClassificationEvaluator(labelCol='label')
auroc = evaluatorFNN.evaluate(predictionsFNN, {evaluatorFNN.metricName: "areaUnderROC"})
auprc = evaluatorFNN.evaluate(predictionsFNN, {evaluatorFNN.metricName: "areaUnderPR"})
evaluatorFNN.metricName=""
print("Area under ROC Curve for test set(MLP): {:.4f}".format(auroc))
print("Area under PR Curve for test set(MLP): {:.4f}".format(auprc))
evaluator11 = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy11 = evaluator11.evaluate(predictionsFNN)
print("Test accuracy(MLP) = %g" % (accuracy11))