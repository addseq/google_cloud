#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import pickle
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Parameters for Cloud SQL Connection
CLOUDSQL_INSTANCE_IP = '204.212.175.30'   # CHANGE (database server IP)
CLOUDSQL_DB_NAME = 'recommendation_spark'
CLOUDSQL_USER = 'root'
CLOUDSQL_PWD = 'root'  # CHANGE

#Optionally pass in the Cloud SQL args
#CLOUDSQL_INSTANCE_IP = sys.argv[1]
#CLOUDSQL_DB_NAME = sys.argv[2]
#CLOUDSQL_USER = sys.argv[3]
#CLOUDSQL_PWD  = sys.argv[4]

# Parameters for Model Training
UNIQUE_USER_IDS = 150  # CHANGE
LIMIT_TOP_PREDICTIONS = 5  # CHANGE
ALS_RANK = 20  # Number of unknown factors that led user to give a rating (E.g. desk, location, age)
ALS_ITERATIONS = 20  # Number of times the training will run for various combos of Rank and Lambda
ALS_LAMBDA = 0.01  # A regularization parameter to prevent overfitting. Higher value means lower overfitting but greater bias
TABLE_SERVICES = "Services"
TABLE_RATINGS = "Rating_Services"
TABLE_RECOMMENDATIONS = "Recommendation"
print("Running with Parameters- "
      "\nUnique Users: " + str(UNIQUE_USER_IDS) +
      "\nTop Predictions Limit: " + str(LIMIT_TOP_PREDICTIONS) +
      "\nALS Rank: " + str(ALS_RANK) +
      "\nALS Iterations: " + str(ALS_ITERATIONS) +
      "\nALS Lambda: " + str(ALS_LAMBDA) +
      "\nInput Table: " + TABLE_SERVICES + ", " + TABLE_RATINGS +
      "\nOutput Table: " + TABLE_RECOMMENDATIONS)

# Initialize SparkSQL Context
conf = SparkConf().setAppName("train_model")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# Specify JDBC Credentials to connect to Cloud SQL via SparkSQL Context
jdbcDriver = 'com.mysql.jdbc.Driver'
jdbcUrl = 'jdbc:mysql://%s:3306/%s?user=%s&password=%s' % (CLOUDSQL_INSTANCE_IP, CLOUDSQL_DB_NAME, CLOUDSQL_USER, CLOUDSQL_PWD)

# Checkpointing helps prevent stack overflow errors
sc.setCheckpointDir('checkpoint/')

# Read the Ratings_Services and Services data from Cloud SQL as DataFrames
dfServices = sqlContext.read.format('jdbc').options(driver=jdbcDriver, url=jdbcUrl, dbtable=TABLE_SERVICES).load()
dfRatings = sqlContext.read.format('jdbc').options(driver=jdbcDriver, url=jdbcUrl, dbtable=TABLE_RATINGS).load()
print("Read ...")

# Split the Dataset randomly to Train (70%) and Test (30%) datasets
#rddTrainData, rddTestData = dfRatings.rdd.randomSplit([7, 3])

# Train the Alternating Least Squares (ALS) Model from Spark MLlib with tunable parameters Rank, Iterations, Lambda
# The Rating table should follow the order of service_id, user_id, rating as ALS works with defined product-user pairs
model = ALS.train(dfRatings.rdd, ALS_RANK, ALS_ITERATIONS, ALS_LAMBDA)
print("Trained ...")

# Use this model to predict what the user would rate Services that he has not rated yet
allPredictions = None
for USER_ID in xrange(0, UNIQUE_USER_IDS):
    # Returns all the Service Ratings given by each User
    dfUserRatings = dfRatings.filter(dfRatings.userId == USER_ID).rdd.map(lambda r: r.service_id).collect()
    # Return only Services that have not yet been rated by the User
    rddPotential = dfServices.rdd.filter(lambda x: x[0] not in dfUserRatings)
    pairsPotential = rddPotential.map(lambda x: (USER_ID, x[0]))
    # Calculate all predictions
    predictions = model.predictAll(pairsPotential).map(lambda p: (str(p[0]), str(p[1]), float(p[2])))
    # Return only top 5 predictions
    topPredictions = predictions.takeOrdered(LIMIT_TOP_PREDICTIONS, key=lambda x: -x[2])
    print("Predicted for User: " + str(USER_ID), " Top Predictions: " + str(topPredictions))
    if (allPredictions == None):
        allPredictions = topPredictions
    else:
        allPredictions.extend(topPredictions)
print("Predicted ...")

# Write the results of the ML Model from Dataframe back to the Cloud SQL Recommendation table
schema = StructType([StructField("user_id", StringType(), True), StructField("service_id", StringType(), True), StructField("prediction", FloatType(), True)])
dfToSave = sqlContext.createDataFrame(allPredictions, schema)
dfToSave.write.jdbc(url=jdbcUrl, table=TABLE_RECOMMENDATIONS, mode='overwrite')
print("Wrote back to Cloud SQL ...")