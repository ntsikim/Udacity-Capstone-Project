#!/usr/bin/env python
# coding: utf-8

"""
Sparkify Churn Prediction Script
Created by Ntsikelelo Myesi for the Data Scientist Nano Degree

This script performs data preprocessing, feature engineering, model training, and evaluation
for predicting customer churn using Spark.

The main steps are:
1. Import Libraries
2. Create a Spark Session
3. Load and Clean Dataset
4. Define Churn
5. Exploratory Data Analysis
6. Feature Engineering
7. Model Training and Evaluation
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, desc, sum as Fsum, count as Fcount
from pyspark.sql.types import IntegerType, StringType, FloatType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt

def create_spark_session():
    """
    Create and return a Spark session
    """
    spark = SparkSession.builder \
        .appName("Sparkify Churn Prediction") \
        .getOrCreate()
    return spark

def load_and_clean_data(spark, file_path):
    """
    Load and clean the dataset
    Args:
    - spark: Spark session
    - file_path: Path to the dataset file
    
    Returns:
    - df: Cleaned Spark DataFrame
    """
    df = spark.read.json(file_path)
    df = df.dropna(how="any", subset=["userId", "sessionId"])
    df = df.filter(df["userId"] != "")
    return df

def define_churn(df):
    """
    Define churn by creating a 'churn' column
    Args:
    - df: Input Spark DataFrame
    
    Returns:
    - df: Spark DataFrame with 'churn' column
    """
    churn_flag = df.filter(df.page == "Cancellation Confirmation") \
                   .select("userId").dropDuplicates() \
                   .withColumn("churn", col("userId").cast(IntegerType()).alias("churn"))
    df = df.join(churn_flag, on="userId", how="left")
    df = df.na.fill({"churn": 0})
    return df

def extract_features(df):
    """
    Extract features for model training
    Args:
    - df: Input Spark DataFrame
    
    Returns:
    - features: Spark DataFrame with extracted features
    """
    num_songs = df.filter(df.page == "NextSong").groupBy("userId").agg(Fcount("page").alias("num_songs"))
    total_session_length = df.groupBy("userId").agg(Fsum("length").alias("total_session_length"))
    num_thumbs_up = df.filter(df.page == "Thumbs Up").groupBy("userId").agg(Fcount("page").alias("num_thumbs_up"))
    num_thumbs_down = df.filter(df.page == "Thumbs Down").groupBy("userId").agg(Fcount("page").alias("num_thumbs_down"))
    num_add_friend = df.filter(df.page == "Add Friend").groupBy("userId").agg(Fcount("page").alias("num_add_friend"))
    num_add_playlist = df.filter(df.page == "Add to Playlist").groupBy("userId").agg(Fcount("page").alias("num_add_playlist"))
    
    features = num_songs.join(total_session_length, on="userId", how="left") \
                        .join(num_thumbs_up, on="userId", how="left") \
                        .join(num_thumbs_down, on="userId", how="left") \
                        .join(num_add_friend, on="userId", how="left") \
                        .join(num_add_playlist, on="userId", how="left") \
                        .join(df.select("userId", "churn").distinct(), on="userId", how="left")
    
    features = features.na.fill(0)
    return features

def assemble_and_scale_features(features):
    """
    Assemble and scale features
    Args:
    - features: Input Spark DataFrame with extracted features
    
    Returns:
    - assembled_data: Spark DataFrame with assembled and scaled features
    """
    feature_columns = ["num_songs", "total_session_length", "num_thumbs_up", "num_thumbs_down", "num_add_friend", "num_add_playlist"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="unscaled_features")
    scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features", withStd=True)
    
    assembled_data = assembler.transform(features)
    assembled_data = scaler.fit(assembled_data).transform(assembled_data)
    return assembled_data

def train_and_evaluate_model(model, train, validation):
    """
    Train and evaluate a model
    Args:
    - model: Machine learning model
    - train: Training dataset
    - validation: Validation dataset
    
    Returns:
    - model_fitted: Trained model
    - f1_score: F1 score on validation set
    """
    pipeline = Pipeline(stages=[assembler, scaler, model])
    model_fitted = pipeline.fit(train)
    predictions = model_fitted.transform(validation)
    evaluator = MulticlassClassificationEvaluator(labelCol="churn", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    return model_fitted, f1_score

def main():
    spark = create_spark_session()
    file_path = "mini_sparkify_event_data.json"  # Path to your dataset
    df = load_and_clean_data(spark, file_path)
    df = define_churn(df)
    
    features = extract_features(df)
    assembled_data = assemble_and_scale_features(features)
    
    # Split data into training and testing sets
    train, test = assembled_data.randomSplit([0.8, 0.2], seed=42)
    
    # Initialize models
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="churn", maxIter=10)
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="churn")
    gbt = GBTClassifier(featuresCol="scaled_features", labelCol="churn", maxIter=10)
    
    # Train and evaluate models
    models = {"Logistic Regression": lr, "Random Forest": rf, "Gradient-Boosted Trees": gbt}
    results = {}
    for name, model in models.items():
        fitted_model, f1 = train_and_evaluate_model(model, train, validation)
        results[name] = {"model": fitted_model, "f1_score": f1}
    
    # Select the best model based on F1 score
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = results[best_model_name]["model"]
    best_f1_score = results[best_model_name]["f1_score"]
    
    # Evaluate the best model on the test set
    test_predictions = best_model.transform(test)
    test_f1_score = evaluator.evaluate(test_predictions)
    
    print(f"Best Model: {best_model_name}")
    print(f"Validation F1 Score: {best_f1_score}")
    print(f"Test F1 Score: {test_f1_score}")
    
    # Display sample predictions
    test_predictions.select("userId", "churn", "prediction", "probability").show(5)
    
if __name__ == "__main__":
    main()
