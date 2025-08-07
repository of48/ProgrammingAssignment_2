# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Start Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

try:
    # Load the CSV training dataset 
    train_data = spark.read.csv("/mnt/nfs_project/TrainingDataset.csv", sep=";", header=True, inferSchema=True)

    # Remove the extra quotes, spaces in column name
    clean_train_cols = [col_name.strip().replace('"', '') for col_name in train_data.columns]
    train_data = train_data.toDF(*clean_train_cols)

    # Loading the CSV validation dataset
    val_data = spark.read.csv("/mnt/nfs_project/ValidationDataset.csv", sep=";", header=True, inferSchema=True)
    clean_val_cols = [col_name.strip().replace('"', '') for col_name in val_data.columns]
    val_data = val_data.toDF(*clean_val_cols)

    # features are all columns except the quality
    feature_columns = train_data.columns[:-1]
    label_column = "quality"

    # Create a feature vector from input columns
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Rename label column for the MLlib
    train_data = train_data.withColumnRenamed(label_column, "label")
    val_data = val_data.withColumnRenamed(label_column, "label")

    # Creating Logistic Regression model
    lr = LogisticRegression(maxIter=100)

    # Creating Random Forest model
    rf = RandomForestClassifier(numTrees=100)

    # Build pipelines & the assembler
    lr_pipeline = Pipeline(stages=[assembler, lr])
    rf_pipeline = Pipeline(stages=[assembler, rf])

    # Training both models thru the pipeline
    lr_model = lr_pipeline.fit(train_data)
    rf_model = rf_pipeline.fit(train_data)

    # Evaluate the models on validation data
    evaluator = MulticlassClassificationEvaluator(metricName="f1")

    lr_predictions = lr_model.transform(val_data)
    rf_predictions = rf_model.transform(val_data)

    lr_f1_score = evaluator.evaluate(lr_predictions)
    rf_f1_score = evaluator.evaluate(rf_predictions)

    # Printing out the F1 scores to compare
    print(f"Logistic_Regression_Model F1 Score: {lr_f1_score}")
    print(f"Random_Forest_Classifier_Model F1 Score: {rf_f1_score}")

    # Choose the best model score
    if lr_f1_score > rf_f1_score:
        best_model = lr_model
    else:
        best_model = rf_model

    # Save best model
    best_model.write().overwrite().save("/mnt/nfs_project/model")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Stop spark
    spark.stop()

