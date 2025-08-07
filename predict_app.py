# Import libraries
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Make sure the right number of command line arguments
if len(sys.argv) != 2:
    print("Usage: python predict.py <path_to_test_file>")
    sys.exit(1)

test_file_path = sys.argv[1]

# Start Spark
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Load the testing data
test_df = spark.read.csv(test_file_path, sep=";", header=True, inferSchema=True)
test_df = test_df.toDF(*[c.strip().replace('"', '') for c in test_df.columns])
# Rename the column
test_df = test_df.withColumnRenamed("quality", "label")

# Load the trained  model
model = PipelineModel.load("/mnt/nfs_project/model")

# use model to make predications
predictions = model.transform(test_df)

# Evaluate 
evaluator = MulticlassClassificationEvaluator(metricName="f1")
f1_score = evaluator.evaluate(predictions)

# Print F1 Score
print(f"F1 Score on test dataset: {f1_score}")

spark.stop()

