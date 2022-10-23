
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
# conf = pyspark.SparkConf().setAppName("synapseml_script").setMaster("local[*]")
# sc = pyspark.SparkContext(conf=conf)
# spark = SparkSession(sc)

# Configure spark logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configure spark ui
spark = SparkSession.builder.master("local").\
appName("Word Count").\
config("spark.driver.bindAddress","localhost").\
config("spark.ui.port","4050").\
getOrCreate()

input("Press anything to terminate")
spark.stop()

exit()

# Please use 0.10.1 version for Spark3.2 and 0.9.5-13-d1b51517-SNAPSHOT version for Spark3.1
spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
            .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.10.1") \
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
            .getOrCreate()


import synapse.ml




from synapse.ml.automl import *
from synapse.ml.train import *
from pyspark.ml.classification import RandomForestClassifier

df = (spark.createDataFrame([
    (0, 2, 0.50, 0.60, 0),
    (1, 3, 0.40, 0.50, 1),
    (0, 4, 0.78, 0.99, 2),
    (1, 5, 0.12, 0.34, 3),
    (0, 1, 0.50, 0.60, 0),
    (1, 3, 0.40, 0.50, 1),
    (0, 3, 0.78, 0.99, 2),
    (1, 4, 0.12, 0.34, 3),
    (0, 0, 0.50, 0.60, 0),
    (1, 2, 0.40, 0.50, 1),
    (0, 3, 0.78, 0.99, 2),
    (1, 4, 0.12, 0.34, 3)
], ["Label", "col1", "col2", "col3", "col4"]))

# mocking models
randomForestClassifier = (TrainClassifier()
      .setModel(RandomForestClassifier()
        .setMaxBins(32)
        .setMaxDepth(5)
        .setMinInfoGain(0.0)
        .setMinInstancesPerNode(1)
        .setNumTrees(20)
        .setSubsamplingRate(1.0)
        .setSeed(0))
      .setFeaturesCol("mlfeatures")
      .setLabelCol("Label"))
model = randomForestClassifier.fit(df)

findBestModel = (FindBestModel()
  .setModels([model, model])
  .setEvaluationMetric("accuracy"))
bestModel = findBestModel.fit(df)
bestModel.transform(df).show()