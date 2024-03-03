from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.feature import BucketedRandomProjectionLSH

def save_ml_component(ml_component, path):
    ml_component.write().overwrite().save(path)

def load_ml_component(path):
    return MLReadable.load(path)
    
def cross_validate_als(interaction_matrix):
    
    # Define the ALS model
    als = ALS(userCol='user_id', itemCol='product_id', ratingCol='interaction', 
              nonnegative=True, coldStartStrategy='drop', implicitPrefs=True)

    # Define the parameter grid for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [5, 10, 15, 20]) \   
        .addGrid(als.regParam, [0.005, 0.01, 0.05, 0.1]) \
        .addGrid(als.alpha, [0, 1.0, 5.0]) \
        .build()
        
    # Define the evaluator for computing the evaluation metrics
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='interaction', predictionCol='prediction')

    # Define the cross-validator for performing 5-fold cross-validation
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5, collectSubModels=False)

    # Define the pipeline for fitting the model and evaluating it
    pipeline = Pipeline(stages=[cv])

    # Fit the pipeline on the data and evaluate the model
    model = pipeline.fit(interaction_matrix)
    
    return model.stages[0].bestModel
    
def simple_als(interaction_matrix):

    # Train-test split
    (train, test) = interaction_matrix.randomSplit([0.8, 0.2])

    # Initialize the model with the optimized parameters
    als = ALS(userCol='user_id', itemCol='product_id', ratingCol='interaction', 
              alpha=1, regParam=0.005, rank=15, implicitPrefs=True, 
              nonnegative=True, coldStartStrategy='drop')

    # Fit the ALS model on the ratings data
    model = als.fit(train)

    # Make predictions
    predictions = model.transform(test)

    # Calculate the RMSE and MAE metrics
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='interaction', predictionCol='prediction')
    rmse = evaluator.evaluate(predictions)
    mae = evaluator.setMetricName('mae').evaluate(predictions)
    print('test rmse:' + str(rmse) + ' mae:' + str(mae))

    return model

def get_prod_vectors(als_model):
    product_vectors = als_model.itemFactors
    product_vectors = product_vectors.rdd.map(lambda row: (row[0], Vectors.dense(row[1])))
    product_vectors = product_vectors.toDF(['product_id', 'features'])

    # Use VectorAssembler to convert the features column into a dense vector column
    assembler = VectorAssembler(inputCols=['features'], outputCol='vector')
    product_vectors = assembler.transform(product_vectors)

    # Normalize the vectors
    normalizer = Normalizer(inputCol='vector', outputCol='norm_vector')
    product_vectors = normalizer.transform(product_vectors)
    save_ml_component(product_vectors, 'product_vectors')
    return product_vectors

def fit_LSH(product_vectors):
    brp = BucketedRandomProjectionLSH(inputCol="norm_vector", outputCol="neighbors", numHashTables=5, bucketLength=0.1)
    brp_model = brp.fit(product_vectors)
    save_ml_component(brp_model, 'brp_model')
    return brp_model
