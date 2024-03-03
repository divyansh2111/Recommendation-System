import pandas as pd
import numpy as np
import os
import sys
import math
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import pyarrow
from data import *
from model import *
from utils import *

sc = SparkSession.builder.appName("Product_Recommendation") \
.config ("spark.sql.shuffle.partitions", "16") \
.config("spark.driver.maxResultSize","4g") \
.config ("spark.sql.execution.arrow.enabled", "true") \
.config("spark.driver.memory", "4g") \
.config("spark.executor.cores", "4") \
.getOrCreate()

sc.sparkContext.setLogLevel("ERROR")

#Load Models and files
model = ALSModel.load('best_als_model')
products = sc.read.csv("products.csv", header=True, inferSchema=True)
users = sc.read.csv("users.csv", header=True, inferSchema=True)
product_vectors = load_ml_component('product_vectors')
brp_model = load_ml_component('brp_model')


def get_user_recom(user_subset):
    recommendations = sc.createDataFrame([(user, 0) for user in user_subset], ['user_id', 'product_id'])
    recommendations = als_model.recommendForUserSubset(recommendations, 500)

    recs_for_user_1 = sc.createDataFrame(recommendations.collect()[1][1])
    recs_user = calculate_recommendation_scores_for_user(user_subset[0], recs_for_user_1, products, users)
    non_interacted_products = recs_user.join(interactions.filter(col('user_id') == user_subset[0]), on='product_id', how='leftanti')
    return non_interacted_products

def get_product_recom(product_id):
    query = product_vectors.filter(col('product_id') == product_id).select('norm_vector').first()[0]
    neighbors = brp_model.approxNearestNeighbors(product_vectors, query, numNearestNeighbors=50)
    recs_product = calculate_recommendation_scores_for_products(neighbors.select('product_id', 'distCol'), products)
    return recs_product
    
def get_recom(user_id, product_id):
    recs_user = get_user_recom([user_id])
    recs_product = get_product_recom(product_id)
    
    recs_user = recs_user.withColumnRenamed('recommendation_score', 'recommendation_score_user')
    recs_paired = recs_product.join(recs_user['product_id', 'recommendation_score_user'], on='product_id', how='left')
    min_user_score = recs_paired.select(min('recommendation_score_user')).collect()[0][0]
    recs_paired = recs_paired.na.fill(min_user_score * 0.9)
    recs_paired = recs_paired.withColumn('paired_score', col('recommendation_score') * 0.5 + col('recommendation_score_user') * 0.5)
    final_result = recs_paired.sort('paired_score', ascending=False).toPandas()
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    final_result.to_excel(f"outputs/{str(user_id)}_{str(product_id)}.xlsx", index=False)
    return final_result
    
if __name__=="__main__":
    user_id = 564068124
    product_id = 5100067
    df = get_recom(user_id, product_id)