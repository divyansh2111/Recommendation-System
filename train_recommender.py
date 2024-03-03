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

sc = SparkSession.builder.appName("Product_Recommendation") \
.config ("spark.sql.shuffle.partitions", "16") \
.config("spark.driver.maxResultSize","4g") \
.config ("spark.sql.execution.arrow.enabled", "true") \
.config("spark.driver.memory", "4g") \
.config("spark.executor.cores", "4") \
.getOrCreate()

sc.sparkContext.setLogLevel("ERROR")

def prepare_data_and_train(file_path):
    df = sc.read.option('header', True).csv(file_path)
    events = df.groupBy('event_type').count().toPandas()
    df = preprocess(df)
    products = product_features(df)
    categories = category_features(df)
    relative_prices = calculate_relative_price(products)
    df = df.join(relative_prices, on='product_id')
    products = products.join(relative_prices, on='product_id')
    avg_purchase_per_view = events[events['event_type'] == 'purchase']['count'].values[0] / events[events['event_type'] == 'view']['count'].values[0]
    avg_cart_per_view = events[events['event_type'] == 'cart']['count'].values[0] / events[events['event_type'] == 'view']['count'].values[0]
    avg_purchase_per_cart = events[events['event_type'] == 'purchase']['count'].values[0] / events[events['event_type'] == 'cart']['count'].values[0]
    categories = category_smoothener(categories, avg_purchase_per_view, 'views', 'purchase_per_view', 2000)
    categories = category_smoothener(categories, avg_cart_per_view, 'views', 'cart_per_view', 2000)
    categories = category_smoothener(categories, avg_purchase_per_cart, 'carts', 'purchase_per_cart', 200)
    products = product_smoothener(products, categories, 'views', 'purchase_per_view', 1000)
    products = product_smoothener(products, categories, 'views', 'cart_per_view', 1000)
    products = product_smoothener(products, categories, 'carts', 'purchase_per_cart', 100)
    users = user_features(df)
    users.write.csv("users.csv", header=True)
    products.write.csv("products.csv", header=True)
    
    # Get the timestamp of the most recent event in the df
    last_date = df.agg(max('event_time')).collect()[0][0]
    df = df.withColumn('last_date', lit(last_date))

    # Calculate the recency of each event in terms of days
    df = df.withColumn('recency', (col('last_date').cast('double') - col('event_time').cast('double')) / 86400)
    df = df.drop('last_date')

    # Half-life decay function
    df = df.withColumn('recency_coef', expr('exp(ln(0.5)*recency/20)'))
    interactions = df.groupby(['user_id', 'product_id']).agg(sum(when(df['event_type'] == 'view', 1) * df['recency_coef']).alias('views'),
                                                         sum(when(df['event_type'] == 'cart', 1) * df['recency_coef']).alias('carts'),
                                                         sum(when(df['event_type'] == 'purchase', 1) * df['recency_coef']).alias('purchases'))
    interactions = interactions.na.fill(0)
    
    interaction_matrix = calculate_interaction_matrix(interactions)
    
    if cross_val:
        als_model = cross_validate_als(interaction_matrix)
    else:
        als_model = simple_als(interaction_matrix)
    
    als_model.save('best_als_model')
    prod_vectors = get_prod_vectors(als_model)
    brp_model = fit_LSH(product_vectors)
    
        
if __name__=="__main__":
    file_path = "ecommerce-behavior-data-from-multi-category-store/2019-Nov.csv"
    prepare_data_and_train(file_path)
