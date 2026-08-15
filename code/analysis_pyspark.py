#!/usr/bin/env python3
"""
================================================================================
IST3134 Big Data Analytics in the Cloud — Group Assignment May 2026
================================================================================
Title   : US Flight Delay Pattern Analysis — PySpark DataFrame API
Authors : Lee Zong Xun | Bernice Wong Jian Xuan
Dataset : BTS Airline Delay and Cancellation Data (2017-2018)
Source  : https://www.kaggle.com/datasets/yuanyuwendymu/
          airline-delay-and-cancellation-data-2009-2018
--------------------------------------------------------------------------------
Infrastructure:
    Platform  : Amazon EMR emr-7.x
    Cluster   : 1x primary + 2x core m5.xlarge nodes
    Framework : Apache Spark 3.5.6, Python 3.11, YARN
    Storage   : HDFS (processing) + S3 (persistent backup)
--------------------------------------------------------------------------------
How to run:
    spark-submit --master yarn --deploy-mode client \
      --num-executors 2 --executor-memory 1g \
      analysis_pyspark.py hdfs:///user/hadoop/flights/
--------------------------------------------------------------------------------
Analyses performed:
    1. Total flights per airline (groupBy + count)
    2. Average arrival delay per airline (filter + groupBy + agg)
    3. Total delay minutes by cause (agg + sum across 5 delay columns)
    4. Top 10 worst airports by departure delay (filter + groupBy + filter)
    5. Cancellations by reason code (filter + groupBy + count)
    6. Monthly average delay trend (withColumn month + groupBy)
    7. Top 3 most delayed routes per airline (window function + row_number)
--------------------------------------------------------------------------------
Output:
    - Console output saved to ~/pyspark_output.txt
    - Parquet results saved to hdfs:///user/hadoop/flights/output/
    - Backed up to s3://[bucket]/flights/output/
================================================================================
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("FlightDelayAnalysis-DataFrame").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Input path passed as command-line argument (HDFS or S3)
INPUT_PATH = sys.argv[1]

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

# Read CSV with header and automatic schema inference
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(INPUT_PATH))

df.printSchema()
total_rows = df.count()
print(f"Total flights loaded: {total_rows:,}")

# ── Analysis 1: Total flights per airline ────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 1: Total Flights per Airline")
print("=" * 60)

flights_per_airline = (df.groupBy("OP_CARRIER")
    # Count all rows per airline carrier code
    .agg(F.count("*").alias("total_flights"))
    # Sort descending to show busiest airlines first
    .orderBy(F.desc("total_flights")))
flights_per_airline.show(10)

# ── Analysis 2: Average arrival delay per airline ─────────────
print("\n" + "=" * 60)
print("ANALYSIS 2: Average Arrival Delay per Airline (delayed only)")
print("=" * 60)

# Filter only delayed flights to avoid skewing average with on-time flights
avg_delay = (df.filter(F.col("ARR_DELAY") > 0)
    # Group by airline IATA code
    .groupBy("OP_CARRIER")
    .agg(
        F.count("*").alias("delayed_flights"),
        # Round to 2 decimal places for readability
        F.round(F.avg("ARR_DELAY"), 2).alias("avg_delay_min"),
        F.round(F.max("ARR_DELAY"), 2).alias("max_delay_min")
    )
    # Sort worst performers first
    .orderBy(F.desc("avg_delay_min")))
avg_delay.show(10)

# ── Analysis 3: Delay reasons breakdown ──────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 3: Total Minutes Lost per Delay Cause")
print("=" * 60)

# Sum all five official BTS delay cause columns across entire dataset
delay_causes = df.agg(
    F.round(F.sum("CARRIER_DELAY"), 0).alias("carrier_delay_min"),
    F.round(F.sum("WEATHER_DELAY"), 0).alias("weather_delay_min"),
    F.round(F.sum("NAS_DELAY"), 0).alias("nas_delay_min"),
    F.round(F.sum("SECURITY_DELAY"), 0).alias("security_delay_min"),
    # Late aircraft = cascading delays from previous flights
    F.round(F.sum("LATE_AIRCRAFT_DELAY"), 0).alias("late_aircraft_delay_min")
)
delay_causes.show()

# ── Analysis 4: Top 10 worst airports by avg departure delay ─
print("\n" + "=" * 60)
print("ANALYSIS 4: Top 10 Worst Origin Airports by Avg Departure Delay")
print("=" * 60)

worst_airports = (df.filter(F.col("DEP_DELAY") > 0)
    .groupBy("ORIGIN")
    .agg(
        F.count("*").alias("delayed_departures"),
        F.round(F.avg("DEP_DELAY"), 2).alias("avg_dep_delay_min")
    )
    # Minimum threshold of 1000 delayed departures to exclude
    # low-volume airports where one event skews the average
    .filter(F.col("delayed_departures") > 1000)
    .orderBy(F.desc("avg_dep_delay_min")))
worst_airports.show(10)

# ── Analysis 5: Cancellation reasons ─────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 5: Cancellations by Reason")
print("=" * 60)

# Cancellation codes: A=Carrier, B=Weather, C=NAS, D=Security
cancellations = (df.filter(F.col("CANCELLED") == 1)
    .groupBy("CANCELLATION_CODE")
    .agg(F.count("*").alias("cancelled_flights"))
    .orderBy(F.desc("cancelled_flights")))
cancellations.show()

# ── Analysis 6: Monthly delay trend ──────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS 6: Monthly Average Arrival Delay Trend")
print("=" * 60)

# Extract month number from FL_DATE timestamp column
monthly = (df.filter(F.col("ARR_DELAY") > 0)
    .withColumn("month", F.month("FL_DATE"))
    .groupBy("month")
    .agg(F.round(F.avg("ARR_DELAY"), 2).alias("avg_delay_min"))
    # Sort chronologically to show seasonal pattern
    .orderBy("month"))
monthly.show(12)

# ── Analysis 7: Window function — top 3 worst routes per airline
print("\n" + "=" * 60)
print("ANALYSIS 7: Top 3 Most Delayed Routes per Airline (Window Function)")
print("=" * 60)

# First aggregate average delay per route
route_delay = (df.filter(F.col("ARR_DELAY") > 0)
    .groupBy("OP_CARRIER", "ORIGIN", "DEST")
    .agg(F.round(F.avg("ARR_DELAY"), 2).alias("avg_delay_min"),
         F.count("*").alias("flight_count"))
    # Minimum 50 flights per route for statistical reliability
    .filter(F.col("flight_count") > 50))

# Window function partitions by airline and ranks routes by delay
# row_number() gives sequential rank without gaps (unlike rank())
w = Window.partitionBy("OP_CARRIER").orderBy(F.desc("avg_delay_min"))
top_routes = (route_delay
    .withColumn("rank", F.row_number().over(w))
    # Keep only top 3 routes per airline
    .filter(F.col("rank") <= 3)
    .orderBy("OP_CARRIER", "rank"))
top_routes.show(30)

# ── Save results to HDFS as Parquet ──────────────────────────
print("\n" + "=" * 60)
print("SAVING RESULTS TO HDFS AS PARQUET")
print("=" * 60)

# Parquet is columnar and compressed — much smaller than CSV
avg_delay.write.mode("overwrite").parquet(
    "hdfs:///user/hadoop/flights/output/avg_delay")
worst_airports.write.mode("overwrite").parquet(
    "hdfs:///user/hadoop/flights/output/worst_airports")
monthly.write.mode("overwrite").parquet(
    "hdfs:///user/hadoop/flights/output/monthly_trend")

print("Results saved successfully.")
spark.stop()
