#!/usr/bin/env python3
"""
================================================================================
IST3134 Big Data Analytics in the Cloud — Group Assignment May 2026
================================================================================
Title   : US Flight Delay Pattern Analysis — Spark SQL
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
      analysis_sql.py hdfs:///user/hadoop/flights/
--------------------------------------------------------------------------------
Note:
    This script produces IDENTICAL results to analysis_pyspark.py
    Both compile to the same physical plan via the Catalyst optimizer
    Spark SQL is preferred when sharing queries with SQL-only colleagues
================================================================================
"""
import sys
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FlightDelayAnalysis-SQL").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Input path passed as command-line argument
INPUT_PATH = sys.argv[1]

# Read CSV and register as a temporary SQL view
# createOrReplaceTempView makes the DataFrame queryable via SQL
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(INPUT_PATH))

df.createOrReplaceTempView("flights")
print(f"Total rows loaded: {df.count():,}")
print("Registered DataFrame as SQL view: 'flights'")

# ── SQL Analysis 1: Total flights per airline ─────────────────
print("=" * 60)
print("SQL ANALYSIS 1: Total Flights per Airline")
print("=" * 60)

spark.sql("""
    SELECT OP_CARRIER,
           COUNT(*) AS total_flights
    FROM flights
    GROUP BY OP_CARRIER
    ORDER BY total_flights DESC
    LIMIT 10
""").show()

# ── SQL Analysis 2: Average arrival delay per airline ─────────
print("=" * 60)
print("SQL ANALYSIS 2: Average Arrival Delay per Airline")
print("=" * 60)

# WHERE clause filters to delayed flights only
# ROUND() for consistent 2 decimal place formatting
spark.sql("""
    SELECT OP_CARRIER,
           COUNT(*) AS delayed_flights,
           ROUND(AVG(ARR_DELAY), 2) AS avg_delay_min,
           ROUND(MAX(ARR_DELAY), 2) AS max_delay_min
    FROM flights
    WHERE ARR_DELAY > 0
    GROUP BY OP_CARRIER
    ORDER BY avg_delay_min DESC
    LIMIT 10
""").show()

# ── SQL Analysis 3: Delay cause breakdown ────────────────────
print("=" * 60)
print("SQL ANALYSIS 3: Total Minutes Lost per Delay Cause")
print("=" * 60)

# SUM across all five official BTS delay cause columns
spark.sql("""
    SELECT ROUND(SUM(CARRIER_DELAY), 0)       AS carrier_delay_min,
           ROUND(SUM(WEATHER_DELAY), 0)       AS weather_delay_min,
           ROUND(SUM(NAS_DELAY), 0)           AS nas_delay_min,
           ROUND(SUM(SECURITY_DELAY), 0)      AS security_delay_min,
           ROUND(SUM(LATE_AIRCRAFT_DELAY), 0) AS late_aircraft_delay_min
    FROM flights
""").show()

# ── SQL Analysis 4: Worst airports by departure delay ─────────
print("=" * 60)
print("SQL ANALYSIS 4: Top 10 Worst Airports by Departure Delay")
print("=" * 60)

# HAVING clause filters airports with fewer than 1000 delayed
# departures to exclude low-volume airports
spark.sql("""
    SELECT ORIGIN,
           COUNT(*) AS delayed_departures,
           ROUND(AVG(DEP_DELAY), 2) AS avg_dep_delay_min
    FROM flights
    WHERE DEP_DELAY > 0
    GROUP BY ORIGIN
    HAVING COUNT(*) > 1000
    ORDER BY avg_dep_delay_min DESC
    LIMIT 10
""").show()

# ── SQL Analysis 5: Cancellations by reason ──────────────────
print("=" * 60)
print("SQL ANALYSIS 5: Cancellations by Reason")
print("=" * 60)

# CANCELLED = 1 flags cancelled flights
# Codes: A=Carrier, B=Weather, C=NAS, D=Security
spark.sql("""
    SELECT CANCELLATION_CODE,
           COUNT(*) AS cancelled_flights
    FROM flights
    WHERE CANCELLED = 1
    GROUP BY CANCELLATION_CODE
    ORDER BY cancelled_flights DESC
""").show()

# ── SQL Analysis 6: Monthly delay trend ──────────────────────
print("=" * 60)
print("SQL ANALYSIS 6: Monthly Delay Trend")
print("=" * 60)

# MONTH() extracts month number from FL_DATE timestamp
spark.sql("""
    SELECT MONTH(FL_DATE) AS month,
           ROUND(AVG(ARR_DELAY), 2) AS avg_delay_min
    FROM flights
    WHERE ARR_DELAY > 0
    GROUP BY MONTH(FL_DATE)
    ORDER BY month
""").show(12)

spark.stop()
print("Spark SQL analysis complete.")
