#!/usr/bin/env python3
"""
================================================================================
IST3134 Big Data Analytics in the Cloud — Group Assignment May 2026
================================================================================
Title   : US Flight Delay Pattern Analysis — Spark RDD API
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
      analysis_rdd.py hdfs:///user/hadoop/flights/
--------------------------------------------------------------------------------
Note:
    This is the lowest-level Spark API implementation.
    Unlike DataFrame API and Spark SQL, the RDD API does NOT benefit
    from Catalyst optimizer — no column pruning or predicate pushdown.
    All parsing, type conversion, and null handling are done manually.
    Included to demonstrate the full Spark API stack for comparison.
    Results are IDENTICAL to analysis_pyspark.py and analysis_sql.py.
================================================================================
"""
import sys
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FlightDelayAnalysis-RDD").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext

# RDD needs explicit wildcard to read multiple CSV files from directory
# Unlike DataFrame API which handles directories automatically
INPUT_PATH = sys.argv[1].rstrip("/") + "/*.csv"

print("=" * 60)
print("RDD ANALYSIS: Loading dataset")
print(f"Reading from: {INPUT_PATH}")
print("=" * 60)

# Read all lines as raw text strings
raw = sc.textFile(INPUT_PATH)

# Extract header from first line — needed to find column indices
header = raw.first()

# Filter out header rows (both files have headers when using wildcard)
data = raw.filter(lambda line: line != header and
                  not line.startswith("FL_DATE"))

# Find column positions from header — avoids hardcoding index numbers
cols = header.split(",")
carrier_idx = cols.index("OP_CARRIER")
arr_delay_idx = cols.index("ARR_DELAY")
cancelled_idx = cols.index("CANCELLED")
cancel_code_idx = cols.index("CANCELLATION_CODE")

# Parse each CSV line into a list of field values
def parse_line(line):
    return line.split(",")

parsed = data.map(parse_line)

# ── RDD Analysis 1: Total flights per airline ─────────────────
print("=" * 60)
print("RDD ANALYSIS 1: Total Flights per Airline")
print("=" * 60)

# map: emit (carrier, 1) for each flight
# reduceByKey: sum all 1s per carrier = total flights
# sortBy: descending order by count
flights_per_carrier = (parsed
    .map(lambda f: (f[carrier_idx].strip(), 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: -x[1]))

print("Carrier | Total Flights")
print("-" * 30)
for carrier, count in flights_per_carrier.take(10):
    print(f"  {carrier:>10} | {count:>12,}")

# ── RDD Analysis 2: Average arrival delay per airline ─────────
print("=" * 60)
print("RDD ANALYSIS 2: Average Arrival Delay per Airline")
print("=" * 60)

def extract_delay(fields):
    """Extract (carrier, (delay_minutes, 1)) for delayed flights only."""
    try:
        carrier = fields[carrier_idx].strip()
        delay = float(fields[arr_delay_idx])
        # Only include flights with positive delay
        if delay > 0:
            return (carrier, (delay, 1))
    except (ValueError, IndexError):
        # Skip malformed rows gracefully
        pass
    return None

# map: extract delay tuples, filter None (malformed/on-time rows)
# reduceByKey: sum delays and counts per carrier
# mapValues: compute average from (total_delay, count) tuple
delay_rdd = (parsed
    .map(extract_delay)
    .filter(lambda x: x is not None)
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
    .mapValues(lambda x: round(x[0]/x[1], 2))
    .sortBy(lambda x: -x[1]))

print("Carrier | Avg Delay (min)")
print("-" * 35)
for carrier, avg in delay_rdd.take(10):
    print(f"  {carrier:>10} | {avg:>15.2f}")

# ── RDD Analysis 3: Cancellations by reason ──────────────────
print("=" * 60)
print("RDD ANALYSIS 3: Cancellations by Reason")
print("=" * 60)

def extract_cancellation(fields):
    """Extract (cancellation_code, 1) for cancelled flights only."""
    try:
        # CANCELLED column = 1.0 means flight was cancelled
        if float(fields[cancelled_idx]) == 1:
            code = fields[cancel_code_idx].strip()
            if code:
                return (code, 1)
    except (ValueError, IndexError):
        pass
    return None

# Decode cancellation codes for human-readable output
code_map = {
    "A": "Carrier",
    "B": "Weather",
    "C": "National Air System",
    "D": "Security"
}

cancel_rdd = (parsed
    .map(extract_cancellation)
    .filter(lambda x: x is not None)
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: -x[1]))

print("Code | Reason               | Count")
print("-" * 45)
for code, count in cancel_rdd.collect():
    reason = code_map.get(code, "Unknown")
    print(f"  {code:>4} | {reason:<20} | {count:>8,}")

spark.stop()
print("\nRDD analysis complete.")
print("Results match analysis_pyspark.py and analysis_sql.py exactly.")
