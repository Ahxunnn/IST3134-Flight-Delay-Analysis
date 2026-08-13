# IST3134 — Flight Delay Analysis Using PySpark on Amazon EMR

**Course:** IST3134 Big Data Analytics in the Cloud  
**Semester:** May 2026  
**Group Members:** [Your Name] | [Partner Name]  

---

## Project Overview

This project analyses US domestic flight delay and cancellation patterns
using PySpark on Amazon EMR. The dataset covers approximately 12 million
flight records from 2017 to 2018, sourced from the Bureau of
Transportation Statistics (BTS) via Kaggle.

The analysis identifies which airlines have the highest average delays,
which airports are the worst for departures, and what causes the most
delay minutes across the US aviation network.

---

## Dataset

**Source:** US Airline Delay and Cancellation Data (2009–2018)  
**Link:** https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018  
**Years used:** 2017 and 2018  
**Rows:** ~12–14 million  
**Columns:** 27  

---

## Repository Structure

code/
analysis_pyspark.py — PySpark DataFrame API implementation
analysis_sql.py — Spark SQL implementation (same analyses)
output/
screenshots/ — Terminal output screenshots from EMR runs
report/
IST3134_Group_Report.pdf — Final submission report

---

## How to Run

### Prerequisites
- Amazon EMR cluster (emr-7.x, 1 primary + 2 core m5.xlarge nodes)
- Dataset uploaded to HDFS at `/user/hadoop/flights/`

### Run PySpark DataFrame version
```bash
spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-memory 1g \
  code/analysis_pyspark.py \
  hdfs:///user/hadoop/flights/
```

### Run Spark SQL version
```bash
spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-memory 1g \
  code/analysis_sql.py \
  hdfs:///user/hadoop/flights/
```

---

## Technologies Used

- Amazon EMR (emr-7.x)
- Apache Spark 3.5 (PySpark)
- Apache Hadoop / YARN
- Amazon S3 (output storage)
- Python 3.11
