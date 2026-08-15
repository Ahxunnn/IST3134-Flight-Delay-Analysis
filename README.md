# IST3134 — Flight Delay Analysis Using PySpark on Amazon EMR

**Course:** IST3134 Big Data Analytics in the Cloud  
**Semester:** May 2026  
**Group Members:** Lee Zong Xun (22083844) | Bernice Wong Jian Xuan (22056998)  
**GitHub Repository:** https://github.com/Ahxunnn/IST3134-Flight-Delay-Analysis

---

## Project Overview

This project analyses US domestic flight delay and cancellation patterns
using PySpark on Amazon EMR. The dataset covers 12,888,067 flight records
from 2017 to 2018, sourced from the Bureau of Transportation Statistics
(BTS) via Kaggle.

The analysis identifies which airlines have the highest average delays,
which airports are the worst for departures, what causes the most delay
minutes across the US aviation network, and how performance changed
between 2017 and 2018.

---

## Dataset

**Source:** US Airline Delay and Cancellation Data (2009–2018)  
**Link:** https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018  
**Years used:** 2017 and 2018  
**Total rows loaded:** 12,888,067  
**Columns:** 27  
**Size:** ~1.4 GB (approximately 700 MB per year)

---

## Repository Structure
IST3134-Flight-Delay-Analysis/
├── README.md
├── code/
│ ├── analysis_pyspark.py — PySpark DataFrame API (7 analyses)
│ ├── analysis_sql.py — Spark SQL (same analyses, different API)
│ └── analysis_rdd.py — RDD API (low-level, third implementation)
├── output/
│ └── screenshots/ — Terminal output screenshots from EMR runs
└── report/
└── IST3134_Group_Report.pdf — Final submission report

---

## Analyses Performed

| # | Analysis | Spark Operation Used |
|---|---|---|
| 1 | Total flights per airline | groupBy + count |
| 2 | Average arrival delay per airline | filter + groupBy + agg |
| 3 | Total delay minutes by cause | agg + sum (5 delay columns) |
| 4 | Top 10 worst airports by departure delay | filter + groupBy + filter |
| 5 | Cancellations by reason code | filter + groupBy + count |
| 6 | Monthly average delay trend | withColumn + groupBy |
| 7 | Top 3 most delayed routes per airline | Window function + row_number |
| 8 | Year-on-year 2017 vs 2018 comparison | withColumn year + groupBy |

---

## Three Implementation Approaches

| Script | API | Lines of Code | Duration | Optimizer |
|---|---|---|---|---|
| analysis_rdd.py | RDD API | ~60 lines | 2 min 21 sec | None |
| analysis_pyspark.py | DataFrame API | ~30 lines | ~3 minutes | Full Catalyst |
| analysis_sql.py | Spark SQL | ~25 lines | ~2 minutes | Full Catalyst |

All three implementations produce **identical results** across every analysis,
confirming that Spark's three API layers are equivalent in correctness.

---

## How to Run

### Prerequisites
- Amazon EMR cluster (emr-7.x, 1 primary + 2 core m5.xlarge nodes)
- Dataset uploaded to HDFS at `/user/hadoop/flights/`

```bash
# Upload dataset to HDFS
hadoop fs -mkdir -p /user/hadoop/flights
hadoop fs -put 2017.csv /user/hadoop/flights/
hadoop fs -put 2018.csv /user/hadoop/flights/
```

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

### Run RDD API version
```bash
spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-memory 1g \
  code/analysis_rdd.py \
  hdfs:///user/hadoop/flights/
```

## Visualization Scripts

| Script | Charts Generated | Output |
|---|---|---|
| `code/visualizations.py` | 7 charts — flights, delays, causes, trend, cancellations, YoY, airports | PNG files |
| `code/viz_yoy.py` | 3 charts — YoY overall performance, delay causes, top airlines | PNG files |

### Run Visualizations
```bash
spark-submit --master yarn --deploy-mode client \
  --num-executors 2 --executor-memory 1g \
  code/visualizations.py \
  hdfs:///user/hadoop/flights/ \
  /home/hadoop/charts
```

---

## Key Findings

- **Southwest Airlines (WN)** operated the most flights — 2,681,996 across 2017–2018
- **ExpressJet (EV)** had the worst average arrival delay at 51.71 minutes
- **Late aircraft cascading delays** caused the most total delay minutes (39.9%)
- **Weather** was the top cancellation cause (55.4%) but only 5.1% of delay minutes
- **Eagle County Regional (EGE)** had the worst average departure delay at 82.95 minutes
- **July** was the worst month for delays; **October** was the best
- **2018 was worse than 2017** across every metric — cancellations up 41%, weather delays up 75.7%

---

## Infrastructure

| Component | Specification |
|---|---|
| Platform | Amazon EMR emr-7.x |
| Primary Node | 1× m5.xlarge (4 vCPU, 16 GB RAM) |
| Core Nodes | 2× m5.xlarge (4 vCPU, 16 GB RAM each) |
| Framework | Apache Spark 3.5.6, Hadoop 3.x, YARN |
| Storage | HDFS (processing) + Amazon S3 (backup) |
| Language | Python 3.11 |

---

## Technologies Used

- Amazon EMR (emr-7.x)
- Apache Spark 3.5.6 (PySpark — DataFrame API, Spark SQL, RDD API)
- Apache Hadoop / YARN
- Amazon S3 (output storage)
- Python 3.11
- Kaggle API (dataset download)
