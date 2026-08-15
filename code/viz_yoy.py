#!/usr/bin/env python3
"""
================================================================================
IST3134 Big Data Analytics in the Cloud — Group Assignment May 2026
================================================================================
Title   : US Flight Delay Pattern Analysis — Year-on-Year Visualizations
Authors : Lee Zong Xun | Bernice Wong Jian Xuan
Dataset : BTS Airline Delay and Cancellation Data (2017-2018)
Source  : https://www.kaggle.com/datasets/yuanyuwendymu/
          airline-delay-and-cancellation-data-2009-2018
--------------------------------------------------------------------------------
How to run:
    spark-submit --master yarn --deploy-mode client \
      --num-executors 2 --executor-memory 1g \
      viz_yoy.py \
      hdfs:///user/hadoop/flights/
--------------------------------------------------------------------------------
Charts generated:
    1. yoy_chart1_overall_performance.png — 2017 vs 2018 overall metrics
    2. yoy_chart2_delay_causes.png        — Delay causes comparison by year
    3. yoy_chart3_top_airlines.png        — Top 5 worst airlines per year
================================================================================
"""
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder.appName("YoYVisualizations").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

INPUT_PATH = sys.argv[1]
OUTPUT_DIR = "/home/hadoop/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(INPUT_PATH))
print(f"Loaded {df.count():,} rows")

# ── YoY Chart 1: Overall Performance ─────────────────────────
print("Generating YoY Chart 1: Overall Performance 2017 vs 2018...")
data1 = (df.withColumn("year", F.year("FL_DATE"))
    .groupBy("year")
    .agg(
        F.count("*").alias("total_flights"),
        F.count(F.when(F.col("ARR_DELAY") > 0, 1)).alias("delayed_flights"),
        F.count(F.when(F.col("CANCELLED") == 1, 1)).alias("cancellations"),
        F.round(F.avg(F.when(F.col("ARR_DELAY") > 0,
                             F.col("ARR_DELAY"))), 2).alias("avg_delay_min"),
        F.round(F.sum(F.when(F.col("CANCELLED") == 1, 1)) /
                F.count("*") * 100, 2).alias("cancel_rate_pct")
    )
    .orderBy("year")
    .toPandas())

years = data1['year'].astype(str).tolist()
x = np.arange(len(years))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Year-on-Year Overall Performance: 2017 vs 2018',
             fontsize=16, fontweight='bold', y=1.02)

ax1 = axes[0]
b1 = ax1.bar(x - width/2, data1['total_flights'] / 1e6,
             width, label='Total Flights (M)', color='#1F3864')
b2 = ax1.bar(x + width/2, data1['delayed_flights'] / 1e6,
             width, label='Delayed Flights (M)', color='#C00000')
ax1.set_title('Total vs Delayed Flights', fontsize=13, fontweight='bold')
ax1.set_ylabel('Flights (Millions)', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(years, fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)
ax1.set_facecolor('#F8F9FA')
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{bar.get_height():.2f}M', ha='center',
             fontsize=9, fontweight='bold')
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{bar.get_height():.2f}M', ha='center',
             fontsize=9, fontweight='bold')

ax2 = axes[1]
ax2_twin = ax2.twinx()
b3 = ax2.bar(x - width/2, data1['cancellations'] / 1e3,
             width, label='Cancellations (K)', color='#FF6B35')
b4 = ax2_twin.bar(x + width/2, data1['avg_delay_min'],
                  width, label='Avg Delay (min)', color='#4472C4')
ax2.set_title('Cancellations and Average Delay', fontsize=13, fontweight='bold')
ax2.set_ylabel('Cancellations (Thousands)', fontsize=11)
ax2_twin.set_ylabel('Average Delay (Minutes)', fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(years, fontsize=12)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_facecolor('#F8F9FA')
for bar in b3:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}K', ha='center',
             fontsize=9, fontweight='bold')
for bar in b4:
    ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                  f'{bar.get_height():.2f}', ha='center', fontsize=9,
                  fontweight='bold', color='#1F3864')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/yoy_chart1_overall_performance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: yoy_chart1_overall_performance.png")

# ── YoY Chart 2: Delay Causes Grouped Bar ────────────────────
print("Generating YoY Chart 2: Delay Causes by Year...")
data2 = (df.withColumn("year", F.year("FL_DATE"))
    .groupBy("year")
    .agg(
        F.round(F.sum("CARRIER_DELAY") / 1e6, 2).alias("carrier_min"),
        F.round(F.sum("WEATHER_DELAY") / 1e6, 2).alias("weather_min"),
        F.round(F.sum("NAS_DELAY") / 1e6, 2).alias("nas_min"),
        F.round(F.sum("LATE_AIRCRAFT_DELAY") / 1e6, 2).alias("late_aircraft_min")
    )
    .orderBy("year")
    .toPandas())

# Use iloc to safely extract row values
row_2017 = data2.iloc[0]
row_2018 = data2.iloc[1]

causes = ['Late Aircraft', 'Carrier', 'NAS', 'Weather']
values_2017 = [
    float(row_2017['late_aircraft_min']),
    float(row_2017['carrier_min']),
    float(row_2017['nas_min']),
    float(row_2017['weather_min'])
]
values_2018 = [
    float(row_2018['late_aircraft_min']),
    float(row_2018['carrier_min']),
    float(row_2018['nas_min']),
    float(row_2018['weather_min'])
]

x2 = np.arange(len(causes))
width2 = 0.3
colors2 = ['#C00000', '#FF6B35', '#FFD700', '#4472C4']

fig, ax = plt.subplots(figsize=(13, 7))
bars_2017 = ax.bar(x2 - width2/2, values_2017, width2,
                   label='2017', color='#AAAAAA',
                   edgecolor=colors2, linewidth=2)
bars_2018 = ax.bar(x2 + width2/2, values_2018, width2,
                   label='2018', color=colors2)

ax.set_title('Delay Minutes by Cause: 2017 vs 2018',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Delay Cause', fontsize=12)
ax.set_ylabel('Total Delay Minutes (Millions)', fontsize=12)
ax.set_xticks(x2)
ax.set_xticklabels(causes, fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for bar in bars_2017:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}M', ha='center',
            fontsize=9, fontweight='bold', color='#555555')
for bar in bars_2018:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}M', ha='center',
            fontsize=9, fontweight='bold')

for i, (v17, v18) in enumerate(zip(values_2017, values_2018)):
    pct_change = (v18 - v17) / v17 * 100
    color = '#C00000' if pct_change > 0 else '#70AD47'
    label = f'+{pct_change:.1f}%' if pct_change > 0 else f'{pct_change:.1f}%'
    ax.annotate(label,
                xy=(x2[i] + width2/2, v18),
                xytext=(x2[i] + width2/2, v18 + 1.5),
                ha='center', fontsize=10, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/yoy_chart2_delay_causes.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: yoy_chart2_delay_causes.png")

# ── YoY Chart 3: Top 5 Airlines per Year ─────────────────────
print("Generating YoY Chart 3: Top 5 Airlines by Avg Delay per Year...")
w = Window.partitionBy("year").orderBy(F.desc("avg_delay_min"))
data3 = (df.filter(F.col("ARR_DELAY") > 0)
    .withColumn("year", F.year("FL_DATE"))
    .groupBy("year", "OP_CARRIER")
    .agg(F.round(F.avg("ARR_DELAY"), 2).alias("avg_delay_min"),
         F.count("*").alias("delayed_flights"))
    .withColumn("rank", F.row_number().over(w))
    .filter(F.col("rank") <= 5)
    .orderBy("year", "rank")
    .toPandas())

data_2017 = data3[data3['year'] == 2017].reset_index(drop=True)
data_2018 = data3[data3['year'] == 2018].reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Top 5 Worst Airlines by Average Arrival Delay: 2017 vs 2018',
             fontsize=15, fontweight='bold', y=1.02)

colors_rank = ['#C00000', '#FF4444', '#FF6B6B', '#FF9999', '#FFBBBB']

ax1 = axes[0]
bars1 = ax1.barh(data_2017['OP_CARRIER'][::-1],
                 data_2017['avg_delay_min'][::-1],
                 color=colors_rank[::-1])
ax1.set_title('2017', fontsize=14, fontweight='bold')
ax1.set_xlabel('Average Delay (Minutes)', fontsize=11)
ax1.set_ylabel('Airline Carrier Code', fontsize=11)
ax1.set_xlim(0, 65)
ax1.grid(axis='x', alpha=0.3)
ax1.set_facecolor('#F8F9FA')
for bar, val in zip(bars1, data_2017['avg_delay_min'][::-1]):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val} min', va='center', fontsize=10, fontweight='bold')

ax2 = axes[1]
bars2 = ax2.barh(data_2018['OP_CARRIER'][::-1],
                 data_2018['avg_delay_min'][::-1],
                 color=colors_rank[::-1])
ax2.set_title('2018', fontsize=14, fontweight='bold')
ax2.set_xlabel('Average Delay (Minutes)', fontsize=11)
ax2.set_ylabel('Airline Carrier Code', fontsize=11)
ax2.set_xlim(0, 65)
ax2.grid(axis='x', alpha=0.3)
ax2.set_facecolor('#F8F9FA')
for bar, val in zip(bars2, data_2018['avg_delay_min'][::-1]):
    ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val} min', va='center', fontsize=10, fontweight='bold')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/yoy_chart3_top_airlines.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: yoy_chart3_top_airlines.png")

print("\n" + "=" * 60)
print(f"All 3 YoY charts saved to: {OUTPUT_DIR}")
print("=" * 60)
spark.stop()
