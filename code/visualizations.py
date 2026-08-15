#!/usr/bin/env python3
"""
================================================================================
IST3134 Big Data Analytics in the Cloud — Group Assignment May 2026
================================================================================
Title   : US Flight Delay Pattern Analysis — Visualizations
Authors : Lee Zong Xun | Bernice Wong Jian Xuan
Dataset : BTS Airline Delay and Cancellation Data (2017-2018)
Source  : https://www.kaggle.com/datasets/yuanyuwendymu/
          airline-delay-and-cancellation-data-2009-2018
--------------------------------------------------------------------------------
How to run:
    spark-submit --master yarn --deploy-mode client \
      --num-executors 2 --executor-memory 1g \
      visualizations.py \
      hdfs:///user/hadoop/flights/ \
      /home/hadoop/charts
--------------------------------------------------------------------------------
Charts generated:
    1. chart1_flights_per_airline.png     — Bar chart: total flights per airline
    2. chart2_avg_delay_per_airline.png   — Horizontal bar: avg delay per airline
    3. chart3_delay_causes_pie.png        — Pie chart: delay minutes by cause
    4. chart4_monthly_trend.png           — Line chart: monthly delay trend
    5. chart5_cancellation_reasons.png    — Bar chart: cancellations by reason
    6. chart6_year_on_year.png            — Grouped bar: 2017 vs 2018 overview
    7. chart7_worst_airports.png          — Horizontal bar: worst airports
================================================================================
"""
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environment
import matplotlib.pyplot as plt
import numpy as np

spark = SparkSession.builder.appName("FlightDelayVisualizations").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

INPUT_PATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/home/hadoop/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .csv(INPUT_PATH))
print(f"Loaded {df.count():,} rows")

# ── Chart 1: Total Flights per Airline (Bar Chart) ───────────
print("Generating Chart 1: Total Flights per Airline...")
data1 = (df.groupBy("OP_CARRIER")
    .agg(F.count("*").alias("total_flights"))
    .orderBy(F.desc("total_flights"))
    .limit(10)
    .toPandas())

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#1F3864' if i == 0 else '#2F5496' if i < 3 else '#4472C4'
          for i in range(len(data1))]
bars = ax.bar(data1['OP_CARRIER'], data1['total_flights'] / 1e6, color=colors)
ax.set_title('Total Flights per Airline (2017-2018)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Airline Carrier Code', fontsize=12)
ax.set_ylabel('Total Flights (Millions)', fontsize=12)
ax.set_ylim(0, data1['total_flights'].max() / 1e6 * 1.2)
for bar, val in zip(bars, data1['total_flights']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val/1e6:.2f}M', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart1_flights_per_airline.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart1_flights_per_airline.png")

# ── Chart 2: Average Delay per Airline (Horizontal Bar) ──────
print("Generating Chart 2: Average Delay per Airline...")
data2 = (df.filter(F.col("ARR_DELAY") > 0)
    .groupBy("OP_CARRIER")
    .agg(F.round(F.avg("ARR_DELAY"), 2).alias("avg_delay_min"))
    .orderBy(F.desc("avg_delay_min"))
    .limit(10)
    .toPandas())

fig, ax = plt.subplots(figsize=(12, 6))
colors2 = ['#C00000' if i == 0 else '#FF0000' if i < 3 else '#FF6B6B'
           for i in range(len(data2))]
bars2 = ax.barh(data2['OP_CARRIER'][::-1],
                data2['avg_delay_min'][::-1],
                color=colors2[::-1])
ax.set_title('Average Arrival Delay per Airline — Delayed Flights Only (2017-2018)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Average Delay (Minutes)', fontsize=12)
ax.set_ylabel('Airline Carrier Code', fontsize=12)
for bar, val in zip(bars2, data2['avg_delay_min'][::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val} min', va='center', fontsize=10, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart2_avg_delay_per_airline.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart2_avg_delay_per_airline.png")

# ── Chart 3: Delay Causes Pie Chart ──────────────────────────
print("Generating Chart 3: Delay Causes Breakdown...")
data3 = df.agg(
    F.sum("CARRIER_DELAY").alias("Carrier"),
    F.sum("WEATHER_DELAY").alias("Weather"),
    F.sum("NAS_DELAY").alias("NAS"),
    F.sum("SECURITY_DELAY").alias("Security"),
    F.sum("LATE_AIRCRAFT_DELAY").alias("Late Aircraft")
).toPandas()

causes = ['Late Aircraft', 'Carrier', 'NAS', 'Weather', 'Security']
values = [
    float(data3['Late Aircraft'].iloc[0]),
    float(data3['Carrier'].iloc[0]),
    float(data3['NAS'].iloc[0]),
    float(data3['Weather'].iloc[0]),
    float(data3['Security'].iloc[0])
]
colors3 = ['#C00000', '#FF6B35', '#FFD700', '#4472C4', '#70AD47']
explode = (0.05, 0, 0, 0, 0)

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    values, labels=causes, colors=colors3,
    explode=explode, autopct='%1.1f%%',
    startangle=140, pctdistance=0.85,
    textprops={'fontsize': 11}
)
for autotext in autotexts:
    autotext.set_fontweight('bold')
ax.set_title('Total Delay Minutes by Cause (2017-2018)',
             fontsize=16, fontweight='bold', pad=20)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart3_delay_causes_pie.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart3_delay_causes_pie.png")

# ── Chart 4: Monthly Delay Trend (Line Chart) ─────────────────
print("Generating Chart 4: Monthly Delay Trend...")
data4 = (df.filter(F.col("ARR_DELAY") > 0)
    .withColumn("month", F.month("FL_DATE"))
    .groupBy("month")
    .agg(F.round(F.avg("ARR_DELAY"), 2).alias("avg_delay_min"))
    .orderBy("month")
    .toPandas())

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data4['month'], data4['avg_delay_min'],
        color='#1F3864', linewidth=2.5, marker='o',
        markersize=8, markerfacecolor='#C00000',
        markeredgecolor='white', markeredgewidth=2)
ax.fill_between(data4['month'], data4['avg_delay_min'],
                alpha=0.15, color='#1F3864')
ax.set_title('Monthly Average Arrival Delay Trend (2017-2018)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Average Delay (Minutes)', fontsize=12)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_ylim(data4['avg_delay_min'].min() - 2,
            data4['avg_delay_min'].max() + 3)
for x, y in zip(data4['month'], data4['avg_delay_min']):
    ax.annotate(f'{y}', (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart4_monthly_trend.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart4_monthly_trend.png")

# ── Chart 5: Cancellation Reasons Bar Chart ───────────────────
print("Generating Chart 5: Cancellations by Reason...")
data5 = (df.filter(F.col("CANCELLED") == 1)
    .groupBy("CANCELLATION_CODE")
    .agg(F.count("*").alias("cancelled_flights"))
    .orderBy(F.desc("cancelled_flights"))
    .toPandas())

code_labels = {
    'A': 'Carrier\n(A)',
    'B': 'Weather\n(B)',
    'C': 'National Air\nSystem (C)',
    'D': 'Security\n(D)'
}
data5['label'] = data5['CANCELLATION_CODE'].map(code_labels)
colors5 = ['#4472C4', '#C00000', '#FF6B35', '#70AD47']

fig, ax = plt.subplots(figsize=(10, 6))
bars5 = ax.bar(data5['label'], data5['cancelled_flights'],
               color=colors5[:len(data5)], width=0.5)
ax.set_title('Flight Cancellations by Reason (2017-2018)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Cancellation Reason', fontsize=12)
ax.set_ylabel('Number of Cancelled Flights', fontsize=12)
for bar, val in zip(bars5, data5['cancelled_flights']):
    pct = val / data5['cancelled_flights'].sum() * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart5_cancellation_reasons.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart5_cancellation_reasons.png")

# ── Chart 6: Year-on-Year Comparison (Grouped Bar) ────────────
print("Generating Chart 6: Year-on-Year Comparison...")
data6 = (df.withColumn("year", F.year("FL_DATE"))
    .groupBy("year")
    .agg(
        F.count("*").alias("total_flights"),
        F.count(F.when(F.col("ARR_DELAY") > 0, 1)).alias("delayed_flights"),
        F.count(F.when(F.col("CANCELLED") == 1, 1)).alias("cancellations")
    )
    .orderBy("year")
    .toPandas())

x = np.arange(len(data6['year']))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
bars_total = ax.bar(x - width, data6['total_flights'] / 1e6,
                    width, label='Total Flights (M)', color='#1F3864')
bars_delayed = ax.bar(x, data6['delayed_flights'] / 1e6,
                      width, label='Delayed Flights (M)', color='#C00000')
bars_cancel = ax.bar(x + width, data6['cancellations'] / 1e3,
                     width, label='Cancellations (K)', color='#FF6B35')

ax.set_title('Year-on-Year Performance Comparison: 2017 vs 2018',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(data6['year'].astype(str), fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

for bar in bars_total:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{bar.get_height():.2f}M', ha='center',
            fontsize=8, fontweight='bold')
for bar in bars_delayed:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{bar.get_height():.2f}M', ha='center',
            fontsize=8, fontweight='bold')
for bar in bars_cancel:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}K', ha='center',
            fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart6_year_on_year.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart6_year_on_year.png")

# ── Chart 7: Top 10 Worst Airports (Horizontal Bar) ──────────
print("Generating Chart 7: Worst Airports...")
data7 = (df.filter(F.col("DEP_DELAY") > 0)
    .groupBy("ORIGIN")
    .agg(
        F.count("*").alias("delayed_departures"),
        F.round(F.avg("DEP_DELAY"), 2).alias("avg_dep_delay_min")
    )
    .filter(F.col("delayed_departures") > 1000)
    .orderBy(F.desc("avg_dep_delay_min"))
    .limit(10)
    .toPandas())

fig, ax = plt.subplots(figsize=(12, 6))
colors7 = ['#C00000' if i == 0 else '#FF6B35' if i < 3 else '#FFD700'
           for i in range(len(data7))]
bars7 = ax.barh(data7['ORIGIN'][::-1],
                data7['avg_dep_delay_min'][::-1],
                color=colors7[::-1])
ax.set_title('Top 10 Worst Origin Airports by Average Departure Delay (2017-2018)',
             fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Average Departure Delay (Minutes)', fontsize=12)
ax.set_ylabel('Airport IATA Code', fontsize=12)
for bar, val in zip(bars7, data7['avg_dep_delay_min'][::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val} min', va='center', fontsize=10, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/chart7_worst_airports.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: chart7_worst_airports.png")

print("\n" + "=" * 60)
print(f"All 7 charts saved to: {OUTPUT_DIR}")
print("=" * 60)
spark.stop()
