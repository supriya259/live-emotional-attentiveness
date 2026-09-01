import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================
# LOAD CSV
# ============================================================

CSV_FILE = "attention_data.csv"

df = pd.read_csv(CSV_FILE)

print("\n===== ATTENTIVENESS REPORT =====\n")

# ============================================================
# BASIC INFORMATION
# ============================================================

print("Total records:", len(df))

# ============================================================
# AVERAGE ATTENTION
# ============================================================

average_attention = df["Attention"].mean()

print(
    "Average Attention:",
    round(average_attention, 2),
    "%"
)

# ============================================================
# TOTAL BLINKS
# ============================================================

total_blinks = df["Blinks"].max()

print(
    "Total Blinks:",
    total_blinks
)

# ============================================================
# TOTAL YAWS
# ============================================================

total_yawns = df["Yawns"].max()

print(
    "Total Yawns:",
    total_yawns
)

# ============================================================
# MOST COMMON EMOTION
# ============================================================

emotion_counts = Counter(
    df["Emotion"].dropna()
)

if emotion_counts:

    most_common_emotion = (
        emotion_counts.most_common(1)[0][0]
    )

else:

    most_common_emotion = "Unknown"

print(
    "Most Common Emotion:",
    most_common_emotion
)

# ============================================================
# HEAD DIRECTION
# ============================================================

head_counts = df["Head"].value_counts()

print("\nHead Direction:")

print(head_counts)

# ============================================================
# ATTENTION STATUS
# ============================================================

if average_attention >= 80:

    status = "Attentive"

elif average_attention >= 50:

    status = "Moderate"

else:

    status = "Distracted"

print(
    "\nOverall Status:",
    status
)

# ============================================================
# ATTENTION GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    df["Time"],
    df["Attention"],
    marker="o"
)

plt.title(
    "Attention Level Over Time"
)

plt.xlabel("Time")

plt.ylabel(
    "Attention (%)"
)

plt.ylim(0, 100)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()