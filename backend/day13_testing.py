import pandas as pd

CSV_FILE = "attention_data.csv"

df = pd.read_csv(CSV_FILE)

print("\n===================================")
print("       DAY 13 TEST REPORT")
print("===================================\n")

# Number of records
print("Total records:", len(df))

# Average attention
average_attention = df["Attention"].mean()

print(
    "Average Attention:",
    round(average_attention, 2),
    "%"
)

# Minimum and maximum attention
print(
    "Minimum Attention:",
    df["Attention"].min(),
    "%"
)

print(
    "Maximum Attention:",
    df["Attention"].max(),
    "%"
)

# Emotion distribution
print("\nEmotion Distribution:")

print(
    df["Emotion"].value_counts()
)

# Head direction
print("\nHead Direction:")

print(
    df["Head"].value_counts()
)

# Blink count
print(
    "\nFinal Blink Count:",
    df["Blinks"].max()
)

# Yawn count
print(
    "Final Yawn Count:",
    df["Yawns"].max()
)

# Overall status
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

print("\n===================================")
print("          TEST COMPLETE")
print("===================================")