from sklearn.ensemble import IsolationForest as isf
from sklearn.preprocessing import LabelEncoder as le, StandardScaler as scaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #FOR NEXT TIME --> PLOT THE DATA

np.random.seed(42) # Change seed to create different datasets
DATA_SIZE = 20000

normal_durations = np.random.normal(loc = 180, scale = 40, size = int(DATA_SIZE * 0.97))
normal_durations = np.clip(normal_durations, 60, 300)
anomaly_durations = np.random.normal(loc = 450, scale = 150, size = int(DATA_SIZE * 0.03))
anomaly_durations = np.clip(anomaly_durations, 250, 800)
durations = np.concatenate([normal_durations, anomaly_durations])
durations = np.rint(durations).astype(int)

# Create true labels before shuffling
true_labels = np.concatenate([np.zeros(int(DATA_SIZE * 0.97)), np.ones(int(DATA_SIZE * 0.03))]).astype(int)

# OMITTED
# 86400 seconds in one day.
# Usual working times are 8:00AM to 5:00PM FOR DOCTORS (second 28800 to 61200)
# 1 Hour = 3600 Seconds. Standard Deviation is 1Hr
# am8 = 28800
# pm5 = 61200
# normal_ts = np.random.normal(loc = (am8 + pm5) // 2, scale = 3600, size = int(DATA_SIZE * 0.97))
# normal_ts = np.clip(normal_ts, am8, pm5)
# anomaly_ts_low = np.random.normal(loc = am8 // 2, scale, )

normal_hour = np.random.choice(range(8, 18), int(DATA_SIZE * 0.97))
anomaly_hour = np.random.choice(list(range(0, 6)) + list(range(20, 24)), int(DATA_SIZE * 0.03))
hour = np.concatenate([normal_hour, anomaly_hour])

normal_dsd = np.random.normal(loc = 10, scale = 5, size = int(DATA_SIZE * 0.97))
normal_dsd = np.clip(normal_dsd, 0, 30)
anomaly_dsd = np.random.normal(loc = 25, scale = 8, size = int(DATA_SIZE * 0.03))
anomaly_dsd = np.clip(anomaly_dsd, 10, 50)
dsd = np.concatenate([normal_dsd, anomaly_dsd])
dsd = np.rint(dsd).astype(int)

users = [f"U{i:04d}" for i in range(500)]
devices = [f"D{i:03d}" for i in range(250)]

df = pd.DataFrame({
    'duration': durations,
    'hour': hour,
    'days_since_discharge': dsd,
    'UserID': np.random.choice(users, size = DATA_SIZE, replace = True),
    'Device': np.random.choice(devices, size = DATA_SIZE, replace = True),
    'true_label': true_labels
})

user_encoder = le()
device_encoder = le()

df['user_enc'] = user_encoder.fit_transform(df['UserID'])
df['device_enc'] = device_encoder.fit_transform(df['Device'])

features = [
    'duration',                 #seconds
    'hour',
    'days_since_discharge',
    'device_enc',
    'user_enc'
]

X = df[features]
X_scaled = scaler().fit_transform(X)

model = isf(
    n_estimators = 300,     # number of tress
    max_samples = 512,
    contamination = 0.03,   # expected anomaly %
    random_state = 42
)

model.fit(X_scaled)

# Extract true labels from dataframe
true_labels = df['true_label'].values

anomaly_scores = model.decision_function(X_scaled)
predictions = (model.predict(X_scaled) == -1).astype(int)

f1 = f1_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
accuracy = accuracy_score(true_labels, predictions)
print(f"F1-SCORE: {f1:.3f}")
print(f"PRECISION: {precision:.3f}")
print(f"RECALL: {recall:.3f}")
print(f"ACCURACY: {accuracy:.3f}")

cm = confusion_matrix(true_labels, predictions)
print(f"\nConfusion Matrix:\n{cm}")
print(f"True Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")

df['anomaly_score'] = anomaly_scores.round(3)
df['anomaly'] = (model.predict(X_scaled) == -1).astype(int)

#### VISUALIZATIONS ####
fig, axes = plt.subplots(2, 2, figsize = (14, 10))

# Plot 1: Duration vs. Hour (colored by anomaly)
axes[0, 0].scatter(df[df['anomaly'] == 0]['hour'], df[df['anomaly'] == 0]['duration'],
                   alpha = 0.5, label = 'Normal', s = 20)
axes[0, 0].scatter(df[df['anomaly'] == 1]['hour'], df[df['anomaly'] == 1]['duration'],
                   alpha = 0.7, label = 'Anomaly', color = 'red', s = 20)
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Duration (seconds)')
axes[0, 0].set_title('Duration vs Hour')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha = 0.3)

# Plot 2: Duration vs. DSD
axes[0, 1].scatter(df[df['anomaly'] == 0]['days_since_discharge'], df[df['anomaly'] == 0]['duration'],
                   alpha = 0.5, label = 'Normal', s = 20)
axes[0, 1].scatter(df[df['anomaly'] == 1]['days_since_discharge'], df[df['anomaly'] == 1]['duration'],
                   alpha = 0.7, label = 'Anomaly', color = 'red', s = 20)
axes[0, 1].set_xlabel('Days Since Discharge')
axes[0, 1].set_ylabel('Duration (seconds)')
axes[0, 1].set_title('Duration vs. Days Since Patient Discharge')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha = 0.3)

# Plot 3: Anomaly Score Distribution
axes[1, 0].hist(df[df['anomaly'] == 0]['anomaly_score'], bins = 50, alpha = 0.6, label = 'Normal')
axes[1, 0].hist(df[df['anomaly'] == 1]['anomaly_score'], bins = 50, alpha = 0.6, label = 'Anomaly', color = 'red')
axes[1, 0].set_xlabel('Anomaly Score')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Anomaly Score Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha = 0.3)

# Anomaly Detection Summary
anomaly_counts = df['anomaly'].value_counts()
axes[1, 1].bar(['Normal', 'Anomaly'], [anomaly_counts.get(0, 0), anomaly_counts.get(1, 0)], color = ['blue', 'red'])
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'Anomaly Detection Summary (F1-Score: {f1:.4f})')
axes[1, 1].grid(True, alpha = 0.3, axis = 'y')

plt.tight_layout()
plt.savefig(Path(__file__).parent / "anomaly_visualization.png", dpi = 300)
print(f"Visualization saved: {Path(__file__).parent / 'anomaly_visualization.png'}")
# plt.show() # Uncomment if you want to see the plot

out_path = Path(__file__).parent / "dataset.csv"
df.to_csv(out_path, index = False)

print(f"Saved: {out_path}")
print(df.head(10))