from sklearn.ensemble import IsolationForest as isf
from sklearn.preprocessing import LabelEncoder as le, StandardScaler as scaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from pathlib import Path
from classifier import run_classification, run_baseline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #FOR NEXT TIME --> PLOT THE DATA

np.random.seed(42) # Change seed to create different datasets
DATA_SIZE = 20000

normal_durations = np.random.normal(loc = 180, scale = 120, size = int(DATA_SIZE * 0.97))
normal_durations = np.clip(normal_durations, 60, 300)
anomaly_durations = np.random.normal(loc = 300, scale = 100, size = int(DATA_SIZE * 0.03))
anomaly_durations = np.clip(anomaly_durations, 250, 800)
durations = np.concatenate([normal_durations, anomaly_durations])
durations = np.rint(durations).astype(int)

# Create true labels before shuffling

normal_hour = np.random.choice(range(6, 20), int(DATA_SIZE * 0.97))
anomaly_hour = np.random.choice(list(range(0, 8)) + list(range(18, 24)), int(DATA_SIZE * 0.03))
hour = np.concatenate([normal_hour, anomaly_hour])

normal_dsd = np.random.normal(loc = 12, scale = 12, size = int(DATA_SIZE * 0.97))
normal_dsd = np.clip(normal_dsd, 0, 40)
anomaly_dsd = np.random.normal(loc = 20, scale = 10, size = int(DATA_SIZE * 0.03))
anomaly_dsd = np.clip(anomaly_dsd, 5, 60)
dsd = np.concatenate([normal_dsd, anomaly_dsd])
dsd = np.rint(dsd).astype(int)

true_labels = np.concatenate([np.zeros(int(DATA_SIZE * 0.97)), np.ones(int(DATA_SIZE * 0.03))]).astype(int)

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

# df['duration_per_hour'] = df['duration'] / (df['hour'] + 1)
# df['dsd_hour_interaction'] = df['days_since_discharge'] * df['hour']

# user_stats = df.groupby('user_enc').agg({
#     'duration': ['mean', 'std'],
#     'true_label': 'mean'
# }).reset_index()

# device_stats = df.groupby('device_enc').agg({
#     'duration': ['mean', 'std'],
#     'true_label': 'mean'
# }).reset_index()
# # Merge back

# user_stats.columns = ['user_enc', 'user_duration_mean', 'user_duration_std', 'user_anomaly_rate']
# device_stats.columns = ['device_enc', 'device_duration_mean', 'device_duration_std', 'device_anomaly_rate']

# df = df.merge(user_stats, on = 'user_enc', how = 'left')
# df = df.merge(device_stats, on = 'device_enc', how = 'left')

# df['user_duration_std'] = df['user_duration_std'].fillna(0)
# df['device_duration_std'] = df['device_duration_std'].fillna(0)


# Shuffle the data to avoid bias from sorted normal/anomaly entries
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

features = [
    'duration',                 #seconds
    'hour',
    'days_since_discharge',
    'device_enc',
    'user_enc',
    # 'duration_per_hour',
    # 'dsd_hour_interaction',
    # 'user_duration_mean',
    # 'user_duration_std',
    # 'user_anomaly_rate',
    # 'device_duration_mean',
    # 'device_duration_std',
    # 'device_anomaly_rate'
]

X = df[features]
X_scaled = scaler().fit_transform(X)

print('Testing contamination parameters...')
for contamination in [0.01, 0.03, 0.05, 0.07, 0.10]:
    print(f"   Testing contamination = {contamination}...", end = " ", flush = True)
    model = isf(
        n_estimators = 300,     # number of tress
        max_samples = 512,
        contamination = contamination,   # expected anomaly %
        random_state = 42
    )

    model.fit(X_scaled)
    predictions = (model.predict(X_scaled) == -1).astype(int)
    f1 = f1_score(true_labels, predictions)
    print(f"F1 = {f1:.3f}")

# Extract true labels from dataframe
true_labels = df['true_label'].values

anomaly_scores = model.decision_function(X_scaled)

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


print("Running baseline model...")
run_baseline(X_scaled, true_labels, model_type = 'svm')

print("\n\nRunning model with Isolation Forest feature...")
run_classification(X_scaled, true_labels, model_type = 'svm')







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

out_path = Path(__file__).parent / "ISF_Output.csv"
df.to_csv(out_path, index = False)

print(f"Saved: {out_path}")
print(df.head(10))