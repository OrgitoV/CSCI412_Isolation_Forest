from sklearn.ensemble import IsolationForest as isf
from sklearn.preprocessing import LabelEncoder as le, StandardScaler as scaler
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
DATA_SIZE = 20000

normal_durations = np.random.normal(loc = 180, scale = 40, size = int(DATA_SIZE * 0.97))
normal_durations = np.clip(normal_durations, 60, 300)
anomaly_durations = np.random.normal(loc = 3650, scale = 500, size = int(DATA_SIZE * 0.03))
anomaly_durations = np.clip(anomaly_durations, 301, 7000)
durations = np.concatenate([normal_durations, anomaly_durations])
durations = np.rint(durations).astype(int)

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
anomaly_dsd = np.random.normal(loc = 120, scale = 30, size = int(DATA_SIZE * 0.03))
anomaly_dsd = np.clip(anomaly_dsd, a_min = 35, a_max = None)
dsd = np.concatenate([normal_dsd, anomaly_dsd])
dsd = np.rint(dsd).astype(int)

users = [f"U{i:04d}" for i in range(500)]
devices = [f"D{i:03d}" for i in range(250)]

df = pd.DataFrame({
    'duration': durations,
    'hour': hour,
    'days_since_discharge': dsd,
    'UserID': np.random.choice(users, size = DATA_SIZE, replace = True),
    'Device': np.random.choice(devices, size = DATA_SIZE, replace = True)
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
    n_estimators = 100,     # number of tress
    max_samples = 'auto',
    contamination = 0.05,   # expected anomaly %
    random_state = 42
)

model.fit(X_scaled)

df['anomaly_score'] = model.decision_function(X_scaled)
df['anomaly'] = (model.predict(X_scaled) == -1).astype(int)

out_path = Path(__file__).parent / "dataset.csv"
df.to_csv(out_path, index = False)

print(f"Saved: {out_path}")
print(df.head())