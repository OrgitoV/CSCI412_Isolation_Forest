from sklearn.ensemble import IsolationForest as isf
from sklearn.preprocessing import LabelEncoder as le, StandardScaler as scaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

np.random.seed()
DATA_SIZE = 20000

rs = RandomState(MT19937(SeedSequence(1234567890)))

normal_durations = np.random.normal(loc = 180, scale = 40, size = int(DATA_SIZE * 0.97))
normal_durations = np.clip(normal_durations, 60, 300)
anomaly_durations = np.random.normal(loc = 3650, scale = 500, size = int(DATA_SIZE * 0.03))
anomaly_durations = np.clip(anomaly_durations, 301, 7000)
durations = np.concatenate([normal_durations, anomaly_durations])

# OMITTED
# 86400 seconds in one day.
# Usual working times are 8:00AM to 5:00PM FOR DOCTORS (second 28800 to 61200)
# 1 Hour = 3600 Seconds. Standard Deviation is 1Hr
# am8 = 28800
# pm5 = 61200
# normal_ts = np.random.normal(loc = (am8 + pm5) // 2, scale = 3600, size = int(DATA_SIZE * 0.97))
# normal_ts = np.clip(normal_ts, am8, pm5)
# anomaly_ts_low = np.random.normal(loc = am8 // 2, scale, )

normal_ts = np.random.choice(range(8, 18), int(DATA_SIZE * 0.97))
anomaly_ts = np.random.choice(list(range(0, 6) + range(20, 24)), int(DATA_SIZE * 0.03))
ts = np.concatenate([normal_ts, anomaly_ts])

normal_dsd = np.random.normal(loc = 10, scale = 5, size = int(DATA_SIZE * 0.97))
normal_dsd = np.clip(normal_dsd, 0, 30)
anomaly_dsd = np.random.normal(loc = 120, scale = 30, size = int(DATA_SIZE * 0.03))
anomaly_dsd = np.clip(anomaly_dsd, min = 35)

df['device_enc'] = le.fit_transform(df['Device'])
df['user_enc'] = le.fit_transform(df['UserID'])
df['ts'] = le.fit_transform(df['Time Stamp'])


features = [
    'duration',                 #seconds
    'ts',
    'days_since_discharge',
    'device_enc',
    'user_enc'
]

X = df[features]
X_scaled = scaler.fit_transform(X)

model = isf(
    n_estimators = 100,     # number of tress
    max_samples = 'auto',
    contamination = 0.05,   # expected anomaly %
    random_state = 42
)

model.fir(X_scaled)

df['anomaly_score'] = model.decision_function(X_scaled)
df['anomaly'] = model.predict(X_scaled)