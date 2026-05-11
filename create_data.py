import faker
import numpy as np
import pandas as pd

def create_isf_data(data_size = 1000, anomaly_split: float = 0.1):
    # Keep a fixed seed for reproducibility
    np.random.seed(42)
    faker.Faker.seed(42)
    
    fake = faker.Faker()
    
    # Data: Normal-to-Anomaly size split
    normal_size = int(data_size * (1 - anomaly_split))
    anomaly_size = int(data_size * anomaly_split)

    ####################################################################################################################

    """=== DURATION ENTRIES ==="""
    # Normal behavior: right-skewed (realistic access patterns)
    normal_duration = np.random.lognormal(mean = 5.5, sigma = 0.8, size = normal_size)

    anomaly_durations_short = np.random.uniform(0.05, 1, size = int(anomaly_size * 0.35))  # Very short accesses
    anomaly_durations_long = np.random.exponential(scale = 2500, size = int(anomaly_size * 0.35))  # Very long accesses
    anomaly_durations_overlap = np.random.lognormal(mean = 5.5, sigma = 0.9, size = int(anomaly_size * 0.3))  # Only 30% blend in with normal

    anomaly_durations = np.concatenate([anomaly_durations_short, anomaly_durations_long, anomaly_durations_overlap])
    """=== DURATION ENTRIES ==="""

    ####################################################################################################################

    """=== HOURS ACCESSED ENTRIES ==="""
    # Normal: business hours (8 AM - 6 PM, peaked at 9-5)
    normal_hours = np.random.normal(loc = 13, scale = 5, size = normal_size)
    normal_hours = np.clip(normal_hours, 0, 23)

    # Add legit off-hours
    legit_offhours = np.random.choice(normal_size, size = int(normal_size * 0.25), replace = False)  # Increased from 15% to 25%
    normal_hours[legit_offhours] = np.random.choice(range(0, 24), size = len(legit_offhours))

    # Add noise
    hour_noise_indices = np.random.choice(normal_size, size = int(normal_size * 0.15), replace = False)  # Increased from 5% to 15%
    normal_hours[hour_noise_indices] = np.random.choice(range(0, 24), size = len(hour_noise_indices))

    anomaly_hours = np.random.normal(loc = 13, scale = 6, size = anomaly_size)  # Changed from loc=2 to loc=13
    anomaly_hours = np.clip(anomaly_hours, 0, 23).astype(int)
    
    # 60% of anomalies occur at truly suspicious times
    hard_hour_anomalies = np.random.choice(anomaly_size, size = int(anomaly_size * 0.6), replace = False)
    anomaly_hours[hard_hour_anomalies] = np.random.choice([0, 1, 2, 3, 4], size = len(hard_hour_anomalies))
    anomaly_hours = np.clip(anomaly_hours, 0, 23).astype(int)
    """=== HOURS ACCESSED ENTRIES ==="""

    ####################################################################################################################

    """=== DAYS SINCE DISCHARGE ENTRIES ==="""
    # Normal: recent discharges (peaks 0-30 days)
    normal_dsd = np.random.lognormal(mean = 2.5, sigma = 1.5, size = normal_size)
    normal_dsd = np.clip(normal_dsd, 0, 365).astype(int) # Cap at 1yr

    # Old but normal accesses
    audit_indices = np.random.choice(normal_size, size = int(normal_size * 0.1), replace = False)
    normal_dsd[audit_indices] = np.random.uniform(300, 1000, size = len(audit_indices)).astype(int)

    # Anomaly: mostly recent accesses like normal (insider threat masquerading as legitimate)
    anomaly_dsd = np.random.lognormal(mean = 2.5, sigma = 1.2, size = anomaly_size).astype(int)
    anomaly_dsd = np.clip(anomaly_dsd, 0, 365).astype(int)
    
    # 50% access very old records (obvious anomalies)
    old_anomaly_dsd = np.random.choice(anomaly_size, size = int(anomaly_size * 0.5), replace = False)
    anomaly_dsd[old_anomaly_dsd] = np.random.uniform(300, 1000, size = len(old_anomaly_dsd)).astype(int)
    """=== DAYS SINCE DISCHARGE ENTRIES ==="""

    ####################################################################################################################

    """=== NUMBER OF LOGIN FAILURES ENTRIES ==="""
    # Normal: few login failures (occasional typo, credential mixup)
    normal_login_fails = np.random.poisson(lam = 0.5, size = normal_size)

    # Anomaly: significantly more login failures (obvious pattern)
    anomaly_login_fails = np.random.poisson(lam = 2.5, size = anomaly_size)
    """=== NUMBER OF LOGIN FAILURES ENTRIES ==="""

    ####################################################################################################################

    """=== IP & MAC ADDRESSES ==="""
    normal_unique_ips = [fake.ipv4(private = 'Public') for _ in range(250)]
    anomaly_unique_ips = [fake.ipv4(private = 'Public') for _ in range(50)]  # Completely different IPs
    unique_macs = [fake.mac_address() for _ in range(430)]

    # Normal: uses many different IPs from their pool
    normal_ips = np.random.choice(normal_unique_ips, size = normal_size)
    normal_macs = np.random.choice(unique_macs, size = normal_size)
    
    # Anomaly: mix patterns - some use concentrated IPs (insider), others use many (external)
    concentrated_anomalies = int(anomaly_size * 0.6)  # 60% use 5 IPs
    scattered_anomalies = anomaly_size - concentrated_anomalies  # 40% use random IPs
    
    anomaly_ips_concentrated = np.random.choice(anomaly_unique_ips, size=5)
    anomaly_ips_part1 = np.random.choice(anomaly_ips_concentrated, size=concentrated_anomalies)
    anomaly_ips_part2 = np.random.choice(normal_unique_ips, size=scattered_anomalies)  # Use normal IPs too
    anomaly_ips = np.concatenate([anomaly_ips_part1, anomaly_ips_part2])
    
    anomaly_macs_list = np.random.choice(unique_macs, size=5)
    anomaly_macs = np.random.choice(anomaly_macs_list, size=anomaly_size)

    all_ips = np.concatenate([normal_ips, anomaly_ips])
    all_macs = np.concatenate([normal_macs, anomaly_macs])
    """=== IP & MAC ADDRESSES ==="""

    ####################################################################################################################

    """=== TRUE_LABELS ==="""
    true_labels = np.concatenate([np.zeros(normal_size), np.ones(anomaly_size)]).astype(bool)
    
    ####################################################################################################################

    """=== PUTTING DATA TOGETHER ==="""
    durations = np.concatenate([normal_duration, anomaly_durations]).astype(int)
    hours = np.concatenate([normal_hours, anomaly_hours]).astype(int)
    dsd = np.concatenate([normal_dsd, anomaly_dsd]).astype(int)
    login_fails = np.concatenate([normal_login_fails, anomaly_login_fails]).astype(int)


    # Shuffle data together
    shuffle_indices = np.random.permutation(len(durations))
    all_ips = [all_ips[i] for i in shuffle_indices]
    all_macs = [all_macs[i] for i in shuffle_indices]
    durations = durations[shuffle_indices]
    hours = hours[shuffle_indices]
    dsd = dsd[shuffle_indices]
    login_fails = login_fails[shuffle_indices]
    true_labels = true_labels[shuffle_indices]

    records = []
    for i, (duration, hour, label, days, login_fail) in enumerate(zip(durations, hours, true_labels, dsd, login_fails)):
        records.append({
            'is-anomaly': label,
            'ip-address': all_ips[i],
            'mac-address': all_macs[i],
            'access-duration': duration,
            'hour-accessed': hour,
            'days-since-discharge': days,
            'login-failures': login_fail
        })
    
    data_frame = pd.DataFrame(records)
    ip_frequency = data_frame['ip-address'].value_counts().to_dict()
    data_frame['ip-access-count'] = data_frame['ip-address'].map(ip_frequency)

    features = [
        'ip-access-count',
        'access-duration',
        'hour-accessed',
        'days-since-discharge',
        'login-failures'
    ]
    """=== PUTTING DATA TOGETHER ==="""

    ####################################################################################################################
    
    return data_frame, features

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

def scores(true_labels, predictions):
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predictions)
    
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Calculate Sensitivity and Specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate

    return f1, precision, recall, accuracy, conf_matrix, sensitivity, specificity

def print_scores(title, f1, precision, recall, accuracy, conf_matrix, sensitivity, specificity):
    print(f"\n\n==== {title} ====")
    print(f"F1-SCORE: {f1:.4f}")
    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL: {recall:.4f}")
    print(f"ACCURACY: {accuracy:.4f}")
    print(f"SENSITIVITY: {sensitivity:.4f}")
    print(f"SPECIFICITY: {specificity:.4f}")

    print(f"\nConfusion Matrix:\n{conf_matrix}")
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print("============================================")