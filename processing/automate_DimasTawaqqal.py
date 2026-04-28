import subprocess
import sys
import os
import importlib.util

_user_site = os.path.join(
    os.path.expanduser("~"),
    ".local", "lib",
    f"python{sys.version_info.major}.{sys.version_info.minor}",
    "site-packages"
)
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

_pkg_to_module = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
}

for _pkg, _mod in _pkg_to_module.items():
    if importlib.util.find_spec(_mod) is None:
        print(f"Menginstall '{_pkg}'...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", _pkg,
            "--break-system-packages", "--quiet"
        ])
        importlib.invalidate_caches()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import argparse

def preprocess_bank_marketing(input_path, output_dir='preprocessed_data'):
    print(f"Memuat dataset dari {input_path}...")
    df = pd.read_csv(input_path, sep=';')
    print(f"Dataset berisi {df.shape[0]} baris dan {df.shape[1]} kolom")
    
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)
        print("Fitur 'duration' dihapus untuk mencegah data leakage")
    
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                            'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    for col in categorical_features:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].replace('unknown', mode_val)
    print("Nilai 'unknown' telah diganti dengan modus")
    
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['y'])
    print("Target variable 'y' telah di-encode (0: tidak, 1: ya)")
    
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    print(f"Setelah one-hot encoding, dataset memiliki {df.shape[1]} kolom")
    
    X = df.drop('y', axis=1)
    y = df['y']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data: train={X_train.shape[0]}, test={X_test.shape[0]}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Setelah SMOTE: train={X_train_resampled.shape[0]}")
    
    numeric_features = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                        'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    available_numeric = [f for f in numeric_features if f in X_train_resampled.columns]
    
    scaler = StandardScaler()
    X_train_resampled[available_numeric] = scaler.fit_transform(X_train_resampled[available_numeric])
    X_test[available_numeric] = scaler.transform(X_test[available_numeric])
    print("Scaling fitur numerik selesai")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_df = pd.concat([X_train_resampled, y_train_resampled.rename('y')], axis=1)
    test_df = pd.concat([X_test, y_test.rename('y')], axis=1)
    
    train_df.to_csv(os.path.join(output_dir, 'X_train_preprocessed.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'X_test_preprocessed.csv'), index=False)
    
    pd.DataFrame(y_train_resampled, columns=['y']).to_csv(os.path.join(output_dir, 'y_train_preprocessed.csv'), index=False)
    pd.DataFrame(y_test, columns=['y']).to_csv(os.path.join(output_dir, 'y_test_preprocessed.csv'), index=False)
    
    import joblib
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder_y.pkl'))
    
    print(f"Hasil preprocessing disimpan di folder '{output_dir}'")
    
    return {
        'X_train': X_train_resampled,
        'X_test': X_test,
        'y_train': y_train_resampled,
        'y_test': y_test,
        'scaler': scaler,
        'label_encoder': le
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing dataset Bank Marketing')
    parser.add_argument('--input', type=str, default='data/raw/bank-additional-full.csv',
                        help='Path ke file CSV raw')
    parser.add_argument('--output', type=str, default='processing',
                        help='Direktori output untuk menyimpan hasil')
    args = parser.parse_args()
    
    preprocess_bank_marketing(args.input, args.output)
    print("Preprocessing selesai!")