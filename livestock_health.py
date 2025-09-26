import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings

warnings.filterwarnings('ignore')
print("--- Task 1: Logistic Regression on Synthetic Dataset ---")

np.random.seed(42)
n_samples = 200
data = {
    'body_temperature': np.random.uniform(36.5, 40.5, n_samples),
    'heart_rate': np.random.uniform(50, 120, n_samples),
    'respiratory_rate': np.random.uniform(10, 40, n_samples),
    'activity_level': np.random.uniform(0, 1, n_samples),
    'feed_intake': np.random.uniform(5, 20, n_samples),
    'water_intake': np.random.uniform(10, 40, n_samples),
}
synthetic_df = pd.DataFrame(data)

synthetic_df['health_status'] = np.where(
    (synthetic_df['body_temperature'] > 39.5) | 
    (synthetic_df['heart_rate'] > 100) | 
    (synthetic_df['respiratory_rate'] > 35) |
    (synthetic_df['activity_level'] < 0.3),
    'Unhealthy', 'Healthy'
)

X_synthetic = synthetic_df.drop('health_status', axis=1)
y_synthetic = synthetic_df['health_status']

le_synthetic = LabelEncoder()
y_synthetic_encoded = le_synthetic.fit_transform(y_synthetic)

X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_synthetic, y_synthetic_encoded, test_size=0.2, random_state=42, stratify=y_synthetic_encoded
)

scaler_syn = StandardScaler()
X_train_syn_scaled = scaler_syn.fit_transform(X_train_syn)
X_test_syn_scaled = scaler_syn.transform(X_test_syn)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_syn_scaled, y_train_syn)

y_pred_syn = lr_model.predict(X_test_syn_scaled)

print("\nLogistic Regression (Synthetic Data) Metrics:")
print(f"Accuracy: {accuracy_score(y_test_syn, y_pred_syn):.4f}")
print(f"Precision: {precision_score(y_test_syn, y_pred_syn):.4f}")
print(f"Recall: {recall_score(y_test_syn, y_pred_syn):.4f}")
print(f"F1 Score: {f1_score(y_test_syn, y_pred_syn):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_syn, y_pred_syn))

print("\n\n--- Task 2: Model Comparison on Provided CSVs ---")


try:
    
    train_df = pd.read_csv('Training.csv', header=0)
    test_df = pd.read_csv('Testing.csv', header=0)
except FileNotFoundError:
    print("\n'Training.csv' or 'Testing.csv' not found. Skipping Task 2.")
else:
    target_column = train_df.columns[-1]
    train_df.dropna(subset=[target_column], inplace=True)c
    test_df.dropna(subset=[target_column], inplace=True)


    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]
    
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0
    X_test = X_test[train_cols]

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    for col in X_train.columns:
        if X_train[col].isnull().any():
            mean_val = X_train[col].mean()
            X_train[col].fillna(mean_val, inplace=True)
            if col in X_test.columns:
                X_test[col].fillna(mean_val, inplace=True)


    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=10, min_samples_split=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5),
        'Gaussian Naive Bayes': GaussianNB()
    }

    results = {}
    best_model_name = ''
    best_f1_score = 0.0

    for name, model in models.items():
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_pred, average='weighted')
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': cm
        }

        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            joblib.dump(model, 'model.pkl')

   
    results_df = pd.DataFrame(results).T.drop(columns=['Confusion Matrix'])
    print("\nModel Performance Comparison:")
    print(results_df)

    print("\nConfusion Matrices:")
    for name, metrics in results.items():
        print(f"\n--- {name} ---")
        print(metrics['Confusion Matrix'])

    print(f"\nBest model based on F1-score is '{best_model_name}'.")
    print("Best model saved as 'model.pkl'.") 