import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

# Step 1: Load full dataset
print(" Loading dataset...")
df_full = pd.read_csv('https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv')

# Step 2: Save the original dataset before any preprocessing
df_full.to_csv('original_data.csv', index=False)
print("Full original dataset saved as 'original_data.csv'.")

# Step 3: Filter required labels
df = df_full[df_full['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
df['koi_disposition'] = df['koi_disposition'].map({'FALSE POSITIVE': 0, 'CONFIRMED': 1})

# Step 4: Define features
features = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
    'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
    'koi_tce_plnt_num', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_kepmag',
    'koi_tce_delivname'
]

# Step 5: Drop rows with missing values
df = df[features + ['koi_disposition']]
df = df.dropna()

# Step 6: Handle missing categorical column if needed
if 'koi_tce_delivname' not in df.columns:
    df['koi_tce_delivname'] = 'unknown'

# Step 7: One-hot encoding for 'koi_tce_delivname'
df = pd.get_dummies(df, columns=['koi_tce_delivname'], prefix='tce_deliv')
final_features = df.drop('koi_disposition', axis=1).columns.tolist()

# Step 8: Train-test split
X = df[final_features]
y = df['koi_disposition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Train ensemble model
print(" Training model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(probability=True, random_state=42)
ensemble = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('lr', lr)], voting='soft')
ensemble.fit(X_train_scaled, y_train)

y_pred = ensemble.predict(X_test_scaled)
print(" Training complete.\n")
print(" Accuracy:", accuracy_score(y_test, y_pred))
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(" Precision:", precision_score(y_test, y_pred))
print(" Recall:", recall_score(y_test, y_pred))
print(" F1 Score:", f1_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['False Positive', 'Confirmed'], yticklabels=['False Positive', 'Confirmed'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# Step 12: Save model, scaler, and features
joblib.dump(ensemble, 'exo_best_model.pkl')
joblib.dump(scaler, 'exo_scaler.pkl')
joblib.dump(final_features, 'exo_features.pkl')
print("Model, scaler, and features saved.")