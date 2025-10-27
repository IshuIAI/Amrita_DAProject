import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load and Initial Clean-up
df = pd.read_csv("healthcare_noshows.csv")
df_clean = df.copy()

# Create target: No_Show (1 = No-Show, 0 = Showed Up)
df_clean['No_Show'] = ~df_clean['Showed_up']
df_clean['No_Show'] = df_clean['No_Show'].astype(int)
df_clean = df_clean.drop(columns=['Showed_up', 'PatientId', 'AppointmentID', 'Date.diff'])

# 2. Date and Time Features
df_clean['ScheduledDay'] = pd.to_datetime(df_clean['ScheduledDay']).dt.date
df_clean['AppointmentDay'] = pd.to_datetime(df_clean['AppointmentDay']).dt.date
df_clean['Waiting_Days'] = (df_clean['AppointmentDay'] - df_clean['ScheduledDay']).dt.days
df_clean['Appointment_Weekday'] = pd.to_datetime(df_clean['AppointmentDay']).dt.weekday

# 3. Handle Anomalies
# Remove rows with negative Age (1 row) or negative Waiting_Days (5 rows)
df_clean = df_clean[df_clean['Age'] >= 0]
df_clean = df_clean[df_clean['Waiting_Days'] >= 0]

# 4. Feature Engineering for Model
conditions = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap']
bins = [0, 12, 18, 40, 60, 120]
labels = ['Child (0-12)', 'Adolescent (13-17)', 'Young Adult (18-39)', 'Middle-Aged (40-59)', 'Senior (60+)']
df_clean['Age_Group'] = pd.cut(df_clean['Age'], bins=bins, labels=labels, right=False, include_lowest=True)

features_for_model = ['Gender', 'Age_Group', 'Waiting_Days', 'SMS_received'] + conditions
df_model = df_clean[features_for_model + ['No_Show']].copy()

# Convert boolean/SMS features to integer (0 or 1)
for col in conditions + ['SMS_received']:
    df_model[col] = df_model[col].astype(int)

# One-hot encode categorical features
df_model = pd.get_dummies(df_model, columns=['Gender'], drop_first=True, prefix='Gender')
df_model = pd.get_dummies(df_model, columns=['Age_Group'], drop_first=True, prefix='Age')

# 5. Prepare Data for Training
X = df_model.drop('No_Show', axis=1)
y = df_model['No_Show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. Train Decision Tree Model
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced')
dt_classifier.fit(X_train, y_train)

# 7. Evaluate Model
y_pred = dt_classifier.predict(X_test)
y_proba = dt_classifier.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred, target_names=['Showed Up', 'No Show']))

# Extract Feature Importances
feature_importances = pd.Series(dt_classifier.feature_importances_, index=X.columns)
print("Top 5 Feature Importances:")
print(feature_importances.sort_values(ascending=False).head())