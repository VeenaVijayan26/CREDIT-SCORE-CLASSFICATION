import pandas as pd
import numpy as np
import category_encoders as ce
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


# Load the dataset'
data = pd.read_csv('cleaned_data_outlier.csv')

data.drop('Month',axis=1,inplace=True)

#numerical columns
num_cols=data.select_dtypes('number').columns

#apply log transfo on all num_cols
for i in num_cols:
  data[i]=np.log1p(data[i])

  label_en = LabelEncoder()

  data['Credit_Score'] = label_en.fit_transform(data['Credit_Score'])

# Separate features and target
X = data.drop('Credit_Score', axis=1)
y = data['Credit_Score']

encoder = ce.BinaryEncoder(cols=['Occupation','Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour' ])



# Perform the encoding
X_encoded = encoder.fit_transform(X)

import joblib
joblib.dump(encoder, 'encoder.joblib')

 # Synthetic Minority Oversampling Technique

smote = SMOTE()
X_encoded, y = smote.fit_resample(X_encoded,y)

#new data merging X_encoded and y
data_encoded = pd.concat([X_encoded, y], axis=1)

#splitting data into train,validaton and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
#split to validation
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)


#scaling
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#selected random forest classifier
rf_clf = RandomForestClassifier(n_estimators=400, max_depth=70, criterion='entropy')
rf_clf.fit(X_train, y_train)
y_val_pred = rf_clf.predict(X_val)
print('Random Forest Accuracy is', accuracy_score(y_val, y_val_pred))

# Predict on the test set
y_test_pred = rf_clf.predict(X_test)
# Calculate the test accuracy
test_accuracy2 = accuracy_score(y_test, y_test_pred)


#dumb to pkl
pickle.dump(rf_clf, open('model.pkl','wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))





