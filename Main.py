# %% Importing Libaries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xg
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


# %% Importing the dataset:

data = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\playground-series-s4e5\train.csv")
test = pd.read_csv(r"C:\Users\umesh\OneDrive\Desktop\playground-series-s4e5\test.csv")

# %% Getting the basic info about the data:

print(data.info())
data_stats = (data.describe())

# %% Data Preprocessing:

sub = test[['id']]
minmax = MinMaxScaler()
print(data.isna().sum())
print(data.duplicated().sum())
data = data.drop(columns=['id'])
test = test.drop(columns=['id'])
data.iloc[:, :] = data.iloc[:, :].astype("float64")
test.iloc[:, :] = test.iloc[:, :].astype("float64")

# %% MinMax Scaling of the continuous data:

data.iloc[:, :20] = minmax.fit_transform(data.iloc[:, :20])
test.iloc[:, :20] = minmax.fit_transform(test.iloc[:, :20])

# %% Feature Engineering and getting new columns:

col = data.columns

for i in col[:-1]:
    mean = data[i].mean()
    data[i + "_Prob"] = data.loc[:, i].apply(lambda x: 1 if x >= mean else 0)
    mean1 = test[i].mean()
    test[i + "_Prob"] = test.loc[:, i].apply(lambda x: 1 if x >= mean1 else 0)

data['ClimateAnthropogenicInteraction'] = (data['MonsoonIntensity'] + data['ClimateChange']) * (
        data['Deforestation'] + data['Urbanization'] + data['AgriculturalPractices'] + data['Encroachments'])
data['InfrastructurePreventionInteraction'] = (data['DamsQuality'] + data['DrainageSystems'] + data[
    'DeterioratingInfrastructure']) * (data['RiverManagement'] + data['IneffectiveDisasterPreparedness'] + data[
    'InadequatePlanning'])

data['ClimateAnthropogenicInteraction'] = (data['MonsoonIntensity'] + data['ClimateChange']) * (
        data['Deforestation'] + data['Urbanization'] + data['AgriculturalPractices'] + data['Encroachments'])
data['InfrastructurePreventionInteraction'] = (data['DamsQuality'] + data['DrainageSystems'] + data[
    'DeterioratingInfrastructure']) * (data['RiverManagement'] + data['IneffectiveDisasterPreparedness'] + data[
    'InadequatePlanning'])

test['ClimateAnthropogenicInteraction'] = (test['MonsoonIntensity'] + test['ClimateChange']) * (
        test['Deforestation'] + test['Urbanization'] + test['AgriculturalPractices'] + test['Encroachments'])
test['InfrastructurePreventionInteraction'] = (test['DamsQuality'] + test['DrainageSystems'] + test[
    'DeterioratingInfrastructure']) * (test['RiverManagement'] + test['IneffectiveDisasterPreparedness'] + test[
    'InadequatePlanning'])

# %% Getting the correlation between different attributes:

data_corr = data.corr()
plt.figure(figsize=(13, 8))
sn.heatmap(data_corr)
plt.show()

# %% Implementation of ML algorithm:

'''

X = data.drop(columns='FloodProbability')
y = data[['FloodProbability']]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
X_train = data.drop(columns='FloodProbability')
y_train = data[['FloodProbability']]
X_test = test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
class1 = xg.XGBRegressor(n_jobs=-1)
class1.fit(X_train, y_train)
prediction = class1.predict(X_test)
#print(r2_score(y_test, p))

'''

# %% Developing a RNN model:

model = Sequential([
    SimpleRNN(50, input_shape=(None, 1), activation='relu', return_sequences=False),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='linear')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=[r2_score])

# %% Splitting the data before fitting:

X = data.drop(columns='FloodProbability')
y = data[['FloodProbability']]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# %% Running the Deep Learning RNN model:

model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

# %% Make predictions for the test data:

predictions = model.predict(X_test)

# %% Getting the prediction for submission:

sub['FloodProbability'] = predictions

sub.to_csv('submission.csv', index=False)
print(sub.head())

# %%
