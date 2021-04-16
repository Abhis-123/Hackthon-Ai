# import all modules
import pandas as pd, numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2, l1

# features
features = ['income', 'age', 'experience', 'married', 'house_ownership',
            'car_ownership', 'profession', 'city', 'state', 'current_job_years',
            'current_house_years', 'fail', 'pass']

data = pd.read_csv("balanced_data.csv")
# pass and fail loans
data['fail'] = data['risk_flag'].apply(lambda x: 1 if x == 0 else 0)
data['pass'] = data['risk_flag'].apply(lambda x: 0 if x == 0 else 1)


# function for normalisaation
def normalise_column(li):
    for column in li:
        min = np.min(data[column])
        max = np.max(data[column])
        data[column] = data[column].apply(lambda x: (x - min) / (max - min))


# x values for training data
x = data[['income', 'age', 'experience', 'married', 'house_ownership',
          'car_ownership', 'profession', 'city', 'state', 'current_job_years',
          'current_house_years']].values
y = data[['pass', 'fail']].values

model = Sequential()
model.add(tf.keras.Input(shape=(11)))
model.add(Dense(11))
model.add(Dense(11, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
y=data['risk_flag'].values

model.compile(
    loss= tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)
model.summary()




#y=np.reshape(y,(-1,2))
print(y.shape)
model.fit(x,y,epochs=3,batch_size=8)


