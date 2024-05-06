# -*- coding: utf-8 -*-


import psutil
import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Scale the data
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, LSTM, Bidirectional, Dense,Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam
import pandas as pd

# Get the current process ID
pid = os.getpid()

# Get the process object for the current process
process = psutil.Process(pid)




input_file = "/content/Scenario2.csv"



x= pd.read_csv(input_file, usecols=[9,13,14,15,16,17,18])


y= pd.read_csv(input_file, usecols=[0 ,1,19])

x=x.dropna()
y=y.dropna()

train_percentage = 1


# Select the index to split the data
split_index = int(train_percentage * len(x))       #case 1 o-a
#split_index=609                                   #case 2 k-a
#split_index=366                                   #case 3 e-a


x=x[:split_index]
y= y[:split_index]

scaler = MinMaxScaler(feature_range=(-1,1))
scaler1 = MinMaxScaler(feature_range=(-1,1))
x = scaler.fit_transform(x)
y = scaler1.fit_transform(y)

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file, usecols=[9,13,14,15,16,17,18])


# Flip the column upside down
df['g1']= df['g1'][:split_index]
df.dropna(subset=['g1'], inplace=True)
df['g1']= df['g1'].values[::-1]*-1

df['g2']= df['g2'][:split_index]
df.dropna(subset=['g2'], inplace=True)
df['g2']= df['g2'].values[::-1]*-1

df['g3']= df['g3'][:split_index]
df.dropna(subset=['g3'], inplace=True)
df['g3']= df['g3'].values[::-1]*-1

df['T']= df['T'][:split_index]
df.dropna(subset=['T'], inplace=True)
df['T']= df['T'].values[::-1]

df['ax']= df['ax'][:split_index]
df.dropna(subset=['ax'], inplace=True)
df['ax']= df['ax'].values[::-1]

df['ay']= df['ay'][:split_index]
df.dropna(subset=['ay'], inplace=True)
df['ay']= df['ay'].values[::-1]

df['az']= df['az'][:split_index]
df.dropna(subset=['az'], inplace=True)
df['az']= df['az'].values[::-1]

# Save the DataFrame to a new CSV file
df.to_csv('XF.csv', index=False)

#########################################################################################

#Single model training
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, LSTM, Bidirectional, Dense,Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam


input_file = "/content/Scenario2.csv"



x= pd.read_csv(input_file, usecols=[9,13,14,15,16,17,18])
y= pd.read_csv(input_file, usecols=[0 ,1,19])

x=x.dropna()
y=y.dropna()

scaler = MinMaxScaler(feature_range=(-1,1))
scaler1 = MinMaxScaler(feature_range=(-1,1))
x = scaler.fit_transform(x)
y = scaler1.fit_transform(y)
x=x[:split_index]
y= y[:split_index]
x= np.reshape(x, (x.shape[0], x.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(Dropout(0.1))


model.add(Bidirectional(LSTM(64, return_sequences=False)))

model.add(Dense(25, activation='relu'))  # Add a dense layer with ReLU activation
model.add(Dense(y_train.shape[1]))  # Output layer

# Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')

import time

pid = os.getpid()

# Get the process object for the current process
process = psutil.Process(pid)

# Get the memory usage before running the algorithm
start_memory = process.memory_info().rss

# Start the timer
start_time = time.time()


# Train the model
model.fit(x, y, batch_size=1, epochs=20)


# Get the memory usage after running the algorithm
end_memory = process.memory_info().rss

# Calculate the memory usage difference
memory_used = end_memory - start_memory
print("Memory used by algorithm: {:.2f} MB".format(memory_used / (1024 * 1024)))

# Stop the timer
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))
####################################################################################################

#Single model prediction

input_file1 = "/content/XF.csv"

xx= pd.read_csv(input_file1, usecols=[0,1,2,3,4,5,6])
xx=xx[:split_index]
xx=scaler.transform(xx)
xx= np.reshape(xx, (xx.shape[0], xx.shape[1], 1))

import time

# Start the timer
start_time = time.time()
predictionsall = model.predict(xx)
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print("Prediction time: {:.2f} seconds".format(training_time))

predictionsall = scaler1.inverse_transform(predictionsall)

# Display predictions
print(predictionsall)
df3 = pd.DataFrame(predictionsall,columns=['x2', 'y2','z2'])

# Save the DataFrame to a CSV file
df3.to_csv('predictions_single.csv', index=False)

#####################################################################################

#Parallel model x training
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Scale the data
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, LSTM, Bidirectional, Dense,Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam


input_file = "/content/Scenario2.csv"
x1= pd.read_csv(input_file, usecols=[9,13,16])
y1= pd.read_csv(input_file, usecols=[0])
x1=x1.dropna()
y1=y1.dropna()

x1=x1[:split_index]
y1= y1[:split_index]

scalerx = MinMaxScaler(feature_range=(-1,1))
scaler1x = MinMaxScaler(feature_range=(-1,1))


x1 = scalerx.fit_transform(x1)
y1 = scaler1x.fit_transform(y1)


x1= np.reshape(x1, (x1.shape[0], x1.shape[1], 1))
# Build the  LSTM model
modelx = Sequential()
modelx.add(LSTM(128, return_sequences=True, input_shape=(x1.shape[1], 1)))
modelx.add(Dropout(0.1))

modelx.add(Bidirectional(LSTM(64, return_sequences=False)))

modelx.add(Dense(25, activation='relu'))  # Add a dense layer with ReLU activation
modelx.add(Dense(y1.shape[1]))  # Output layer

# Compile the model
modelx.compile(optimizer='adam', loss='mean_squared_error')


import time

# Start the timer
start_time = time.time()


import psutil
import os

# Get the current process ID
pid = os.getpid()

# Get the process object for the current process
process = psutil.Process(pid)

# Get the memory usage before running the algorithm
start_memory = process.memory_info().rss

modelx.fit(x1, y1, batch_size=1, epochs=20)


# Get the memory usage after running the algorithm
end_memory = process.memory_info().rss

# Calculate the memory usage difference
memory_used = end_memory - start_memory
print("Memory used by algorithm: {:.2f} MB".format(memory_used / (1024 * 1024)))

end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print("Prediction time: {:.2f} seconds".format(training_time))

##############################################################################

#Parallel model x prediction

input_file1 = "/content/XF.csv"


xfx= pd.read_csv(input_file1, usecols=[0,1,4])
xfx=xfx[:split_index]
xfx=scalerx.transform(xfx)



xfx= np.reshape(xfx, (xfx.shape[0], xfx.shape[1], 1))

import time

# Start the timer
start_time = time.time()
predictionsx = modelx.predict(xfx)

end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print("Prediction time: {:.2f} seconds".format(training_time))

# Inverse transform the predictions if using scaling
predictionsx = scaler1x.inverse_transform(predictionsx)

# Display predictions
print(predictionsx)

###################################################################################

#Parallel model y training
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, LSTM, Bidirectional, Dense,Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam


input_file = "/content/Scenario2.csv"
x2= pd.read_csv(input_file, usecols=[9,14,17])
y2= pd.read_csv(input_file, usecols=[1])



x2=x2.dropna()
y2=y2.dropna()


x2=x2[:split_index]
y2= y2[:split_index]

scalery = MinMaxScaler(feature_range=(-1,1))
scaler1y = MinMaxScaler(feature_range=(-1,1))
x2 = scalery.fit_transform(x2)
y2 = scaler1y.fit_transform(y2)
x2= np.reshape(x2, (x2.shape[0], x2.shape[1], 1))

# Build the LSTM model
modely = Sequential()
modely.add(LSTM(128, return_sequences=True, input_shape=(x2.shape[1], 1)))
modely.add(Dropout(0.1))

modely.add(Bidirectional(LSTM(64, return_sequences=False)))

modely.add(Dense(25, activation='relu'))  # Add a dense layer with ReLU activation
modely.add(Dense(y2.shape[1]))  # Output layer

# Compile the model
modely.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
modely.fit(x2, y2, batch_size=1, epochs=20)

###############################################################################################

#Parallel model y prediction
input_file1 = "/content/XF.csv"
xfy= pd.read_csv(input_file1, usecols=[0,2,5])


xfy=scalery.transform(xfy)
xfy= np.reshape(xfy, (xfy.shape[0], xfy.shape[1], 1))


predictionsy = modely.predict(xfy)

# Inverse transform the predictions
predictionsy = scaler1y.inverse_transform(predictionsy)

# Display predictions
print(predictionsy)

##############################################################################################

#Parallel model z training
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, LSTM, Bidirectional, Dense,Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam

input_file = "/content/Scenario2.csv"
x3= pd.read_csv(input_file, usecols=[9,15,18])

y3= pd.read_csv(input_file, usecols=[19])


x3=x3.dropna()
y3=y3.dropna()
x3=x3[:split_index]
y3= y3[:split_index]


scalerz = MinMaxScaler(feature_range=(-1,1))
scaler1z = MinMaxScaler(feature_range=(-1,1))
x3 = scalerz.fit_transform(x3)
y3 = scaler1z.fit_transform(y3)
x3= np.reshape(x3, (x3.shape[0], x3.shape[1], 1))
# Build the LSTM model
modelz = Sequential()
modelz.add(LSTM(128, return_sequences=True, input_shape=(x3.shape[1], 1)))
modelz.add(Dropout(0.1))
modelz.add(Bidirectional(LSTM(64, return_sequences=False)))

modelz.add(Dense(25, activation='relu'))  # Add a dense layer with ReLU activation
modelz.add(Dense(y3.shape[1]))  # Output layer

# Compile the model
modelz.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
modelz.fit(x3, y3, batch_size=1, epochs=20)

###########################################################################################

#Parallel model z prediction

input_file1 = "/content/XF.csv"

xfz= pd.read_csv(input_file1, usecols=[0,3,6])

xfz=xfz[:split_index]


xfz=scalerz.transform(xfz)
xfz= np.reshape(xfz, (xfz.shape[0], xfz.shape[1], 1))


predictionsz = modelz.predict(xfz)

# Inverse transform the predictions
predictionsz = scaler1z.inverse_transform(predictionsz)

# Display predictions
print(predictionsz)

#############################################################################################

# Create a DataFrame from the predictions

df1 = pd.DataFrame(predictionsx, columns=['x1'])
df1['y1']=predictionsy
df1['z1']=predictionsz
# Save the DataFrame to a CSV file
df1.to_csv('predictions_parallel.csv', index=False)

df2 = pd.DataFrame(scaler1.inverse_transform(y),columns=['x2', 'y2','z2'])       #actual data

# Save the DataFrame to a CSV file
df2.to_csv('actual.csv', index=False)