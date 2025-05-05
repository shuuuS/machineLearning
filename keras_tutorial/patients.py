import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras

train_labels = []
train_samples = []

# Instruction
# Experiment drug was tested on 2100 individuals between 13 to 100
#  years of age.
# Half the participants are under 65 
# around 95% of participants under 65 experienced no side effects
# 95% of participants over 65 experienced side effects

for i in range(50):
    # 5% younger people who had side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # 5% older people who didn't have side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

    # index [0] for no side effects; index [1] for side effects

for i in range(1000):
    # 95% younger people who didn't have side effects
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # 95% older people who had side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

print(len(train_samples), train_samples[:5])
print(len(train_labels), train_labels[:5])


train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_samples, train_labels = shuffle(train_samples, train_labels)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
print(scaled_train_samples[:5])

# Creating Artificial Neural Network
