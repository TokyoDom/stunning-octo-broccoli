import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQNAgent:

  def __init__(self, observation_space, action_space):
    self.action_space = action_space

    self.gamma = 0.95

    self.memory = deque(maxlen=1000000)
    self.batch_size = 20

    self.explore_rate = 1
    self.explore_decay = 0.995
    self.explore_min = 0.01

    #model
    self.model = Sequential()
    self.model.add(Input(shape=(observation_space,)))
    self.model.add(Dense(24, "relu"))
    self.model.add(Dense(24, "relu"))
    self.model.add(Dense(self.action_space, "linear"))
    self.model.compile(Adam(), "mse")
  
  def action(self, state):
    #random exploration if true
    if np.random.rand() < self.explore_rate:
      return random.randrange(self.action_space)
    else:
      QValues = self.model.predict(state)
      return np.argmax(QValues[0])
  
  def save_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def replay(self):
    if len(self.memory) < self.batch_size: return

    batch = random.sample(self.memory, self.batch_size)
    for state, action, reward, next_state, done in batch:
      QUpdate = reward
      if not done:
        QUpdate = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
      
      QValues = self.model.predict(state)
      QValues[0][action] = QUpdate

      self.model.fit(state, QValues, verbose=0)
    
    self.explore_rate *= self.explore_decay
    self.explore_rate = max(self.explore_min, self.explore_rate)