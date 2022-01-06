import time
import gym
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array



class DogsvsCatsEnv(gym.Env):
    def __init__(self):
        # Two discrete actions (cat = 0, dog = 1)
        self.action_space = gym.spaces.Discrete(2)
        # image as input, already resized
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                        shape=(200, 200, 3), dtype=np.uint8)
        self.c = 0
        self.expected_class = 1
        
    def step(self, action):
        reward = int(action == self.expected_class)
        img = self.next_img()
        self.c += 1
        done = True        
        return img, reward, done, {}
    
    def reset(self):
        img = self.next_img()
        return img
    
    def next_img(self):
        if self.expected_class == 1:
            self.expected_class = 0
            filename = 'train/cat.' + str(self.c) + '.jpg'
    
        if self.expected_class == 0:
            self.expected_class = 1
            filename = 'train/dog.' + str(self.c) + '.jpg'
            self.c+=1

        picture = load_img(filename, target_size=(200, 200))
	    # convert to numpy array
        picture = img_to_array(picture)
        return picture

