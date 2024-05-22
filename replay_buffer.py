import random

import numpy as np
from collections import namedtuple, deque

# Define a named tuple for storing experiences
Experience = namedtuple("Experience", field_names=["query", "trajectory", "reward"])

class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize a ReplayBuffer.
        :param capacity: Maximum number of experiences to store in the buffer.
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, query, trajectory, reward):
        """
        Add a new experience to memory.
        :param query: The query associated with the experience.
        :param trajectory: The trajectory of the experience.
        :param reward: The reward of the experience.
        """
        e = Experience(query, trajectory, reward)
        self.memory.append(e)

    def sample(self, batch_size=1):
        """
        Randomly sample a batch of experiences from memory.
        :param batch_size: Number of experiences to sample.
        :return: A list of sampled experiences.
        """
        return random.sample(self.memory, k=batch_size)
       

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
