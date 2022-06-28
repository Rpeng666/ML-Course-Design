import torch
from torchvision import transforms
import numpy as np
import pickle



with open('./origin_data/data_batch_1', 'rb') as file:

    data = pickle.load(file, encoding='bytes')

    data = data[b'data']

print(data)