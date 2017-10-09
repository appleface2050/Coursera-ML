"""
http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import os
path = "D:\git\Coursera-ML\johnwittenauer\data\ex1data2.txt"
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()
