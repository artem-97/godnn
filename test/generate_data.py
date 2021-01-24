import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import os

n_samples = 10_000

X, y = datasets.make_moons(n_samples=n_samples)

# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

np.savetxt(os.path.join('data', 'X_train.csv'), X_train, delimiter=',')
np.savetxt(os.path.join('data', 'y_train.csv'), y_train, delimiter=',')
np.savetxt(os.path.join('data', 'X_test.csv'), X_test, delimiter=',')
np.savetxt(os.path.join('data', 'y_test.csv'), y_test, delimiter=',')
