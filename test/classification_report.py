from sklearn.metrics import classification_report
import pandas as pd

import os

# read data
torch_pred = pd.read_csv(os.path.join('data', 'torch_pred.csv'),
                         header=None).values

godnn_pred = pd.read_csv(os.path.join('data', 'godnn_pred.csv'),
                         header=None).values

y_test = pd.read_csv(os.path.join('data', 'y_test.csv'), header=None).values

print("GoDNN model")
print(classification_report(y_test, godnn_pred))

print("PyTorch model")
print(classification_report(y_test, torch_pred))
