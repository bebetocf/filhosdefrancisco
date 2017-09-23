import pandas as pd

data = pd.read_csv("segmentation_test.csv")

data_shape = pd.concat([data.iloc[:, 0:9], data.iloc[:, 19]], axis=1)
data_color = data.iloc[:, 9:20]
