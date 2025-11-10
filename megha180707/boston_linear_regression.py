import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Users\\gowda\\Downloads\\Boston.csv")       
x = data['rm'].values
y = data['medv'].values
mean_x = np.mean(x)
mean_y = np.mean(y)
numerator = np.sum ((x - mean_x)*(y - mean_y))
denominator = np.sum ((x-mean_x)**2)
w = numerator/denominator
b = mean_y - w * mean_x
y_pred = w * x + b
plt.scatter (x, y, color = 'blue', label = "actual data")
plt.plot (x, y_pred, color = 'red', label = "regression line")
plt.xlabel("RM")
plt.ylabel("cost")
plt.title ("linear regression")
plt.legend()
plt.show()
mse = np.sum ((y-y_pred)**2)
print (f"slope (w) : {w:.4f}")
print (f"intercept (b) : {b:.4f}")
print (f"mean squared error : {mse : .4f}")