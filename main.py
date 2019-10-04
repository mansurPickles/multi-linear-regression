import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("ex1data2.txt", names=['x1', 'x2', 'y1'])

theta0 = 0
theta1 = 0
theta2 = 0


size = len(df['y1'])

learning_rate = .0000003


"""

#theta0 = 142.41073111330002
#theta1 = 164.96471558005143
#theta2 = 247.06855807929162

for m in range(size):
    cost = theta0 + (theta1 * df['x1'][m]) + (theta2 * df['x2'][m])
    tcost = cost - df['y1'][m]
    print(f'prediction {cost}    actual value {df["y1"][m]}')


"""


for it in range(50000):
    print_cost = 0
    temp0 = 0
    temp1 = 0
    temp2 = 0

    for m in range(size):
        cost = theta0 + (theta1 * df['x1'][m]) + (theta2 * df['x2'][m])
        cost = cost - df['y1'][m]
        print_cost += (pow(cost,2))
        temp0 += cost
        temp1 += (cost * df['x1'][m])
        temp2 += (cost * df['x2'][m])

    #update
    print_cost = print_cost / (2*size)

    theta0 = theta0 - (learning_rate * temp0)/size
    theta1 = theta1 - (learning_rate * temp1)/size
    theta2 = theta2 - (learning_rate * temp2)/size

print(f'cost {print_cost}')
print(theta0)
print(theta1)
print(theta2)

