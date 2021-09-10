import pandas as pd
import plotly.express as pe
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.core.fromnumeric import reshape,shape

yvalue = []

df = pd.read_csv('data.csv')
velocity = df['Velocity'].tolist()
escaped = df['Escaped'].tolist()

velocity_array = np.array(velocity)
escaped_array = np.array(escaped)

m, c = np.polyfit(velocity_array, escaped_array, 1)
print(m, c)

for x in velocity:
    y = m * x + c
    yvalue.append(y)

graph = pe.scatter(x=velocity, y=escaped)
graph.update_layout(shapes=[dict(type="line", y0=min(
    yvalue), y1=max(yvalue), x0=min(velocity), x1=max(velocity))])
graph.show()

lr = LogisticRegression()

velocity_reshape = np.reshape(velocity, (len(velocity),1))
escaped_reshape = np.reshape(escaped, (len(escaped),1))

lr.fit(velocity_reshape,escaped_reshape)

def logistic(x) :
    y = 1 / (1 + np.exp(-x))
    return y

#Giving Input
#print(lr.coef_,lr.intercept_)
m = lr.coef_
c = lr.intercept_
yvaluee = logistic(m*12 + c)
print(yvaluee)
