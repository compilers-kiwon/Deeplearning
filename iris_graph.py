import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset/iris.csv',names=["sepal_length",
                 "sepal_width","petal_length","petal_width","species"])
print(df.head())

sns.pairplot(df,hue='species')
plt.show()