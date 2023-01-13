import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset/pima-indians-diabetes.csv',
                 names=["pregnant","plasma","pressure","thickness",
                 "insulin","BMI","pedigree","age","class"])

print(df.head(5))
print(df.info())
print(df.describe())

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),linewidth=0.1,vmax=0.5,
            cmap=plt.cm.gist_heat,linecolor='white',annot=True)
plt.show()

grid = sns.FacetGrid(df,col='class')
grid.map(plt.hist,'plasma',bins=10)
plt.show()

print(df[['plasma','class']].groupby(['plasma'],
      as_index=False).mean().sort_values(by='plasma',ascending=True))
