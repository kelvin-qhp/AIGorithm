### ##1. Pandas DataFrame 

![](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1760511608228.png)

~~~
dataset.groupby('category')['category'].count()
dataset['category'].value_counts()
dataset['category'].unique()

~~~

![](C:\Users\user\AppData\Roaming\Typora\typora-user-images\1760598396261.png)







### PD dataframe groupby

```
# df1 = dataset_df.groupby('category').agg(cnt=('category','count'))
# df1.reset_index()

df2 =dataset_df.groupby('category',as_index=False)['category'].agg({'cnt':'count'})
```

