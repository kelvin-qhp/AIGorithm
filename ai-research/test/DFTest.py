import pandas as pd

data = {
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'one'],
    'C': [1, 2, 3, 4, 5, 6, 7, 8],
    'D': [2, 3, 4, 5, 6, 7, 8, 9]
}
df = pd.DataFrame(data)
# print(df)

# result = df.agg(pd.Series.nunique)
# print(result)

# result = df.agg({'A': [pd.Series.nunique,pd.Series.count]})
# print(result)

# result = df.agg({'A': pd.Series.nunique, 'B': pd.Series.nunique})
# print(result)

# result = df.groupby('A',as_index=False).agg({'B': pd.Series.nunique})
# print(result)

# result = df.groupby('A',as_index=False)['A'].agg({'cnt': pd.Series.nunique})
# result = df.groupby('A',as_index=False).agg(cnt=('A','count'))
# print(result)

# result = df.agg({'A': lambda x: x.nunique()})
# print(result)

# result = df.agg({'A': ['count', pd.Series.nunique]})
# print(result)

# result = df.agg({'A': pd.Series.nunique, 'C': 'sum'})
# print(result)

# result = df[df['C'] > 3].agg({'A': pd.Series.nunique})
# print(result)

# def count_unique(series):
#     return series.nunique()
# result = df.agg({'A': count_unique})
# print(result)


# result = df.groupby('A').agg({'B': ['count', pd.Series.nunique]})
# print(result)

# result = df.apply(pd.Series.nunique)
# print(result)

# result = df.groupby('A').agg({'B': pd.Series.nunique}).reset_index()
# print(result)

# result = df.groupby(['A', 'B']).agg({'C': pd.Series.nunique})
# print(result)

# result = df.groupby(['A']).agg({'B': pd.Series.nunique}).reset_index().sort_values(by='B',ascending=True)
result = df.groupby('B',as_index=False)['B'].agg({'cnt': pd.Series.count}).sort_values(by='cnt',ascending=False)
print(result)
