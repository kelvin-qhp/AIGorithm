###  Misc
~~~
Python Study:
https://liaoxuefeng.com/books/python/async-io/asyncio/index.html#0
~~~



### 1. Pandas DataFrame 

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

### 2. Matrix Multiple
2.1 逐元素相乘：torch.mul()  or *
~~~
要求形状相同（shape)
x = 3
y = 4
A = torch.randn(x, y)
B = torch.randn(x, y)

print(A*B == torch.mul(A,B))
~~~

2.2 标准矩阵乘法，行 × 列
~~~
A = torch.tensor([[1, 2,3],
                  [4,5, 6]])

B = torch.tensor([[7, 8],
                  [9, 10],
                  [11, 12]])
print(A.shape,B.shape)
print(f'A*b=:{A@B},torch.mutmul:{torch.matmul(A,B)}')
print(A@B == torch.matmul(A,B))
~~~

2.3 Dot Product）— torch.dot()
~~~
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

torch.dot(a, b)  
# 1*4 + 2*5 + 3*6
~~~

2.4. 批量矩阵乘法 — torch.bmm() or torch.matmul():
(batch, n, m) * (batch, m, p) = (batch, n, p)
~~~
batch = 2

a = torch.randn(batch, 3, 4)
b = torch.randn(batch, 4, 5)
c = torch.bmm(a, b)
print(c.shape)

#torch.Size([2, 3, 5])

# matmul支持广播
a = torch.randn(2, 3, 4)
b = torch.randn(4, 5)
c1 = torch.matmul(a, b)
c2 = torch.bmm(a, b.unsqueeze(0).expand(2,-1,-1))
print(c1.shape == c2.shape)
    
~~~

