# Python数据分析

## 目录
[TOC]

# 1 Jupyter Notebook环境配置

## 1.1  常用快捷键

+ 要更改模式（编辑、命令）：
  ```python
  Esc -将模式更改为命令模式
  Enter -将模式更改为编辑模式
  ```

+ 更改内容类型（代码或降价）[在命令模式下]：

  ```python
  m -更改为降价
  y -更改为代码
  ```

+ 执行代码或降价[任何模式]
  ```python
  Shift + Enter -执行并转到下一个单元格
  Ctrl + Enter -执行并位于同一单元格中
  ```
  可以独立运行某段代码，也可以 Run All(Cell中)
  
+ 插入单元格[在命令模式下]
  ```python
  a -在单元格上方创建单元格
  b -在单元格下方创建单元格
  ```

 + 剪切复制粘贴[在命令模式下]
   ```python
   x - 剪切可以粘贴任意次数的单元格
   c - 复制可以在任何地方粘贴任意次数的单元格
   v - 粘贴单元格
   ```



### 一些启示

+ 也可以在先前的单元格直接修改并运行，这也是一个Jupyter Notebook的优越性
+ Insert可以添加单元格，注意位置
+ 也可以加入Markdown或者Heading(几级标题)，便于分析与调试
+ 添加markdown单元格可以添加笔记和注释
+ Edit中有很多其他功能，例如merge和move



## Python基础

详见Pythonrev.md

# 2 Python数据分析之Numpy
NumPy是Python的一种开源的数值计算扩展。这种工具可用来存储和处理大型矩阵，比Python自身的嵌套列表(nested list structure)结构要高效的多。

NumPy(Numeric Python)提供了许多高级的数值编程工具。Numpy的一个重要特性是它的**数组计算**，是我们做数据分析必不可少的一个包。

## 2.1 导入Numpy
导入python库使用关键字import，后面可以自定义库的简称，但是一般都将Numpy命名为np，pandas命名为pd。
```python
import numpy
import numpy as np # 导入整个库并取别名
```

## 2.2 Numpy的数组对象及其索引
### 2.2.1 数组上的数学操作
如果我们想让列表中的每个元素+1，但列表并不支持a+1

用推导式：

```python
[x+1 for x in a]
```



如果想令数组相加，得到对应元素相加的结果，并不是我们所想要的，而是merge了

我们用推导式：

```python
[x+y for(x,y) in zip(a,b)]  #都需要利用到列表生成式
```

这样操作比较麻烦，而且数据量大的时候会很耗时。

但是我们有**Numpy**！

```python
a = np.array([1,2,3,4])
a

# array([1, 2, 3, 4])
```



```python
a+1

# array([2, 3, 4, 5])
```



```python
a*2

# array([2, 4, 6, 8])
```



### 2.2.2 产生数组

#### 2.2.2.1 从列表产生数组

```python
l = [0,1,2,3]
a = np.array(l)
a

# array([0, 1, 2, 3])
```



#### 2.2.2.2 从列表传入

```python
a = np.array([1,2,3,4])
a

# array([1, 2, 3, 4])
```



#### 2.2.2.3 生成全0数组：

```python
np.zeros(5) #括号内传个数，默认浮点数

# array([0., 0., 0., 0., 0.])
```



#### 2.2.2.4 生成全1数组：

```python
np.ones(5) #括号内传个数，默认浮点数

# array([1., 1., 1., 1., 1.])
```

```python
np.ones(5,dtype="bool") #可以自己指定类型，np.zeros函数同理

# array([ True,  True,  True,  True,  True])
```

与列表不同，**数组中要求所有元素的 dtype 是一样的**，如果传入参数的类型与数组类型不一样，需要按照已有的类型进行转换。

#### 2.2.2.5 可以使用 fill 方法将数组设为指定值

```python
a = np.array([1,2,3,4])
a.fill(5) #让数组中的每一个元素都等于5
a

# array([5, 5, 5, 5])
```

```python
a.fill(2.5) #自动进行取整
a

# array([2, 2, 2, 2])
```
```python
a = a.astype("float") #强制类型转换
a.fill(2.5)
a

# array([2.5, 2.5, 2.5, 2.5])
```



还可以使用一些特定的方法生成特殊的数组:

#### 2.2.2.6 生成特殊数组
```python
# 生成整数数列
a = np.arange(1,10) # 左闭右开区间，和range的使用方式同理
a
# array([1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3])

# 生成等差数列
a = np.linspace(1,10,21) #右边是包括在里面的，从a-b一共c个数的等差数列，其实np.arange也可以做,利用步长便可，但linspace的第三参数是元素个数
a
"""
array([1.  ,  1.45,  1.9 ,  2.35,  2.8 ,  3.25,  3.7 ,  4.15,  4.6 ,
         5.05,  5.5 ,  5.95,  6.4 ,  6.85,  7.3 ,  7.75,  8.2 ,  8.65,
         9.1 ,  9.55, 10.  ])
"""
        
# 生成随机数
np.random.rand(10)
"""
array([0.48272736, 0.00581325, 0.16110313, 0.52234425, 0.63905254,
       0.42691432, 0.37196789, 0.57188523, 0.46437865, 0.43126794])
"""

# 标准正态分布
np.random.randn(10) 
"""
array([-1.4237624 ,  1.63058904, -1.9223658 ,  0.17736421,  0.54337908,
       -1.46049834,  0.2146448 , -0.32785131, -1.08990638, -0.75152502])
"""

# 生成随机整数，从1-20中随机10个
np.random.randint(1,20,10)
"""
array([12,  6,  4,  3, 13, 19,  5,  4, 14, 16])
"""
```



### 2.2.3 数组属性

```python
a
"""
  array([ 1.  ,  1.45,  1.9 ,  2.35,  2.8 ,  3.25,  3.7 ,  4.15,  4.6 ,5.05,  5.5 ,  5.95,  6.4 ,  6.85,  7.3 ,  7.75,  8.2 ,  8.65,9.1 ,  9.55, 10.  ])
"""

#查看类型
type(a)
#numpy.ndarray

#查看数组中的数据类型：(data type)
a.dtype
#dtype('float64')

#查看形状，会返回一个元组，每个元素代表这一维的元素数目：
a.shape
#(4,)
#或者使用
np.shape(a)
#(4,)

#要看数组里面元素的个数：
a.size
#4

#查看数组维度：(n dimension)
a.ndim
#1
```



### 2.2.4 索引和切片

和列表相似，数组也支持索引和切片操作。

```python
#使用索引
a = np.array([0,1,2,3])
a[0]
#0

#修改元素
a[0] = 10
a
#array([10,  1,  2,  3])

#切片，支持负索引：
a = np.array([11,12,13,14,15])
a[1:3] #左闭右开，从0开始算
#array([12, 13])

a[1:-2] 
a[-4:3]
#等价于a[1:3]
array([12, 13])

#省略参数：
a[-2:] #从倒数第2个取到底
#array([14, 15])
a[::2] #从头取到尾，间隔2
#array([11,13,15])
#步长

```

假设我们记录一部电影的累计票房：
```python
ob = np.array([21000,21800,22240,23450,25000])
ob
#array([21000, 21800, 22240, 23450, 25000])

#可以这样计算每天的票房：利用错位相减
ob2 = ob[1:]-ob[:-1]
ob2
#array([ 800,  440, 1210, 1550])
```

### 2.2.5 多维数组及其属性
array还可以用来生成和处理多维数组：
```python
#生成多维数组
a = np.array([[0,1,2,3],[10,11,12,13]])
a
#array([[ 0,  1,  2,  3],[10, 11, 12, 13]])
#事实上我们传入的是一个以列表为元素的列表，最终得到一个二维数组。

#查看形状
a.shape
#(2, 4)
#2维，每维4元素

查看总的元素个数：
a.size
#8

查看维数：
a.ndim
#2
```

### 2.2.6 多维数组索引
对于二维数组，可以传入两个数字来索引：
```python
a
#array([[ 0,  1,  2,  3],[10, 11, 12, 13]])

a[1,3]
#13 
#其中，1是行索引，3是列索引，中间用逗号隔开。事实上，Python会将它们看成一个元组（1,3），然后按照顺序进行对应。

#可以利用索引给它赋值：
a[1,3] = -1
a
#array([[ 0,  1,  2,  3],[10, 11, 12, -1]])

#事实上，我们还可以使用单个索引来索引一整行内容：与C有点类似了
a[1]
#array([10, 11, 12, -1])

#Python会将这单个元组当成对第一维的索引，然后返回对应的内容。
#记住：行前列后
a[:,1]
#array([ 1, 11])
```

### 2.2.7 多维数组切片
多维数组，也支持切片操作：
```python
a = np.array([[0,1,2,3,4,5],[10,11,12,13,14,15],[20,21,22,23,24,25],[30,31,32,33,34,35],[40,41,42,43,44,45],[50,51,52,53,54,55]])
a
"""
array([[ 0,  1,  2,  3,  4,  5],
[10, 11, 12, 13, 14, 15],
[20, 21, 22, 23, 24, 25],
[30, 31, 32, 33, 34, 35],
[40, 41, 42, 43, 44, 45],
[50, 51, 52, 53, 54, 55]])
"""

#想得到第一行的第4和第5两个元素：
a[0,3:5]
#array([3, 4])

#得到最后两行的最后两列：
a[4:,4:]
#array([[44, 45],[54, 55]])

#得到第三列：
a[:,2]
#array([ 2, 12, 22, 32, 42, 52])

#每一维都支持切片的规则，包括负索引，省略
#[lower:upper:step]

例如，取出3,5行的奇数列：
a[2::2,::2]
#array([[20, 22, 24],[40, 42, 44]])

```

### 2.2.8 切片是引用
切片在内存中使用的是引用机制
引用机制意味着，Python并没有为b分配新的空间来存储它的值，而是让b指向了a所分配的内存空间：
```python
a = np.array([0,1,2,3,4])
b = a[2:4]
print(b)
#[2 3]

#因此，改变b会改变a的值：
b[0] = 10
a
#array([ 0,  1, 10,  3,  4])

#而这种现象在列表中并不会出现：
a = [1,2,3,4,5]
b = a[2:4]
b[0] = 10
print(a)
#[1, 2, 3, 4, 5]

"""
这样做的好处在于，对于很大的数组，不用大量复制多余的值，节约了空间。
缺点在于，可能出现改变一个值改变另一个值的情况。
"""

#一个解决方法是使用copy()方法产生一个复制，这个复制会申请新的内存：
a = np.array([0,1,2,3,4])
b = a[2:4].copy()
b[0] = 10
a
#array([0, 1, 2, 3, 4])
```

### 2.2.9 花式索引
切片只能支持连续或者等间隔的切片操作，要想实现任意位置的操作。需要使用花式索引（fancy slicing）

#### 2.2.9.1 一维花式索引
与range函数类似，我们可以使用arange函数来产生等差数组。
```python
#产生等差数组
a = np.arange(0,100,10)
a
#array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

#花式索引需要指定索引位置：
index = [1,2,-3]
y = a[index]
print(y)
#[10 20 70]

#还可以使用布尔数组来花式索引：
mask = np.array([0,2,2,0,0,1,0,0,1,0],dtype = bool)
mask
"""
array([False,  True,  True, False, False,  True, False, False,  True, False])
"""

#mask必须是布尔数组，长度必须和数组长度相等。
a[mask]
#array([10, 20, 50, 80])
```

#### 2.2.9.2 二维花式索引
对于二维花式索引，我们需要给定行和列的值：
```python
a = np.array([[0,1,2,3,4,5],[10,11,12,13,14,15],[20,21,22,23,24,25],[30,31,32,33,34,35],[40,41,42,43,44,45],[50,51,52,53,54,55]])
a
"""
 array([[ 0,  1,  2,  3,  4,  5],
           [10, 11, 12, 13, 14, 15],
           [20, 21, 22, 23, 24, 25],
           [30, 31, 32, 33, 34, 35],
           [40, 41, 42, 43, 44, 45],
           [50, 51, 52, 53, 54, 55]])
"""
#返回的是一条次对角线上的5个值。

a[(0,1,2,3,4),(1,2,3,4,5)]
#array([ 1, 12, 23, 34, 45])
#返回的是最后三行的1,3,5列。

a[3:,[0,2,4]]
"""
array([[30, 32, 34],
           [40, 42, 44],
           [50, 52, 54]])
"""

#也可以使用mask进行索引：
mask = np.array([1,0,1,0,0,1],dtype = bool)
a[mask,2]
#array([ 2, 22, 52])
#与切片不同，花式索引返回的是原对象的一个复制而不是引用。
```

### 2.2.10 “不完全”索引 

只给定行索引的时候，返回整行：
```python
y = a[:3]
y
"""
array([[ 0,  1,  2,  3,  4,  5],
           [ 1,  1,  1,  1,  1,  1],
           [20, 21, 22, 23, 24, 25]])
"""

#这时候也可以使用花式索引取出第2,3,5行：
con = np.array([0,1,1,0,1,0],dtype = bool)
a[con]
"""
array([[ 1,  1,  1,  1,  1,  1],
           [20, 21, 22, 23, 24, 25],
           [40, 41, 42, 43, 44, 45]])
"""
```

### 2.2.11 where语句
```python
where(array)
#where函数会返回所有非零元素的索引。
```

#### 2.2.11.1 一维数组
```python
#先看一维的例子：
a = np.array([0,12,5,20])

#判断数组中的元素是不是大于10：
a>10
#array([False,  True, False,  True])

#数组中所有大于10的元素的索引位置：
np.where(a>10)
#(array([1, 3], dtype=int64),)
#注意到where的返回值是一个元组。返回的是索引位置，索引[1,3]大于10的数

#也可以直接用数组操作:
a[a>10]
#array([12, 20])

a[np.where(a>10)]
#array([12,20])
```

## 2.3 数组类型
### 2.3.1 具体如下：

|**基本类型**|**可用的Numpy类型**|**备注**|
|-:|-:|-:|
|布尔型|bool|占一个字节|
|整型|int8,int16,int32,int64,int128,int|int跟C语言中long一样大|
|无符号整型|uint8,uint16,uint32,uint64,uint128,uint|uint跟C语言中的unsigned long一样大|
|浮点数|float16,float32,float|默认为双精度float64，longfloat精度大小与系统有关|
|复数|complex64,complex128,complex,longcomplex|默认为complex128,即实部虚部都为双精度|
|字符串|string,unicode|可以使用dtype=S4表示一个4字节字符串的数组|
|对象|object|数组中可以使用任意值|
|时间|datetime64,timedelta64||

### 2.3.2 类型转换
trans.py
```python
a = np.array([1.5,-3],dtype = float)
a
#array([ 1.5, -3. ])

```

### 2.3.3 asarray 函数
asarray.py
```python
a = np.array([1,2,3])
np.asarray(a,dtype = float)#强制转换为浮点型
#array([1., 2., 3.])
```

### 2.3.4 astype 方法
astype.py
```python
a = np.array([1,2,3])
a.astype(float)
#array([1., 2., 3.])
```
此处a本身并未发生变化，因为这是一个拷贝的形式

## 2.4 数组操作
以豆瓣10部高分电影为例 
movies.py
```python
#电影名称
mv_name = ["肖申克的救赎","控方证人","美丽人生","阿甘正传","霸王别姬","泰坦尼克号","辛德勒的名单","这个杀手不太冷","疯狂动物城","海豚湾"]

#评分人数
mv_num = np.array([692795,42995,327855,580897,478523,157074,306904,662552,284652,159302])

#评分
mv_score = np.array([9.6,9.5,9.5,9.4,9.4,9.4,9.4,9.3,9.3,9.3])

#电影时长（分钟）
mv_length = np.array([142,116,116,142,171,194,195,133,109,92])

```

### 2.4.1 数组排序
#### 2.4.1.1 sort函数
sort.py
```python
np.sort(mv_num)#sort不改变原来数组
```

#### 2.4.1.2 argsort函数
argsort返回从小到大的排列在数组中的索引位置：
argsort.py
```python
order = np.argsort(mv_num)
order
#array([1, 5, 9, 8, 6, 2, 4, 3, 7, 0], dtype=int64)

```

### 2.4.2 数组求和（sum函数/方法）
sum.py
```python
np.sum(mv_num)

mv_num.sum()
```

### 2.4.3 求最值（max/min函数/方法）
max_min.py
```python
np.max(mv_length)

np.min(mv_score)

mv_length.max()

mv_score.min()
```

### 2.4.4 求均值和标准差（mean/std函数/方法）
mean_std.py
```python
np.mean(mv_length)

mv_length.mean()

np.std(mv_length)

mv_length.std()
```

### 2.4.5 相关系数矩阵（cov函数）
cov.py
```python
np.cov(mv_score,mv_length)
```

### 2.4.6 矩阵加减乘除

**矩阵和矩阵(向量)相乘：` (M行, N列)*(N行, L列) = (M行, L列)`**

```python
# 矩阵加法
np.add(A, B)

# 矩阵减法
np.subtract(A, B)

# 矩阵乘法
# 在进行矩阵乘法时，前一个矩阵的列数必须等于后一个矩阵的行数，才能进行乘法运算
# (M行, N列)*(N行, L列) = (M行, L列)
p.dot(A, B)

# 矩阵除法
np.divide(A, B)
```
```python
import numpy as np

# 创建矩阵A和B
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B:")
print(np.add(A, B))
[1, 2]  	[5, 6]		[6, 8]
		+  			=  
[3, 4]		[7, 8]		[10, 12]


print("A - B:")
print(np.subtract(A, B))
[1, 2]  	[5, 6]		[-4, -4]
		-  			=  
[3, 4]		[7, 8]		[-4, -4]

print("A * B:")
print(np.dot(A, B))
[1, 2]  	[5, 6]		[19, 22]
		*  			=  
[3, 4]		[7, 8]		 [43, 50]

print("A / B:")
print(np.divide(A, B))
[1, 2]  	[5, 6]		[0.2,  0.33333333]
		/  			=  
[3, 4]		[7, 8]		[0.42857143, 0.5]

```

### 2.4.7 矩阵向量乘法
m×n 的矩阵乘以 n×1 的向量，得到的是 m×1 的向量
```python
[1, 2]  	[1]			[19]
		*  			=  
[3, 4]		[1]			 [43]

```

### 2.4.8 矩阵求逆
使用numpy.linalg.inv()函数进行矩阵求逆操作
```python
import numpy as np

# 创建矩阵
matrix = np.array([[1, 2], [3, 4]])

# 求逆矩阵
result = np.linalg.inv(matrix)

print(result)
```

### 2.4.9 矩阵求迹
使用numpy.trace()函数可以计算矩阵的迹
```python
import numpy as np

# 创建矩阵
matrix = np.array([[1, 2], [3, 4]])

# 计算矩阵的迹
result = np.trace(matrix)

print(result)

```

### 2.4.10 向量点积
使用numpy.dot()函数进行向量点积操作
```python
import numpy as np

# 创建两个向量
vector1 = np.array([1, 2])
vector2 = np.array([3, 4])

# 向量点积
result = np.dot(vector1, vector2)

print(result)

```

### 2.4.11 向量范数
使用numpy.linalg.norm()函数可以计算向量的范数
```python
import numpy as np

# 创建向量
vector = np.array([1, 2, 3])

# 计算向量的L2范数
result = np.linalg.norm(vector)

print(result)

```

### 2.4.12 矩阵存取

### 2.4.13 解线性方程组

```python
import numpy as np

a = np.array([[2, 3], [4, 5]])
b = np.array([5, 6])

x = np.linalg.solve(a, b)

print(x) # [-4.  5.]

```

### 2.4.14 矩阵广播机制 ？

**广播机制**是Numpy中的一个重要特性，是指**对ndarray执行某些数值计算时（这里是指矩阵间的数值计算，对应位置元素1对1执行标量运算，而非线性代数中的矩阵间运算），可以确保在数组间形状不完全相同时可以自动的通过广播机制扩散到相同形状，进而执行相应的计算功能**。

当然，这里的广播机制是有条件的，而非对任意形状不同的数组都能完成自动广播，显然，理解这里的"条件"是理解广播机制的核心原理。

维度相等时相当于**无需广播**，所以严格的说广播仅适用于某一维度从1广播到N；如果当前维度满足广播要求，则同时前移一个维度继续比较，直至首先完成其中一个矩阵的所有维度——另一矩阵如果还有剩余的话，其实也无所谓了，这很好理解。

## 2.5 数组

### 2.5.1 数组形状

ndim_shape.py
```python
a = np.arange(6)
a
#array([0, 1, 2, 3, 4, 5])

a.shape=(2,3)
a
#array([[0, 1, 2],[3, 4, 5]])

a.shape
#(2, 3)

#与之对应的方法是reshape，但它不会修改原来数组的值，而是返回一个新的数组：
a = np.arange(6)
a.reshape(2,3)
#array([[0, 1, 2],[3, 4, 5]])

```

### 2.5.2 数组转置(T/transpose)
tp.py
```python
a = a.reshape(2,3)
a
#array([[0, 1, 2],[3, 4, 5]])

a.T
array([[0, 3],[1, 4],[2, 5]])

a.transpose() #只要没赋值给本身，a的数值不会变换
array([[0, 3],[1, 4],[2, 5]])

```

### 2.5.3 数组连接
有时候我们需要将不同的数组按照一定的顺序连接起来：  
concatenate((a0,a1,...,aN),axis = 0)

**注意，这些数组要用()包括到一个元组中去。除了给定的轴外，这些数组其他轴的长度必须是一样的。**
connect.py
```python
x = np.array([[0,1,2],[10,11,12]])
y = np.array([[50,51,52],[60,61,62]])
print(x.shape)
print(y.shape)

"""
(2, 3)
(2, 3)
"""

#默认沿着第一维进行连接：
z = np.concatenate((x,y))
z
"""
 array([[ 0,  1,  2],
           [10, 11, 12],
           [50, 51, 52],
           [60, 61, 62]])
"""

#沿着第二维进行连接：
z = np.concatenate((x,y),axis = 1)
z

"""
 array([[ 0,  1,  2, 50, 51, 52],
[10, 11, 12, 60, 61, 62]])
"""

#注意到这里x和y的形状是一样的，还可以将它们连接成三维的数组，但concatenate不能提供这样的功能，不过可以这样：
z = np.array((x,y))
z
"""
array([[[ 0,  1,  2],
[10, 11, 12]],  
[[50, 51, 52],
[60, 61, 62]]])
"""
```



事实上，Numpy提供了分别对应这三种情况的函数：
* vstack
* hstack
* dstack

stack.py

```python
np.vstack((x,y))

np.dstack((x,y))
```

**NumPy的线性代数模块（numpy.linalg）提供了许多矩阵运算函数，如矩阵乘法、求逆、行列式、特征值等，该库包含了线性代数所需的所有功能。**

np.dot(a, b)：                 两个数组的点积，即元素对应相乘
np.matmul(a, b)                两个数组的矩阵积
np.linalg.inv(a)               计算矩阵的逆
np.linalg.det(a)               计算矩阵的行列式
np.linalg.eig(a)               计算矩阵的特征值和特征向量
np.linalg.solve(a, b)          解线性方程组 ax=b



## 2.6 Numpy内置函数

```python
a = np.array([-1,2,3,-2])

np.abs(a) #绝对值

np.exp(a) #指数

np.median(a) #中值

np.cumsum(a) #累积和
```

不要死记，查资料：

https://blog.csdn.net/nihaoxiaocui/article/details/51992860?locationNum=5&fps=1



## 2.7 数组属性方法总结

|            **调用方法** |                                         **作用** |
| ----------------------: | -----------------------------------------------: |
|                   **1** |                                     **基本属性** |
|                 a.dtype |                    数组元素类型float32,uint8,... |
|                 a.shape |                              数组形状(m,n,o,...) |
|                  a.size |                                       数组元素数 |
|              a.itemsize |                                 每个元素占字节数 |
|                a.nbytes |                                 所有元素占的字节 |
|                  a.ndim |                                         数组维度 |
|                       - |                                                - |
|                   **2** |                                     **形状相关** |
|                  a.flat |                                 所有元素的迭代器 |
|             a.flatten() |                            返回一个1维数组的复制 |
|               a.ravel() |                           返回一个一维数组，高效 |
|      a.resize(new_size) |                                         改变形状 |
| a.swapaxes(axis1,axis2) |                               交换两个维度的位置 |
|     a.transpose(* axex) |                               交换所有维度的位置 |
|                     a.T |                              转置，a.transpose() |
|             a.squeeze() |                            去除所有长度为1的维度 |
|                       - |                                                - |
|                   **3** |                                     **填充复制** |
|                a.copy() |                               返回数组的一个复制 |
|           a.fill(value) |                         将数组的元组设置为特定值 |
|                       - |                                                - |
|                   **4** |                                         **转化** |
|              a.tolist() |                                 将数组转化为列表 |
|            a.tostring() |                                     转换为字符串 |
|         a.astype(dtype) |                                   转换为指定类型 |
|       a.byteswap(False) |                                   转换大小字节序 |
|   a.view(type_or_dtype) | 生成一个使用相同内存，但使用不同的表示方法的数组 |
|                       - |                                                - |
|                   **5** |                                     **查找排序** |
|             a.nonzero() |                           返回所有非零元素的索引 |
|         a.sort(axis=-1) |                                     沿某个轴排序 |
|      a.argsort(axis=-1) |                       沿某个轴，返回按排序的索引 |
|       a.searchsorted(b) |           返回将b中元素插入a后能保持有序的索引值 |
|                       - |                                                - |
|                   **6** |                                 **元素数学操作** |
|        a.clip(low,high) |                           将数值限制在一定范围内 |
|     a.round(decimals=0) |                                   近似到指定精度 |
|     a.cumsum(axis=None) |                                           累加和 |
|    a.cumprod(axis=None) |                                           累乘积 |
|                       - |                                                - |
|                   **7** |                                     **约简操作** |
|        a.sum(axis=None) |                                             求和 |
|       a.prod(axis=None) |                                             求积 |
|        a.min(axis=None) |                                           最小值 |
|        a.max(axis=None) |                                           最大值 |
|     a.argmin(axis=None) |                                       最小值索引 |
|     a.argmax(axis=None) |                                       最大值索引 |
|        a.ptp(axis=None) |                                   最大值减最小值 |
|       a.mean(axis=None) |                                           平均值 |
|        a.std(axis=None) |                                           标准差 |
|        a.var(axis=None) |                                             方差 |
|        a.any(axis=None) |                  只要有一个不为0，返回真，逻辑或 |
|        a.all(axis=None) |                      所有都不为0，返回真，逻辑与 |



# 3 Python数据分析之Pandas 

## 3.1 Panads基本介绍

Python Data Analysis Library 或 Pandas是基于Numpy的一种工具，该工具是为了解决数据分析任务而创建的。Pandas纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集所需的工具。pandas提供了大量能使我们快速便捷地处理数据的函数和方法。

```python
import pandas as pd
import numpy as np
```

### 3.2 Pandas 基本数据结构 

``pandas``有两种常用的基本结构： 

+ ``Series``
    + **一维数组**，与Numpy中的一维**array**类似。二者与Python基本的数据结构**List**也很接近。Series**能保存不同种数据类型**，字符串、boolean值、数字等都能保存在Series中。
+ ``DataFrame``
    + **二维的表格型数据结构**。很多功能与R中的**data.frame**类似。**可以将DataFrame理解为Series的容器。**以下的内容主要以DataFrame为主。

### 3.2.1 Pandas库的Series类型 
一维``Series``可以用一维列表初始化：
pd_Series.py

```python
s = pd.Series([1,3,5,np.nan,6,8])#index = ['a','b','c','d','x','y'])设置索引（替换原来的0，1，2，3），np.nan设置空值
print(s)

"""
    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64
默认情况下，``Series``的下标都是数字（可以使用额外参数指定），类型是统一的。
"""


#索引：数据的行标签
s.index #从0到6（不含），1为步长

#RangeIndex(start=0, stop=6, step=1)
#这里也是左闭右开


#值
s.values
#array([ 1.,  3.,  5., nan,  6.,  8.])

s[3]
#nan


#切片操作
s[2:5] #左闭右开
"""
    2    5.0
    3    NaN
    4    6.0
    dtype: float64
"""

s[::2]
"""
	0    1.0
    2    5.0
    4    6.0
    dtype: float64
"""


#索引赋值
s.index.name = '索引'
s
"""
	索引 #在此添加
    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64
"""

#把索引号换成abcdef
s.index = list('abcdef')
s
"""
    a    1.0
    b    3.0
    c    5.0
    d    NaN
    e    6.0
    f    8.0
    dtype: float64
"""

#依据自己定义的数据类型进行切片，不是左闭右开了
s['a':'c':2] 
"""
    a    1.0
    c    5.0
    dtype: float64
"""
```

### 3.2.2 Pandas库的Dataframe类型

#### 3.2.2.1 创建结构

``DataFrame``则是个二维结构，这里首先构造一组**时间序列**，作为我们第一维的下标：
pd_Dataframe.py

```python
date = pd.date_range("20180101", periods = 6)#持续6次
print(date)
"""
 DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04','2018-01-05', '2018-01-06'], dtype='datetime64[ns]', freq='D')
 """


#然后创建一个``DataFrame``结构：
df = pd.DataFrame(np.random.randn(6,4), index = date, columns = list("ABCD"))#结合numpy中的随机数（六行四列）
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>0.391943</td>
      <td>-1.252843</td>
      <td>-0.247644</td>
      <td>-0.320195</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>0.845487</td>
      <td>0.208064</td>
      <td>-0.069838</td>
      <td>0.137163</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>0.776754</td>
      <td>-2.215175</td>
      <td>-1.116371</td>
      <td>1.763836</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>0.016040</td>
      <td>2.006192</td>
      <td>0.227209</td>
      <td>1.783695</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>-0.006219</td>
      <td>0.592141</td>
      <td>0.462352</td>
      <td>0.993924</td>
    </tr>
    <tr>
      <th>2018-01-06</th>
      <td>1.112720</td>
      <td>-0.223669</td>
      <td>0.084223</td>
      <td>-0.550868</td>
    </tr>
  </tbody>
</table>


默认情况下，如果不指定``index``参数和``columns``，那么它们的值将从用0开始的数字替代。

```python
df = pd.DataFrame(np.random.randn(6,4))
df
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.284776</td>
      <td>0.568122</td>
      <td>-2.376747</td>
      <td>1.146297</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.534887</td>
      <td>0.142495</td>
      <td>0.628169</td>
      <td>-1.991141</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.569047</td>
      <td>0.255749</td>
      <td>0.139962</td>
      <td>-0.551621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.334796</td>
      <td>1.025724</td>
      <td>-1.231977</td>
      <td>-0.656463</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.113152</td>
      <td>0.917217</td>
      <td>-0.747063</td>
      <td>-0.686428</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.078839</td>
      <td>-0.668506</td>
      <td>0.099214</td>
      <td>-0.038008</td>
    </tr>
  </tbody>
</table>




除了向``DataFrame``中传入二维数组，我们也可以使用字典传入数据：

```python
df2 = pd.DataFrame({'A':1.,'B':pd.Timestamp("20181001"),'C':pd.Series(1,index = list(range(4)),dtype = float),'D':np.array([3]*4, dtype = int),'E':pd.Categorical(["test","train","test","train"]),'F':"abc"}) #B:时间戳,E分类类型
df2
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>abc</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2018-10-01</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>abc</td>
    </tr>
  </tbody>
</table>


```python
df2.dtypes #查看各个列的数据类型
```

​    A           float64
​    B    datetime64[ns]
​    C           float64
​    D             int32
​    E          category
​    F            object
​    dtype: object

字典的每个``key``代表一列，其``value``可以是各种能够转化为``Series``的对象。

与``Series``要求**所有的类型都一致**不同，``DataFrame``**只要求每一列**数据的格式相同。

#### 3.2.2.2 查看数据
头尾数据：
``head``和``tail``方法可以**分别查看最前面几行和最后面几行的数据（默认为5）**：

```python
df.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.284776</td>
      <td>0.568122</td>
      <td>-2.376747</td>
      <td>1.146297</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.534887</td>
      <td>0.142495</td>
      <td>0.628169</td>
      <td>-1.991141</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.569047</td>
      <td>0.255749</td>
      <td>0.139962</td>
      <td>-0.551621</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.334796</td>
      <td>1.025724</td>
      <td>-1.231977</td>
      <td>-0.656463</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.113152</td>
      <td>0.917217</td>
      <td>-0.747063</td>
      <td>-0.686428</td>
    </tr>
  </tbody>
</table>


最后3行：
```python
df.tail(3)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.334796</td>
      <td>1.025724</td>
      <td>-1.231977</td>
      <td>-0.656463</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.113152</td>
      <td>0.917217</td>
      <td>-0.747063</td>
      <td>-0.686428</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-2.078839</td>
      <td>-0.668506</td>
      <td>0.099214</td>
      <td>-0.038008</td>
    </tr>
  </tbody>
</table>


#### 3.2.2.3 下标index，列标columns，数据values
pd_sign.py
```python
#下标使用``index``属性查看：
df.index
"""
DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04','2018-01-05', '2018-01-06'],dtype='datetime64[ns]', freq='D')
"""

#列标使用``columns``属性查看：
df.columns
"""
Index(['A', 'B', 'C', 'D'], dtype='object')
"""

#数据值使用``values``查看：
df.values
"""
array([[ 0.39194344, -1.25284255, -0.24764423, -0.32019526],[ 0.84548738,  0.20806449, -0.06983781,  0.13716277],[ 0.7767544 , -2.21517465, -1.11637102,  1.76383631],[ 0.01603994,  2.00619213,  0.22720908,  1.78369472],[-0.00621932,  0.59214148,  0.46235154,  0.99392424],[ 1.11272049, -0.22366925,  0.08422338, -0.5508679 ]])
"""
```

## 3.3 Pandas读取数据及数据操作 
我们将以豆瓣的电影数据作为我们深入了解Pandas的一个示例。
```python
df = pd.read_excel(r"C:\Users\Lovetianyi\Desktop\python\作业3\豆瓣电影数据.xlsx",index_col = 0) 
#csv:read_csv;绝对路径或相对路径默认在当前文件夹下。r告诉编译器不需要转义
#具体其它参数可以去查帮助文档 ?pd.read_excel
```

```python
df.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


### 3.3.1 行操作

```python
df.iloc[0]
```
    名字                   肖申克的救赎
    投票人数                 692795
    类型                    剧情/犯罪
    产地                       美国
    上映时间    1994-09-10 00:00:00
    时长                      142
    年代                     1994
    评分                      9.6
    首映地点                 多伦多电影节
    Name: 0, dtype: object

```python
df.iloc[0:5] #左闭右开
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>

也可以使用loc

```python
df.loc[0:5] #不是左闭右开
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>


#### 3.3.1.1 添加一行 append

```python
dit = {"名字":"复仇者联盟3","投票人数":123456,"类型":"剧情/科幻","产地":"美国","上映时间":"2018-05-04 00:00:00","时长":142,"年代":2018,"评分":np.nan,"首映地点":"美国"}
s = pd.Series(dit)#进行字典数据的传入
s.name = 38738
s
```

    名字                   复仇者联盟3
    投票人数                 123456
    类型                    剧情/科幻
    产地                       美国
    上映时间    2018-05-04 00:00:00
    时长                      142
    年代                     2018
    评分                      8.7
    首映地点                     美国
    Name: 38738, dtype: object

```python
df = df.append(s) #覆盖掉原来的数据重新进行赋值
df[-5:]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38734</th>
      <td>1935年</td>
      <td>57.0</td>
      <td>喜剧/歌舞</td>
      <td>美国</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>1935</td>
      <td>7.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38735</th>
      <td>血溅画屏</td>
      <td>95.0</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>1986</td>
      <td>7.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38736</th>
      <td>魔窟中的幻想</td>
      <td>51.0</td>
      <td>惊悚/恐怖/儿童</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>1986</td>
      <td>8.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38737</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32.0</td>
      <td>剧情/战争</td>
      <td>苏联</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>1977</td>
      <td>6.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38738</th>
      <td>复仇者联盟3</td>
      <td>123456.0</td>
      <td>剧情/科幻</td>
      <td>美国</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>2018</td>
      <td>NaN</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
</div>

#### 3.3.1.2 删除一行 drop

```python
df = df.drop([38738])
df[-5:]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38733</th>
      <td>神学院 S</td>
      <td>46.0</td>
      <td>Adult</td>
      <td>法国</td>
      <td>1905-06-05 00:00:00</td>
      <td>58</td>
      <td>1983</td>
      <td>8.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38734</th>
      <td>1935年</td>
      <td>57.0</td>
      <td>喜剧/歌舞</td>
      <td>美国</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>1935</td>
      <td>7.6</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38735</th>
      <td>血溅画屏</td>
      <td>95.0</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>1986</td>
      <td>7.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38736</th>
      <td>魔窟中的幻想</td>
      <td>51.0</td>
      <td>惊悚/恐怖/儿童</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>1986</td>
      <td>8.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38737</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32.0</td>
      <td>剧情/战争</td>
      <td>苏联</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>1977</td>
      <td>6.6</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


### 3.3.2 列操作

```python
df.columns
```

    Index(['名字', '投票人数', '类型', '产地', '上映时间', '时长', '年代', '评分', '首映地点'], dtype='object')

```python
df["名字"][:5] #后面中括号表示只想看到的行数，下同
```
    0    肖申克的救赎
    1      控方证人
    2     美丽人生 
    3      阿甘正传
    4      霸王别姬
    Name: 名字, dtype: object

```python
df[["名字","类型"]][:5]#选取列 ？多重[]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>类型</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>剧情/犯罪</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>剧情/悬疑/犯罪</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>剧情/喜剧/爱情</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>剧情/爱情</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>剧情/爱情/同性</td>
    </tr>
  </tbody>
</table>


#### 3.3.2.1 增加一列 直接赋值

直接赋值

```python
df["序号"] = range(1,len(df)+1) #生成序号的基本方式 len(df)+1表示加一行以后的长度
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>序号</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
      <td>5</td>
    </tr>
  </tbody>
</table>


#### 3.3.2.2 删除一列 drop axis

```python
df = df.drop("序号",axis = 1) #axis指定方向，0为行1为列，默认为0
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


#### 3.3.2.3 通过标签选择数据
``df.loc[[index],[colunm]]``通过标签选择数据

```python
df.loc[1,"名字"]
```

    '控方证人'

```python
df.loc[[1,3,5,7,9],["名字","评分"]] #多行跳行多列跳列选择
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>新世纪福音战士剧场版：Air/真心为你 新世紀エヴァンゲリオン劇場版 Ai</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>这个杀手不太冷</td>
      <td>9.4</td>
    </tr>
  </tbody>
</table>


### 3.3.3 条件选择
#### 3.3.3.1选取产地为美国的所有电影 

```python
df[df["产地"] == "美国"][:5] #内部为bool，需要再加上df[]来输出表格
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>6</th>
      <td>辛德勒的名单</td>
      <td>306904.0</td>
      <td>剧情/历史/战争</td>
      <td>美国</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>1993</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
    </tr>
  </tbody>
</table>


#### 3.3.3.2 选取产地为美国的所有电影，并且评分大于9分的电影 &

```python
df[(df.产地 == "美国") & (df.评分 > 9)][:5] #df.标签:更简洁的写法
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>6</th>
      <td>辛德勒的名单</td>
      <td>306904.0</td>
      <td>剧情/历史/战争</td>
      <td>美国</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>1993</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
    </tr>
  </tbody>
</table>


#### 3.3.3.3 选取产地为美国或中国大陆的所有电影，并且评分大于9分 & |

```python
df[((df.产地 == "美国") | (df.产地 == "中国大陆")) & (df.评分 > 9)][:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074.0</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>


## 3.4 缺失值及异常值处理 
### 3.4.1 缺失值处理方法：
|**方法**|**说明**|
|-:|-:|
|**dropna**|根据标签中的缺失值进行过滤，删除缺失值|
|**fillna**|对缺失值进行填充|
|**isnull**|返回一个布尔值对象，判断哪些值是缺失值|
|**notnull**|isnull的否定式|

### 3.4.2 判断缺失值
```python
df[df["名字"].isnull()][:10]#内部是布尔值，加上df[]就显示所有缺失的
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>231</th>
      <td>NaN</td>
      <td>144.0</td>
      <td>纪录片/音乐</td>
      <td>韩国</td>
      <td>2011-02-02 00:00:00</td>
      <td>90</td>
      <td>2011</td>
      <td>9.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>361</th>
      <td>NaN</td>
      <td>80.0</td>
      <td>短片</td>
      <td>其他</td>
      <td>1905-05-17 00:00:00</td>
      <td>4</td>
      <td>1964</td>
      <td>5.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>369</th>
      <td>NaN</td>
      <td>5315.0</td>
      <td>剧情</td>
      <td>日本</td>
      <td>2004-07-10 00:00:00</td>
      <td>111</td>
      <td>2004</td>
      <td>7.5</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>372</th>
      <td>NaN</td>
      <td>263.0</td>
      <td>短片/音乐</td>
      <td>英国</td>
      <td>1998-06-30 00:00:00</td>
      <td>34</td>
      <td>1998</td>
      <td>9.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>374</th>
      <td>NaN</td>
      <td>47.0</td>
      <td>短片</td>
      <td>其他</td>
      <td>1905-05-17 00:00:00</td>
      <td>3</td>
      <td>1964</td>
      <td>6.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>375</th>
      <td>NaN</td>
      <td>1193.0</td>
      <td>短片/音乐</td>
      <td>法国</td>
      <td>1905-07-01 00:00:00</td>
      <td>10</td>
      <td>2010</td>
      <td>7.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>411</th>
      <td>NaN</td>
      <td>32.0</td>
      <td>短片</td>
      <td>其他</td>
      <td>1905-05-17 00:00:00</td>
      <td>3</td>
      <td>1964</td>
      <td>7.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>432</th>
      <td>NaN</td>
      <td>1081.0</td>
      <td>剧情/动作/惊悚/犯罪</td>
      <td>美国</td>
      <td>2016-02-26 00:00:00</td>
      <td>115</td>
      <td>2016</td>
      <td>6.0</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>441</th>
      <td>NaN</td>
      <td>213.0</td>
      <td>恐怖</td>
      <td>美国</td>
      <td>2007-03-06 00:00:00</td>
      <td>83</td>
      <td>2007</td>
      <td>3.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>448</th>
      <td>NaN</td>
      <td>110.0</td>
      <td>纪录片</td>
      <td>荷兰</td>
      <td>2002-04-19 00:00:00</td>
      <td>48</td>
      <td>2000</td>
      <td>9.3</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df[df["名字"].notnull()][:5]#非缺失
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795.0</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995.0</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855.0</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897.0</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523.0</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


### 3.4.3 填充缺失值
```python
df[df["评分"].isnull()][:10] #注意这里特地将前面加入的复仇者联盟令其评分缺失来举例
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38738</th>
      <td>复仇者联盟3</td>
      <td>123456.0</td>
      <td>剧情/科幻</td>
      <td>美国</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>2018</td>
      <td>NaN</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df["评分"].fillna(np.mean(df["评分"]), inplace = True) #使用均值来进行替代，inplace意为直接在原始数据中进行修改
df[-5:]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38734</th>
      <td>1935年</td>
      <td>57.0</td>
      <td>喜剧/歌舞</td>
      <td>美国</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>1935</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38735</th>
      <td>血溅画屏</td>
      <td>95.0</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>1986</td>
      <td>7.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38736</th>
      <td>魔窟中的幻想</td>
      <td>51.0</td>
      <td>惊悚/恐怖/儿童</td>
      <td>中国大陆</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>1986</td>
      <td>8.000000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38737</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32.0</td>
      <td>剧情/战争</td>
      <td>苏联</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>1977</td>
      <td>6.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38738</th>
      <td>复仇者联盟3</td>
      <td>123456.0</td>
      <td>剧情/科幻</td>
      <td>美国</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>2018</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df1 = df.fillna("未知电影") #谨慎使用，除非确定所有的空值都是在一列中，否则所有的空值都会填成这个
#不可采用df["名字"].fillna("未知电影")的形式，因为填写后数据格式就变了，变成Series了
```

```python
df1[df1["名字"].isnull()][:10]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>


### 3.4.4 删除缺失值
```python
df.dropna() 参数

how = 'all':删除全为空值的行或列
inplace = True: 覆盖之前的数据
axis = 0: 选择行或列，默认是行

len(df)
#38739

df2 = df.dropna()
len(df2)
#38176

df.dropna(inplace = True)
len(df) #inplace覆盖掉原来的值
#38176
```

### 3.4.5 处理异常值

异常值，即在数据集中存在不合理的值，又称离群点。比如年龄为-1，笔记本电脑重量为1吨等，都属于异常值的范围。

```python
df[df["投票人数"] < 0] #直接删除，或者找原始数据来修正都行
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19777</th>
      <td>皇家大贼 皇家大</td>
      <td>-80.0</td>
      <td>剧情/犯罪</td>
      <td>中国香港</td>
      <td>1985-05-31 00:00:00</td>
      <td>60</td>
      <td>1985</td>
      <td>6.3</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>19786</th>
      <td>日本的垃圾去中国大陆 にっぽんの“ゴミ” 大陆へ渡る ～中国式リサイクル錬</td>
      <td>-80.0</td>
      <td>纪录片</td>
      <td>日本</td>
      <td>1905-06-26 00:00:00</td>
      <td>60</td>
      <td>2004</td>
      <td>7.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>19797</th>
      <td>女教徒</td>
      <td>-118.0</td>
      <td>剧情</td>
      <td>法国</td>
      <td>1966-05-06 00:00:00</td>
      <td>135</td>
      <td>1966</td>
      <td>7.8</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df[df["投票人数"] % 1 != 0] #小数异常值
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19791</th>
      <td>女教师 女教</td>
      <td>8.30</td>
      <td>剧情/犯罪</td>
      <td>日本</td>
      <td>1977-10-29 00:00:00</td>
      <td>100</td>
      <td>1977</td>
      <td>6.6</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>19804</th>
      <td>女郎漫游仙境 ドレミファ娘の血は騒</td>
      <td>5.90</td>
      <td>喜剧/歌舞</td>
      <td>日本</td>
      <td>1985-11-03 00:00:00</td>
      <td>80</td>
      <td>1985</td>
      <td>6.7</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>19820</th>
      <td>女仆日记</td>
      <td>12.87</td>
      <td>剧情</td>
      <td>法国</td>
      <td>2015-04-01 00:00:00</td>
      <td>96</td>
      <td>2015</td>
      <td>5.7</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>38055</th>
      <td>逃出亚卡拉</td>
      <td>12.87</td>
      <td>剧情/动作/惊悚/犯罪</td>
      <td>美国</td>
      <td>1979-09-20 00:00:00</td>
      <td>112</td>
      <td>1979</td>
      <td>7.8</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


**对于异常值，一般来说数量都会很少，在不影响整体数据分布的情况下，我们直接删除就可以了**

**其他属性的异常值处理，我们会在格式转换部分，进一步讨论**

```python
df = df[df.投票人数 > 0]
df = df[df["投票人数"] % 1 == 0]
```



## 3.5 数据保存与读入

数据处理之后，然后将数据重新保存到movie_data.xlsx 

```python
df.to_excel("movie_data.xlsx") #默认路径为现在文件夹所在的路径
```
读入已保存文件:

```python
df = pd.read_excel(r"C:\Users\Lovetianyi\Desktop\python\作业3\movie_data.xlsx",index_col = 0)

df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


## 3.6 数据格式转换
在做数据分析的时候，原始数据往往会因为各种各样的原因产生各种数据格式的问题。  
数据格式是我们非常需要注意的一点，数据格式错误往往会造成很严重的后果。  
并且，很多异常值也是我们经过格式转换后才会发现，对我们规整数据，清洗数据有着重要的作用。

### 3.6.1 查看格式 

```python
df["投票人数"].dtype

#dtype('int64')

df["投票人数"] = df["投票人数"].astype("int") #转换格式
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


```python
df["产地"].dtype
#dtype('O')

df["产地"] = df["产地"].astype("str")
```

### 3.6.2 将年份转化为整数格式 

```python
df["年代"] = df["年代"].astype("int") #有异常值会报错
```

    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-9-aafea50a8773> in <module>
    ----> 1 df["年代"] = df["年代"].astype("int") #有异常值会报错


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5696         else:
       5697             # else, only a single dtype is given
    -> 5698             new_data = self._data.astype(dtype=dtype, copy=copy, errors=errors)
       5699             return self._constructor(new_data).__finalize__(self)
       5700 


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in astype(self, dtype, copy, errors)
        580 
        581     def astype(self, dtype, copy: bool = False, errors: str = "raise"):
    --> 582         return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
        583 
        584     def convert(self, **kwargs):


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in apply(self, f, filter, **kwargs)
        440                 applied = b.apply(f, **kwargs)
        441             else:
    --> 442                 applied = getattr(b, f)(**kwargs)
        443             result_blocks = _extend_blocks(applied, result_blocks)
        444 


    ~\anaconda3\lib\site-packages\pandas\core\internals\blocks.py in astype(self, dtype, copy, errors)
        623             vals1d = values.ravel()
        624             try:
    --> 625                 values = astype_nansafe(vals1d, dtype, copy=True)
        626             except (ValueError, TypeError):
        627                 # e.g. astype_nansafe can fail on object-dtype of strings


    ~\anaconda3\lib\site-packages\pandas\core\dtypes\cast.py in astype_nansafe(arr, dtype, copy, skipna)
        872         # work around NumPy brokenness, #1987
        873         if np.issubdtype(dtype.type, np.integer):
    --> 874             return lib.astype_intsafe(arr.ravel(), dtype).reshape(arr.shape)
        875 
        876         # if we have a datetime/timedelta array of objects


    pandas\_libs\lib.pyx in pandas._libs.lib.astype_intsafe()


    ValueError: invalid literal for int() with base 10: '2008\u200e'

```python
df[df.年代 == "2008\u200e"] #找到异常数据
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15205</th>
      <td>狂蟒惊魂</td>
      <td>544</td>
      <td>恐怖</td>
      <td>中国大陆</td>
      <td>2008-04-08 00:00:00</td>
      <td>93</td>
      <td>2008‎</td>
      <td>2.7</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df[df.年代 == "2008\u200e"]["年代"].values #后面是unicode的控制字符，使得其显示靠左，因此需要处理删除
```

    array(['2008\u200e'], dtype=object)

```python
df.loc[[14934,15205],"年代"] = 2008
```

```python
df.loc[14934]
```
    名字                    奶奶强盗团
    投票人数                  12591
    类型                 剧情/喜剧/动作
    产地                       韩国
    上映时间    2010-03-18 00:00:00
    时长                      107
    年代                     2008
    评分                      7.7
    首映地点                     韩国
    Name: 14934, dtype: object

```python
df["年代"] = df["年代"].astype("int")#此时无报错
```

```python
df["年代"][:5]
```
    0    1994
    1    1957
    2    1997
    3    1994
    4    1993
    Name: 年代, dtype: int32

### 3.6.3 将时长转化为整数格式 

```python
df["时长"] = df["时长"].astype("int")
```

    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-16-97b0e6bbe2ae> in <module>
    ----> 1 df["时长"] = df["时长"].astype("int")


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5696         else:
       5697             # else, only a single dtype is given
    -> 5698             new_data = self._data.astype(dtype=dtype, copy=copy, errors=errors)
       5699             return self._constructor(new_data).__finalize__(self)
       5700 


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in astype(self, dtype, copy, errors)
        580 
        581     def astype(self, dtype, copy: bool = False, errors: str = "raise"):
    --> 582         return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
        583 
        584     def convert(self, **kwargs):


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in apply(self, f, filter, **kwargs)
        440                 applied = b.apply(f, **kwargs)
        441             else:
    --> 442                 applied = getattr(b, f)(**kwargs)
        443             result_blocks = _extend_blocks(applied, result_blocks)
        444 


    ~\anaconda3\lib\site-packages\pandas\core\internals\blocks.py in astype(self, dtype, copy, errors)
        623             vals1d = values.ravel()
        624             try:
    --> 625                 values = astype_nansafe(vals1d, dtype, copy=True)
        626             except (ValueError, TypeError):
        627                 # e.g. astype_nansafe can fail on object-dtype of strings


    ~\anaconda3\lib\site-packages\pandas\core\dtypes\cast.py in astype_nansafe(arr, dtype, copy, skipna)
        872         # work around NumPy brokenness, #1987
        873         if np.issubdtype(dtype.type, np.integer):
    --> 874             return lib.astype_intsafe(arr.ravel(), dtype).reshape(arr.shape)
        875 
        876         # if we have a datetime/timedelta array of objects


    pandas\_libs\lib.pyx in pandas._libs.lib.astype_intsafe()


    ValueError: invalid literal for int() with base 10: '8U'

```python
df[df["时长"] == "8U"] #寻找异常值，不知道怎么改的话可以删除
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31644</th>
      <td>一个被隔绝的世界</td>
      <td>46</td>
      <td>纪录片/短片</td>
      <td>瑞典</td>
      <td>2001-10-25 00:00:00</td>
      <td>8U</td>
      <td>1948</td>
      <td>7.8</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>


```python
df.drop([31644], inplace = True)
```

```python
df["时长"] = df["时长"].astype("int")
```

    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-19-97b0e6bbe2ae> in <module>
    ----> 1 df["时长"] = df["时长"].astype("int")


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in astype(self, dtype, copy, errors)
       5696         else:
       5697             # else, only a single dtype is given
    -> 5698             new_data = self._data.astype(dtype=dtype, copy=copy, errors=errors)
       5699             return self._constructor(new_data).__finalize__(self)
       5700 


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in astype(self, dtype, copy, errors)
        580 
        581     def astype(self, dtype, copy: bool = False, errors: str = "raise"):
    --> 582         return self.apply("astype", dtype=dtype, copy=copy, errors=errors)
        583 
        584     def convert(self, **kwargs):


    ~\anaconda3\lib\site-packages\pandas\core\internals\managers.py in apply(self, f, filter, **kwargs)
        440                 applied = b.apply(f, **kwargs)
        441             else:
    --> 442                 applied = getattr(b, f)(**kwargs)
        443             result_blocks = _extend_blocks(applied, result_blocks)
        444 


    ~\anaconda3\lib\site-packages\pandas\core\internals\blocks.py in astype(self, dtype, copy, errors)
        623             vals1d = values.ravel()
        624             try:
    --> 625                 values = astype_nansafe(vals1d, dtype, copy=True)
        626             except (ValueError, TypeError):
        627                 # e.g. astype_nansafe can fail on object-dtype of strings


    ~\anaconda3\lib\site-packages\pandas\core\dtypes\cast.py in astype_nansafe(arr, dtype, copy, skipna)
        872         # work around NumPy brokenness, #1987
        873         if np.issubdtype(dtype.type, np.integer):
    --> 874             return lib.astype_intsafe(arr.ravel(), dtype).reshape(arr.shape)
        875 
        876         # if we have a datetime/timedelta array of objects


    pandas\_libs\lib.pyx in pandas._libs.lib.astype_intsafe()


    ValueError: invalid literal for int() with base 10: '12J'

```python
df[df["时长"] == "12J"]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32949</th>
      <td>渔业危机</td>
      <td>41</td>
      <td>纪录片</td>
      <td>英国</td>
      <td>2009-06-19 00:00:00</td>
      <td>12J</td>
      <td>2008</td>
      <td>8.2</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>


```python
df.drop([32949], inplace = True) #删数据，inplace替换原来数据
```

```python
df["时长"] = df["时长"].astype("int")
```

```python
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>




## 3.7 排序

### 3.7.1 默认排序

```python
df[:10]#默认根据index进行排序
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>美国</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>1957</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>意大利</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>1997</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>中国大陆</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>1993</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>美国</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>2012</td>
      <td>9.4</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>6</th>
      <td>辛德勒的名单</td>
      <td>306904</td>
      <td>剧情/历史/战争</td>
      <td>美国</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>1993</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
    </tr>
    <tr>
      <th>7</th>
      <td>新世纪福音战士剧场版：Air/真心为你 新世紀エヴァンゲリオン劇場版 Ai</td>
      <td>24355</td>
      <td>剧情/动作/科幻/动画/奇幻</td>
      <td>日本</td>
      <td>1997-07-19 00:00:00</td>
      <td>87</td>
      <td>1997</td>
      <td>9.4</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>8</th>
      <td>银魂完结篇：直到永远的万事屋 劇場版 銀魂 完結篇 万事屋よ</td>
      <td>21513</td>
      <td>剧情/动画</td>
      <td>日本</td>
      <td>2013-07-06 00:00:00</td>
      <td>110</td>
      <td>2013</td>
      <td>9.4</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>9</th>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>法国</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>1994</td>
      <td>9.4</td>
      <td>法国</td>
    </tr>
  </tbody>
</table>


### 3.7.2 按照投票人数进行排序  sort_values(by= , ascending = <Bool>)[:]


```python
df.sort_values(by = "投票人数", ascending = False)[:5] #默认从小到大
```

**by:指定某项** 

**ascending:**

​	**True:默认升序**

​	**False:降序**



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>美国</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>9</th>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>法国</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>1994</td>
      <td>9.4</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>22</th>
      <td>盗梦空间</td>
      <td>642134</td>
      <td>剧情/动作/科幻/悬疑/冒险</td>
      <td>美国</td>
      <td>2010-09-01 00:00:00</td>
      <td>148</td>
      <td>2010</td>
      <td>9.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>美国</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>1994</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>99</th>
      <td>三傻大闹宝莱坞</td>
      <td>549808</td>
      <td>剧情/喜剧/爱情/歌舞</td>
      <td>印度</td>
      <td>2011-12-08 00:00:00</td>
      <td>171</td>
      <td>2009</td>
      <td>9.1</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>


### 3.7.3 按照年代进行排序


```python
df.sort_values(by = "年代")[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1700</th>
      <td>朗德海花园场景</td>
      <td>650</td>
      <td>短片</td>
      <td>英国</td>
      <td>1888-10-14</td>
      <td>60</td>
      <td>1888</td>
      <td>8.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>14048</th>
      <td>利兹大桥</td>
      <td>126</td>
      <td>短片</td>
      <td>英国</td>
      <td>1888-10</td>
      <td>60</td>
      <td>1888</td>
      <td>7.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>26170</th>
      <td>恶作剧</td>
      <td>51</td>
      <td>短片</td>
      <td>美国</td>
      <td>1905-03-04 00:00:00</td>
      <td>60</td>
      <td>1890</td>
      <td>4.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>10627</th>
      <td>可怜的比埃洛</td>
      <td>176</td>
      <td>喜剧/爱情/动画/短片</td>
      <td>法国</td>
      <td>1892-10-28</td>
      <td>60</td>
      <td>1892</td>
      <td>7.5</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>21765</th>
      <td>胚胎植入前遗传学筛查</td>
      <td>69</td>
      <td>纪录片/短片</td>
      <td>美国</td>
      <td>1894-05-18</td>
      <td>60</td>
      <td>1894</td>
      <td>5.7</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>




### 3.7.4 多个值排序，先按照评分，再按照投票人数  sort_values(by = [" "," "],ascending = <Bool>)


```python
df.sort_values(by = ["评分","投票人数"], ascending = False) #列表中的顺序决定先后顺序
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9278</th>
      <td>平安结祈 平安結</td>
      <td>208</td>
      <td>音乐</td>
      <td>日本</td>
      <td>2012-02-24 00:00:00</td>
      <td>60</td>
      <td>2012</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>13882</th>
      <td>武之舞</td>
      <td>128</td>
      <td>纪录片</td>
      <td>中国大陆</td>
      <td>1997-02-01 00:00:00</td>
      <td>60</td>
      <td>34943</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>23559</th>
      <td>未作回答的问题：伯恩斯坦哈佛六讲</td>
      <td>61</td>
      <td>纪录片</td>
      <td>美国</td>
      <td>1905-05-29 00:00:00</td>
      <td>60</td>
      <td>1973</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>25273</th>
      <td>索科洛夫：巴黎现场</td>
      <td>43</td>
      <td>音乐</td>
      <td>法国</td>
      <td>2002-11-04 00:00:00</td>
      <td>127</td>
      <td>2002</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>11479</th>
      <td>公园现场</td>
      <td>163</td>
      <td>音乐</td>
      <td>英国</td>
      <td>2012-12-03 00:00:00</td>
      <td>60</td>
      <td>2012</td>
      <td>9.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2790</th>
      <td>爸爸我来救你了</td>
      <td>278</td>
      <td>喜剧/动作/家庭/儿童/冒险</td>
      <td>中国大陆</td>
      <td>2016-01-22 00:00:00</td>
      <td>90</td>
      <td>2015</td>
      <td>2.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>10219</th>
      <td>大震撼</td>
      <td>185</td>
      <td>剧情</td>
      <td>中国大陆</td>
      <td>2011-05-19 00:00:00</td>
      <td>60</td>
      <td>2011</td>
      <td>2.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>9045</th>
      <td>钢管侠</td>
      <td>168</td>
      <td>动作</td>
      <td>中国大陆</td>
      <td>2015-07-28 00:00:00</td>
      <td>60</td>
      <td>2015</td>
      <td>2.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>13137</th>
      <td>嫁给大山的女人</td>
      <td>2690</td>
      <td>剧情</td>
      <td>中国大陆</td>
      <td>2009-04-22 00:00:00</td>
      <td>88</td>
      <td>2009</td>
      <td>2.1</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>29100</th>
      <td>都是手机惹的祸</td>
      <td>42</td>
      <td>喜剧</td>
      <td>中国大陆</td>
      <td>2013-01-18 00:00:00</td>
      <td>60</td>
      <td>2012</td>
      <td>2.0</td>
      <td>中国大陆</td>
    </tr>
  </tbody>
</table>
<p>38167 rows × 9 columns</p>




## 3.8 基本统计分析

### 3.8.1 描述性统计

**dataframe.describe()：对dataframe中的数值型数据进行描述性统计(概览)**


```python
df.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>38167.000000</td>
      <td>38167.000000</td>
      <td>38167.000000</td>
      <td>38167.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6268.131291</td>
      <td>89.475594</td>
      <td>1998.805224</td>
      <td>6.922154</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26298.331602</td>
      <td>83.763856</td>
      <td>255.065394</td>
      <td>1.263782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.000000</td>
      <td>1.000000</td>
      <td>1888.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>101.000000</td>
      <td>60.000000</td>
      <td>1990.000000</td>
      <td>6.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>354.000000</td>
      <td>93.000000</td>
      <td>2005.000000</td>
      <td>7.100000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1798.500000</td>
      <td>106.000000</td>
      <td>2010.000000</td>
      <td>7.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>692795.000000</td>
      <td>11500.000000</td>
      <td>39180.000000</td>
      <td>9.900000</td>
    </tr>
  </tbody>
</table>




**通过描述性统计，可以发现一些异常值，很多异常值往往是需要我们逐步去发现的。** 


```python
df[df["年代"] > 2018] #异常值
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13882</th>
      <td>武之舞</td>
      <td>128</td>
      <td>纪录片</td>
      <td>中国大陆</td>
      <td>1997-02-01 00:00:00</td>
      <td>60</td>
      <td>34943</td>
      <td>9.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>17115</th>
      <td>妈妈回来吧-中国打工村的孩子</td>
      <td>49</td>
      <td>纪录片</td>
      <td>日本</td>
      <td>2007-04-08 00:00:00</td>
      <td>109</td>
      <td>39180</td>
      <td>8.9</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>



```python
df[df["时长"] > 1000] #异常值
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>产地</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>年代</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19690</th>
      <td>怒海余生</td>
      <td>54</td>
      <td>剧情/家庭/冒险</td>
      <td>美国</td>
      <td>1937-09-01 00:00:00</td>
      <td>11500</td>
      <td>1937</td>
      <td>7.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>38730</th>
      <td>喧闹村的孩子们</td>
      <td>36</td>
      <td>家庭</td>
      <td>瑞典</td>
      <td>1986-12-06 00:00:00</td>
      <td>9200</td>
      <td>1986</td>
      <td>8.7</td>
      <td>瑞典</td>
    </tr>
  </tbody>
</table>



```python
df.drop(df[df["年代"] > 2018].index, inplace = True)
df.drop(df[df["时长"] > 1000].index, inplace = True) #删除异常数据
```


```python
df.index = range(len(df)) #解决删除后索引不连续的问题
```

不要忽视**索引不连续**！

### 3.8.2 最值 max()/min()

```python
df["投票人数"].max()
#692795

df["投票人数"].min()
#21

df["评分"].max()
#9.9

df["评分"].min()
#2.0

df["年代"].min()
#1888
```



### 3.8.3 均值和中值 mean()/median()

```python
df["投票人数"].mean()
#6268.7812802976705

df["投票人数"].median()
#354.0

df["评分"].mean()
#6.921951515969828

df["评分"].median()
#7.1
```



### 3.8.4 方差和标准差 var()/std()

```python
df["评分"].var()
#1.5968697056255758

df["评分"].std()
#1.263673100776295
```



### 3.8.5 求和 sum()

```python
df["投票人数"].sum()
#239235500
```



### 3.8.6 相关系数和协方差 corr()/cov()

```python
df[["投票人数", "评分"]].corr()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>投票人数</th>
      <td>1.000000</td>
      <td>0.126953</td>
    </tr>
    <tr>
      <th>评分</th>
      <td>0.126953</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>



```python
df[["投票人数", "评分"]].cov()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>投票人数</th>
      <td>6.916707e+08</td>
      <td>4219.174348</td>
    </tr>
    <tr>
      <th>评分</th>
      <td>4.219174e+03</td>
      <td>1.596870</td>
    </tr>
  </tbody>
</table>




### 3.8.7 计数

```python
len(df)
```


    38163


```python
df["产地"].unique() #指定唯一值
```


    array(['美国', '意大利', '中国大陆', '日本', '法国', '英国', '韩国', '中国香港', '阿根廷', '德国',
           '印度', '其他', '加拿大', '波兰', '泰国', '澳大利亚', '西班牙', '俄罗斯', '中国台湾', '荷兰',
           '丹麦', '比利时', '巴西', '瑞典', '墨西哥'], dtype=object)


```python
len(df["产地"].unique())
```


    25

产地中包含了一些重复的数据，比如美国和USA，德国和西德，俄罗斯和苏联
我们可以通过**数据替换**的方法将这些相同国家的电影数据合并一下。

**replace():替换**

**unique():独特**


```python
df["产地"].replace("USA","美国",inplace = True) #第一个参数是要替换的值，第二个参数是替换后的值
```


```python
df["产地"].replace(["西德","苏联"],["德国","俄罗斯"], inplace = True) #注意一一对应
```


```python
len(df["产地"].unique())
```


    25


```python
df["年代"].unique()
```


    array([1994, 1957, 1997, 1993, 2012, 2013, 2003, 2016, 2009, 2008, 2001,
           1931, 1961, 2010, 2004, 1998, 1972, 1939, 2015, 1946, 2011, 1982,
           1960, 2006, 1988, 2002, 1995, 1996, 1984, 2014, 1953, 2007, 2000,
           1967, 1983, 1963, 1977, 1966, 1971, 1974, 1985, 1987, 1973, 1962,
           1969, 1989, 1979, 1981, 1936, 1954, 1992, 1970, 1991, 2005, 1920,
           1933, 1990, 1999, 1896, 1965, 1921, 1947, 1975, 1964, 1943, 1928,
           1986, 1895, 1949, 1932, 1905, 1940, 1908, 1900, 1978, 1951, 1958,
           1898, 1976, 1938, 1907, 1948, 1952, 1926, 1955, 1906, 1959, 1934,
           1944, 1888, 1909, 1925, 1956, 1923, 1945, 1913, 1903, 1904, 1980,
           1968, 1917, 1935, 1942, 1950, 1902, 1941, 1930, 1937, 1922, 1916,
           1929, 1927, 1919, 1914, 1912, 1924, 1918, 1899, 1901, 1915, 1892,
           1894, 1910, 1897, 1911, 1890, 2018])


```python
len(df["年代"].unique())
```


    127

**计算每一年电影的数量：value_counts(ascending = <Bool>)[:]**


```python
df["年代"].value_counts(ascending = True)[:10] #默认从大到小
```


    1890    1
    2018    1
    1892    1
    1899    2
    1898    2
    1888    2
    1894    3
    1897    3
    1911    3
    1909    4
    Name: 年代, dtype: int64

电影产出前5的国家或地区：


```python
df["产地"].value_counts()[:5]
```


    美国      11714
    日本       5006
    中国大陆     3791
    中国香港     2847
    法国       2787
    Name: 产地, dtype: int64

保存数据


```python
df.to_excel("movie_data2.xlsx")
```

## 3.9 数据透视
Excel中数据透视表的使用非常广泛，其实Pandas也提供了一个类似的功能，名为**pivot_table**。

pivot_table非常有用，我们将重点解释pandas中的函数pivot_table。

使用pandas中的pivot_table的一个挑战是，你需要确保你理解你的数据，并清楚地知道你想通过透视表解决什么问题。虽然pivot_table看起来只是一个简单的函数，但是它能够快速地对数据进行强大的分析。

### 3.9.1 基础形式
```python
pd.set_option("max_columns",100) #设置可展示的行和列，让数据进行完整展示
pd.set_option("max_rows",500)
```

```python
pd.pivot_table(df, index = ["年代"]) #统计各个年代中所有数值型数据的均值（默认）
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>388.000000</td>
      <td>60.000000</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51.000000</td>
      <td>60.000000</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176.000000</td>
      <td>60.000000</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>112.666667</td>
      <td>60.000000</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>959.875000</td>
      <td>60.000000</td>
      <td>7.575000</td>
    </tr>
    <tr>
      <th>1896</th>
      <td>984.250000</td>
      <td>60.000000</td>
      <td>7.037500</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>67.000000</td>
      <td>60.000000</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1898</th>
      <td>578.500000</td>
      <td>60.000000</td>
      <td>7.450000</td>
    </tr>
    <tr>
      <th>1899</th>
      <td>71.000000</td>
      <td>9.500000</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>175.285714</td>
      <td>36.714286</td>
      <td>7.228571</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>164.500000</td>
      <td>47.250000</td>
      <td>7.250000</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>2309.600000</td>
      <td>29.600000</td>
      <td>7.680000</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>349.846154</td>
      <td>27.000000</td>
      <td>7.015385</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>249.166667</td>
      <td>35.000000</td>
      <td>7.616667</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>332.600000</td>
      <td>43.800000</td>
      <td>6.820000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>189.857143</td>
      <td>30.571429</td>
      <td>7.342857</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>213.600000</td>
      <td>31.800000</td>
      <td>7.020000</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>368.750000</td>
      <td>35.000000</td>
      <td>7.425000</td>
    </tr>
    <tr>
      <th>1909</th>
      <td>62.000000</td>
      <td>10.750000</td>
      <td>7.650000</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>105.200000</td>
      <td>41.800000</td>
      <td>6.940000</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>308.000000</td>
      <td>28.666667</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>181.000000</td>
      <td>18.000000</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>1913</th>
      <td>62.285714</td>
      <td>64.571429</td>
      <td>6.671429</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>104.923077</td>
      <td>25.923077</td>
      <td>6.473077</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>314.900000</td>
      <td>56.800000</td>
      <td>7.260000</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>666.636364</td>
      <td>42.363636</td>
      <td>7.690909</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>124.416667</td>
      <td>31.333333</td>
      <td>7.075000</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>357.083333</td>
      <td>35.166667</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>194.777778</td>
      <td>64.611111</td>
      <td>7.494444</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>636.500000</td>
      <td>59.357143</td>
      <td>7.492857</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>729.818182</td>
      <td>57.363636</td>
      <td>7.750000</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>767.090909</td>
      <td>66.363636</td>
      <td>7.804545</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>447.705882</td>
      <td>74.882353</td>
      <td>7.811765</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>384.518519</td>
      <td>81.962963</td>
      <td>8.059259</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>1104.280000</td>
      <td>84.440000</td>
      <td>7.788000</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>443.608696</td>
      <td>80.304348</td>
      <td>7.773913</td>
    </tr>
    <tr>
      <th>1927</th>
      <td>695.275862</td>
      <td>87.241379</td>
      <td>7.751724</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>413.666667</td>
      <td>72.076923</td>
      <td>7.964103</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>740.542857</td>
      <td>69.371429</td>
      <td>7.440000</td>
    </tr>
    <tr>
      <th>1930</th>
      <td>555.080000</td>
      <td>74.160000</td>
      <td>7.360000</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>1468.666667</td>
      <td>78.523810</td>
      <td>7.483333</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>600.081081</td>
      <td>77.540541</td>
      <td>7.294595</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>756.020833</td>
      <td>79.187500</td>
      <td>7.420833</td>
    </tr>
    <tr>
      <th>1934</th>
      <td>791.460000</td>
      <td>83.260000</td>
      <td>7.536000</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>887.695652</td>
      <td>73.673913</td>
      <td>7.515217</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>1489.220339</td>
      <td>77.440678</td>
      <td>7.615254</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>1612.104167</td>
      <td>87.187500</td>
      <td>7.568750</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>552.000000</td>
      <td>85.973684</td>
      <td>7.736842</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>5911.857143</td>
      <td>97.387755</td>
      <td>7.520408</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>5689.815789</td>
      <td>93.684211</td>
      <td>7.544737</td>
    </tr>
    <tr>
      <th>1941</th>
      <td>1552.808511</td>
      <td>89.127660</td>
      <td>7.427660</td>
    </tr>
    <tr>
      <th>1942</th>
      <td>2607.754717</td>
      <td>78.264151</td>
      <td>7.554717</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>755.357143</td>
      <td>79.714286</td>
      <td>7.605357</td>
    </tr>
    <tr>
      <th>1944</th>
      <td>1007.370370</td>
      <td>81.925926</td>
      <td>7.538889</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>989.020408</td>
      <td>86.959184</td>
      <td>7.673469</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>1034.457627</td>
      <td>85.016949</td>
      <td>7.606780</td>
    </tr>
    <tr>
      <th>1947</th>
      <td>443.702703</td>
      <td>87.486486</td>
      <td>7.502703</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>1199.255814</td>
      <td>88.534884</td>
      <td>7.645349</td>
    </tr>
    <tr>
      <th>1949</th>
      <td>641.685393</td>
      <td>81.988764</td>
      <td>7.646067</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>2235.026316</td>
      <td>80.157895</td>
      <td>7.655263</td>
    </tr>
    <tr>
      <th>1951</th>
      <td>967.884615</td>
      <td>86.653846</td>
      <td>7.637179</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>1507.305882</td>
      <td>82.658824</td>
      <td>7.775294</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>4840.620690</td>
      <td>84.448276</td>
      <td>7.579310</td>
    </tr>
    <tr>
      <th>1954</th>
      <td>2201.356436</td>
      <td>86.326733</td>
      <td>7.714851</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>2000.491228</td>
      <td>82.912281</td>
      <td>7.567544</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>1061.675862</td>
      <td>76.944828</td>
      <td>7.591724</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>3057.586538</td>
      <td>88.884615</td>
      <td>7.622115</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>886.196721</td>
      <td>82.975410</td>
      <td>7.536885</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>1725.070312</td>
      <td>90.070312</td>
      <td>7.571875</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>1457.107692</td>
      <td>101.530769</td>
      <td>7.580769</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>3249.750000</td>
      <td>99.510000</td>
      <td>7.741000</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>1985.661972</td>
      <td>92.225352</td>
      <td>7.707042</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>1184.027972</td>
      <td>92.020979</td>
      <td>7.536364</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>1125.441341</td>
      <td>91.162011</td>
      <td>7.540782</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>2012.248447</td>
      <td>91.434783</td>
      <td>7.591304</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>1326.640449</td>
      <td>89.651685</td>
      <td>7.521910</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>1243.891429</td>
      <td>92.074286</td>
      <td>7.477143</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>1107.841176</td>
      <td>92.594118</td>
      <td>7.324706</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>601.279412</td>
      <td>99.475490</td>
      <td>7.367647</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>683.468085</td>
      <td>97.484043</td>
      <td>7.294149</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>1380.576037</td>
      <td>96.695853</td>
      <td>7.149309</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>2332.132353</td>
      <td>96.372549</td>
      <td>7.253431</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>810.809524</td>
      <td>95.471429</td>
      <td>7.238095</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>1693.171171</td>
      <td>95.477477</td>
      <td>7.063964</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>2256.704663</td>
      <td>97.813472</td>
      <td>7.056995</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>1340.618026</td>
      <td>95.055794</td>
      <td>7.107725</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>965.444954</td>
      <td>98.197248</td>
      <td>7.151376</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>1001.820896</td>
      <td>94.467662</td>
      <td>7.096517</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>1902.991071</td>
      <td>96.250000</td>
      <td>7.292857</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>2417.040359</td>
      <td>94.443946</td>
      <td>7.182063</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>1620.566176</td>
      <td>93.709559</td>
      <td>7.154044</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>2174.809160</td>
      <td>92.889313</td>
      <td>7.286260</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>1551.894545</td>
      <td>91.985455</td>
      <td>7.296727</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>2676.129693</td>
      <td>91.023891</td>
      <td>7.380887</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>1612.430380</td>
      <td>93.974684</td>
      <td>7.267722</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>2836.777090</td>
      <td>89.235294</td>
      <td>7.249536</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>3146.832845</td>
      <td>89.269795</td>
      <td>7.282111</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>4859.352332</td>
      <td>89.292746</td>
      <td>7.265544</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>3009.383632</td>
      <td>88.501279</td>
      <td>7.199233</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>5003.312821</td>
      <td>91.976923</td>
      <td>7.156923</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>4708.414216</td>
      <td>93.661765</td>
      <td>7.154412</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>5573.535047</td>
      <td>92.670561</td>
      <td>7.190187</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>8154.555809</td>
      <td>95.473804</td>
      <td>7.186560</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>11591.339468</td>
      <td>91.554192</td>
      <td>7.257260</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>8266.887734</td>
      <td>95.114345</td>
      <td>7.275052</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>5154.665362</td>
      <td>95.819961</td>
      <td>7.249119</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>8383.681818</td>
      <td>95.446970</td>
      <td>7.325758</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>7587.203540</td>
      <td>92.021239</td>
      <td>7.223540</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>7228.077796</td>
      <td>92.914100</td>
      <td>7.171151</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6022.013947</td>
      <td>92.917713</td>
      <td>7.112413</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>8740.049751</td>
      <td>91.342040</td>
      <td>7.070647</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>6705.952218</td>
      <td>92.268487</td>
      <td>7.045620</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>8104.751896</td>
      <td>90.695558</td>
      <td>7.114410</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>8758.644749</td>
      <td>91.376256</td>
      <td>6.999909</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>6626.300242</td>
      <td>92.804681</td>
      <td>7.011864</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>7407.740591</td>
      <td>90.619624</td>
      <td>6.901546</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>6346.090801</td>
      <td>86.842136</td>
      <td>6.859703</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>6876.232486</td>
      <td>85.589517</td>
      <td>6.892164</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>8694.675558</td>
      <td>86.681546</td>
      <td>6.732825</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>9691.044576</td>
      <td>83.785177</td>
      <td>6.752793</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>8860.328455</td>
      <td>84.437398</td>
      <td>6.560325</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>7165.750248</td>
      <td>85.321606</td>
      <td>6.444896</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>7694.106727</td>
      <td>85.337380</td>
      <td>6.375974</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>7803.983931</td>
      <td>86.354580</td>
      <td>6.249384</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>7954.999363</td>
      <td>90.338432</td>
      <td>6.121925</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>7341.388889</td>
      <td>91.646825</td>
      <td>5.834524</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>123456.000000</td>
      <td>142.000000</td>
      <td>6.935704</td>
    </tr>
  </tbody>
</table>


### 3.9.2 多索引
也可以有多个索引。实际上，大多数的pivot_table参数可以通过列表获取多个值。

```python
pd.pivot_table(df, index = ["年代", "产地"]) #双索引
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <th>英国</th>
      <td>388.000000</td>
      <td>60.000000</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <th>美国</th>
      <td>51.000000</td>
      <td>60.000000</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <th>法国</th>
      <td>176.000000</td>
      <td>60.000000</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1894</th>
      <th>法国</th>
      <td>148.000000</td>
      <td>60.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>95.000000</td>
      <td>60.000000</td>
      <td>6.450000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2016</th>
      <th>法国</th>
      <td>44.666667</td>
      <td>104.333333</td>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>11224.225806</td>
      <td>93.161290</td>
      <td>6.522581</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>14607.272727</td>
      <td>85.545455</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>1739.850000</td>
      <td>106.100000</td>
      <td>5.730000</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>123456.000000</td>
      <td>142.000000</td>
      <td>6.935704</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 3 columns</p>


### 3.9.3 指定需要统计汇总的数据
```python
pd.pivot_table(df, index = ["年代", "产地"], values = ["评分"])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <th>英国</th>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <th>美国</th>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <th>法国</th>
      <td>7.500000</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1894</th>
      <th>法国</th>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>6.450000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2016</th>
      <th>法国</th>
      <td>7.300000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>6.522581</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>5.730000</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>6.935704</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 1 columns</p>


### 3.9.4 指定函数统计不同统计值
```python
pd.pivot_table(df, index = ["年代", "产地"], values = ["投票人数"], aggfunc = np.sum)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <th>英国</th>
      <td>776</td>
    </tr>
    <tr>
      <th>1890</th>
      <th>美国</th>
      <td>51</td>
    </tr>
    <tr>
      <th>1892</th>
      <th>法国</th>
      <td>176</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1894</th>
      <th>法国</th>
      <td>148</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>190</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">2016</th>
      <th>法国</th>
      <td>134</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>695902</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>160680</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>34797</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>123456</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 1 columns</p>


通过将“投票人数”列和“评分”列进行对应分组，对“产地”实现数据聚合和总结。

```python
pd.pivot_table(df, index = ["产地"], values = ["投票人数", "评分"], aggfunc = [np.sum, np.mean])
```


<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">sum</th>
      <th colspan="2" halign="left">mean</th>
    </tr>
    <tr>
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>5237466</td>
      <td>4367.200000</td>
      <td>8474.864078</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>41380993</td>
      <td>22984.800000</td>
      <td>10915.587708</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>23179448</td>
      <td>18430.200000</td>
      <td>8141.709870</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>394745</td>
      <td>1427.600000</td>
      <td>2003.781726</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>486082</td>
      <td>3597.200000</td>
      <td>1021.180672</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3048849</td>
      <td>13607.900000</td>
      <td>1619.144450</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1362581</td>
      <td>4769.600000</td>
      <td>1921.834979</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>1146173</td>
      <td>2443.900000</td>
      <td>3219.587079</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>139462</td>
      <td>829.000000</td>
      <td>1191.982906</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>357027</td>
      <td>716.000000</td>
      <td>3606.333333</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2679856</td>
      <td>7338.300000</td>
      <td>2624.736533</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>2500842</td>
      <td>5322.700000</td>
      <td>3374.955466</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>17981631</td>
      <td>36006.000000</td>
      <td>3592.015781</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>170449</td>
      <td>986.000000</td>
      <td>1244.153285</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>10208966</td>
      <td>20186.500000</td>
      <td>3663.066380</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>159577</td>
      <td>1347.000000</td>
      <td>881.640884</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>1564881</td>
      <td>1796.100000</td>
      <td>5322.724490</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>1415443</td>
      <td>2051.300000</td>
      <td>4798.111864</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>289794</td>
      <td>1388.600000</td>
      <td>1549.700535</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>101645832</td>
      <td>81100.135704</td>
      <td>8677.294861</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>13236409</td>
      <td>19930.800000</td>
      <td>4979.837848</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>144596</td>
      <td>1081.200000</td>
      <td>957.589404</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>1486383</td>
      <td>3112.100000</td>
      <td>3355.266366</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>258085</td>
      <td>819.100000</td>
      <td>2283.938053</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>8759930</td>
      <td>8523.200000</td>
      <td>6527.518629</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>


### 3.9.5 非数值(NaN)的处理
非数值（NaN）难以处理。如果想移除它们，可以使用“fill_value”将其设置为0。
```python
pd.pivot_table(df, index = ["产地"], aggfunc = [np.sum, np.mean], fill_value = 0)
```


<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">sum</th>
      <th colspan="4" halign="left">mean</th>
    </tr>
    <tr>
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1235388</td>
      <td>5237466</td>
      <td>53925</td>
      <td>4367.200000</td>
      <td>1999.009709</td>
      <td>8474.864078</td>
      <td>87.257282</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>7599372</td>
      <td>41380993</td>
      <td>309031</td>
      <td>22984.800000</td>
      <td>2004.582432</td>
      <td>10915.587708</td>
      <td>81.517014</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>5668630</td>
      <td>23179448</td>
      <td>252111</td>
      <td>18430.200000</td>
      <td>1991.088865</td>
      <td>8141.709870</td>
      <td>88.553214</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>393821</td>
      <td>394745</td>
      <td>17436</td>
      <td>1427.600000</td>
      <td>1999.091371</td>
      <td>2003.781726</td>
      <td>88.507614</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>944809</td>
      <td>486082</td>
      <td>45744</td>
      <td>3597.200000</td>
      <td>1984.892857</td>
      <td>1021.180672</td>
      <td>96.100840</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3763593</td>
      <td>3048849</td>
      <td>165057</td>
      <td>13607.900000</td>
      <td>1998.721721</td>
      <td>1619.144450</td>
      <td>87.656399</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1419787</td>
      <td>1362581</td>
      <td>57140</td>
      <td>4769.600000</td>
      <td>2002.520451</td>
      <td>1921.834979</td>
      <td>80.592384</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>714150</td>
      <td>1146173</td>
      <td>43058</td>
      <td>2443.900000</td>
      <td>2006.039326</td>
      <td>3219.587079</td>
      <td>120.949438</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>233156</td>
      <td>139462</td>
      <td>10839</td>
      <td>829.000000</td>
      <td>1992.786325</td>
      <td>1191.982906</td>
      <td>92.641026</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>197989</td>
      <td>357027</td>
      <td>8749</td>
      <td>716.000000</td>
      <td>1999.888889</td>
      <td>3606.333333</td>
      <td>88.373737</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2037971</td>
      <td>2679856</td>
      <td>94196</td>
      <td>7338.300000</td>
      <td>1996.053869</td>
      <td>2624.736533</td>
      <td>92.258570</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1471329</td>
      <td>2500842</td>
      <td>77311</td>
      <td>5322.700000</td>
      <td>1985.599190</td>
      <td>3374.955466</td>
      <td>104.333333</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>10011432</td>
      <td>17981631</td>
      <td>425563</td>
      <td>36006.000000</td>
      <td>1999.886536</td>
      <td>3592.015781</td>
      <td>85.010587</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>273932</td>
      <td>170449</td>
      <td>11380</td>
      <td>986.000000</td>
      <td>1999.503650</td>
      <td>1244.153285</td>
      <td>83.065693</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>5551130</td>
      <td>10208966</td>
      <td>251524</td>
      <td>20186.500000</td>
      <td>1991.794044</td>
      <td>3663.066380</td>
      <td>90.249013</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>359652</td>
      <td>159577</td>
      <td>14613</td>
      <td>1347.000000</td>
      <td>1987.027624</td>
      <td>881.640884</td>
      <td>80.734807</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>590684</td>
      <td>1564881</td>
      <td>26002</td>
      <td>1796.100000</td>
      <td>2009.129252</td>
      <td>5322.724490</td>
      <td>88.442177</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>590875</td>
      <td>1415443</td>
      <td>25250</td>
      <td>2051.300000</td>
      <td>2002.966102</td>
      <td>4798.111864</td>
      <td>85.593220</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>371589</td>
      <td>289794</td>
      <td>17695</td>
      <td>1388.600000</td>
      <td>1987.106952</td>
      <td>1549.700535</td>
      <td>94.625668</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>23363806</td>
      <td>101645832</td>
      <td>1053980</td>
      <td>81100.135704</td>
      <td>1994.519891</td>
      <td>8677.294861</td>
      <td>89.976097</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>5307045</td>
      <td>13236409</td>
      <td>237129</td>
      <td>19930.800000</td>
      <td>1996.630926</td>
      <td>4979.837848</td>
      <td>89.213318</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>302181</td>
      <td>144596</td>
      <td>11459</td>
      <td>1081.200000</td>
      <td>2001.198675</td>
      <td>957.589404</td>
      <td>75.887417</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>886685</td>
      <td>1486383</td>
      <td>40271</td>
      <td>3112.100000</td>
      <td>2001.546275</td>
      <td>3355.266366</td>
      <td>90.905192</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>226476</td>
      <td>258085</td>
      <td>10458</td>
      <td>819.100000</td>
      <td>2004.212389</td>
      <td>2283.938053</td>
      <td>92.548673</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2694871</td>
      <td>8759930</td>
      <td>134225</td>
      <td>8523.200000</td>
      <td>2008.100596</td>
      <td>6527.518629</td>
      <td>100.018629</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>


### 3.9.6 显示总和数据
加入margins = True，可以在下方显示一些总和数据。

```python
pd.pivot_table(df, index = ["产地"], aggfunc = [np.sum, np.mean], fill_value = 0, margins = True)
```



<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">sum</th>
      <th colspan="4" halign="left">mean</th>
    </tr>
    <tr>
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1235388</td>
      <td>5237466</td>
      <td>53925</td>
      <td>4367.200000</td>
      <td>1999.009709</td>
      <td>8474.864078</td>
      <td>87.257282</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>7599372</td>
      <td>41380993</td>
      <td>309031</td>
      <td>22984.800000</td>
      <td>2004.582432</td>
      <td>10915.587708</td>
      <td>81.517014</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>5668630</td>
      <td>23179448</td>
      <td>252111</td>
      <td>18430.200000</td>
      <td>1991.088865</td>
      <td>8141.709870</td>
      <td>88.553214</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>393821</td>
      <td>394745</td>
      <td>17436</td>
      <td>1427.600000</td>
      <td>1999.091371</td>
      <td>2003.781726</td>
      <td>88.507614</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>944809</td>
      <td>486082</td>
      <td>45744</td>
      <td>3597.200000</td>
      <td>1984.892857</td>
      <td>1021.180672</td>
      <td>96.100840</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3763593</td>
      <td>3048849</td>
      <td>165057</td>
      <td>13607.900000</td>
      <td>1998.721721</td>
      <td>1619.144450</td>
      <td>87.656399</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1419787</td>
      <td>1362581</td>
      <td>57140</td>
      <td>4769.600000</td>
      <td>2002.520451</td>
      <td>1921.834979</td>
      <td>80.592384</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>714150</td>
      <td>1146173</td>
      <td>43058</td>
      <td>2443.900000</td>
      <td>2006.039326</td>
      <td>3219.587079</td>
      <td>120.949438</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>233156</td>
      <td>139462</td>
      <td>10839</td>
      <td>829.000000</td>
      <td>1992.786325</td>
      <td>1191.982906</td>
      <td>92.641026</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>197989</td>
      <td>357027</td>
      <td>8749</td>
      <td>716.000000</td>
      <td>1999.888889</td>
      <td>3606.333333</td>
      <td>88.373737</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2037971</td>
      <td>2679856</td>
      <td>94196</td>
      <td>7338.300000</td>
      <td>1996.053869</td>
      <td>2624.736533</td>
      <td>92.258570</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1471329</td>
      <td>2500842</td>
      <td>77311</td>
      <td>5322.700000</td>
      <td>1985.599190</td>
      <td>3374.955466</td>
      <td>104.333333</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>10011432</td>
      <td>17981631</td>
      <td>425563</td>
      <td>36006.000000</td>
      <td>1999.886536</td>
      <td>3592.015781</td>
      <td>85.010587</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>273932</td>
      <td>170449</td>
      <td>11380</td>
      <td>986.000000</td>
      <td>1999.503650</td>
      <td>1244.153285</td>
      <td>83.065693</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>5551130</td>
      <td>10208966</td>
      <td>251524</td>
      <td>20186.500000</td>
      <td>1991.794044</td>
      <td>3663.066380</td>
      <td>90.249013</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>359652</td>
      <td>159577</td>
      <td>14613</td>
      <td>1347.000000</td>
      <td>1987.027624</td>
      <td>881.640884</td>
      <td>80.734807</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>590684</td>
      <td>1564881</td>
      <td>26002</td>
      <td>1796.100000</td>
      <td>2009.129252</td>
      <td>5322.724490</td>
      <td>88.442177</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>590875</td>
      <td>1415443</td>
      <td>25250</td>
      <td>2051.300000</td>
      <td>2002.966102</td>
      <td>4798.111864</td>
      <td>85.593220</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>371589</td>
      <td>289794</td>
      <td>17695</td>
      <td>1388.600000</td>
      <td>1987.106952</td>
      <td>1549.700535</td>
      <td>94.625668</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>23363806</td>
      <td>101645832</td>
      <td>1053980</td>
      <td>81100.135704</td>
      <td>1994.519891</td>
      <td>8677.294861</td>
      <td>89.976097</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>5307045</td>
      <td>13236409</td>
      <td>237129</td>
      <td>19930.800000</td>
      <td>1996.630926</td>
      <td>4979.837848</td>
      <td>89.213318</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>302181</td>
      <td>144596</td>
      <td>11459</td>
      <td>1081.200000</td>
      <td>2001.198675</td>
      <td>957.589404</td>
      <td>75.887417</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>886685</td>
      <td>1486383</td>
      <td>40271</td>
      <td>3112.100000</td>
      <td>2001.546275</td>
      <td>3355.266366</td>
      <td>90.905192</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>226476</td>
      <td>258085</td>
      <td>10458</td>
      <td>819.100000</td>
      <td>2004.212389</td>
      <td>2283.938053</td>
      <td>92.548673</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2694871</td>
      <td>8759930</td>
      <td>134225</td>
      <td>8523.200000</td>
      <td>2008.100596</td>
      <td>6527.518629</td>
      <td>100.018629</td>
      <td>6.351118</td>
    </tr>
    <tr>
      <th>All</th>
      <td>76210353</td>
      <td>239235500</td>
      <td>3394146</td>
      <td>264162.435704</td>
      <td>1996.969656</td>
      <td>6268.781280</td>
      <td>88.938134</td>
      <td>6.921952</td>
    </tr>
  </tbody>
</table>
此处的All便是总和数据

### 3.9.7 对不同值执行不同函数：传递字典
**对不同值执行不同的函数：可以向aggfunc传递一个字典。不过，这样做有一个副作用，那就是必须将标签做的更加整洁才行。**

```python
pd.pivot_table(df, index = ["产地"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0)
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>5237466</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>41380993</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>23179448</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>394745</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>486082</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3048849</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1362581</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>1146173</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>139462</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>357027</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2679856</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>2500842</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>17981631</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>170449</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>10208966</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>159577</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>1564881</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>1415443</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>289794</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>101645832</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>13236409</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>144596</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>1486383</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>258085</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>8759930</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>


对各个年份的投票人数求和，对评分求均值


```python
pd.pivot_table(df, index = ["年代"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0, margins = True)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>776</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>338</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>7679</td>
      <td>7.575000</td>
    </tr>
    <tr>
      <th>1896</th>
      <td>7874</td>
      <td>7.037500</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>201</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1898</th>
      <td>1157</td>
      <td>7.450000</td>
    </tr>
    <tr>
      <th>1899</th>
      <td>142</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>1227</td>
      <td>7.228571</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>658</td>
      <td>7.250000</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>11548</td>
      <td>7.680000</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>4548</td>
      <td>7.015385</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>1495</td>
      <td>7.616667</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>1663</td>
      <td>6.820000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>1329</td>
      <td>7.342857</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>1068</td>
      <td>7.020000</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>1475</td>
      <td>7.425000</td>
    </tr>
    <tr>
      <th>1909</th>
      <td>248</td>
      <td>7.650000</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>526</td>
      <td>6.940000</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>924</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>724</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>1913</th>
      <td>436</td>
      <td>6.671429</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>2728</td>
      <td>6.473077</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>6298</td>
      <td>7.260000</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>7333</td>
      <td>7.690909</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>1493</td>
      <td>7.075000</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>4285</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>3506</td>
      <td>7.494444</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>8911</td>
      <td>7.492857</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>16056</td>
      <td>7.750000</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>16876</td>
      <td>7.804545</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>7611</td>
      <td>7.811765</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>10382</td>
      <td>8.059259</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>27607</td>
      <td>7.788000</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>10203</td>
      <td>7.773913</td>
    </tr>
    <tr>
      <th>1927</th>
      <td>20163</td>
      <td>7.751724</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>16133</td>
      <td>7.964103</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>25919</td>
      <td>7.440000</td>
    </tr>
    <tr>
      <th>1930</th>
      <td>13877</td>
      <td>7.360000</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>61684</td>
      <td>7.483333</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>22203</td>
      <td>7.294595</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>36289</td>
      <td>7.420833</td>
    </tr>
    <tr>
      <th>1934</th>
      <td>39573</td>
      <td>7.536000</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>40834</td>
      <td>7.515217</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>87864</td>
      <td>7.615254</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>77381</td>
      <td>7.568750</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>20976</td>
      <td>7.736842</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>289681</td>
      <td>7.520408</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>216213</td>
      <td>7.544737</td>
    </tr>
    <tr>
      <th>1941</th>
      <td>72982</td>
      <td>7.427660</td>
    </tr>
    <tr>
      <th>1942</th>
      <td>138211</td>
      <td>7.554717</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>42300</td>
      <td>7.605357</td>
    </tr>
    <tr>
      <th>1944</th>
      <td>54398</td>
      <td>7.538889</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>48462</td>
      <td>7.673469</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>61033</td>
      <td>7.606780</td>
    </tr>
    <tr>
      <th>1947</th>
      <td>32834</td>
      <td>7.502703</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>103136</td>
      <td>7.645349</td>
    </tr>
    <tr>
      <th>1949</th>
      <td>57110</td>
      <td>7.646067</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>169862</td>
      <td>7.655263</td>
    </tr>
    <tr>
      <th>1951</th>
      <td>75495</td>
      <td>7.637179</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>128121</td>
      <td>7.775294</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>421134</td>
      <td>7.579310</td>
    </tr>
    <tr>
      <th>1954</th>
      <td>222337</td>
      <td>7.714851</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>228056</td>
      <td>7.567544</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>153943</td>
      <td>7.591724</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>317989</td>
      <td>7.622115</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>108116</td>
      <td>7.536885</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>220809</td>
      <td>7.571875</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>189424</td>
      <td>7.580769</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>324975</td>
      <td>7.741000</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>281964</td>
      <td>7.707042</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>169316</td>
      <td>7.536364</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>201454</td>
      <td>7.540782</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>323972</td>
      <td>7.591304</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>236142</td>
      <td>7.521910</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>217681</td>
      <td>7.477143</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>188333</td>
      <td>7.324706</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>122661</td>
      <td>7.367647</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>128492</td>
      <td>7.294149</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>299585</td>
      <td>7.149309</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>475755</td>
      <td>7.253431</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>170270</td>
      <td>7.238095</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>375884</td>
      <td>7.063964</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>435544</td>
      <td>7.056995</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>312364</td>
      <td>7.107725</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>210467</td>
      <td>7.151376</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>201366</td>
      <td>7.096517</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>426270</td>
      <td>7.292857</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>539000</td>
      <td>7.182063</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>440794</td>
      <td>7.154044</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>569800</td>
      <td>7.286260</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>426771</td>
      <td>7.296727</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>784106</td>
      <td>7.380887</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>509528</td>
      <td>7.267722</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>916279</td>
      <td>7.249536</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>1073070</td>
      <td>7.282111</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>1875710</td>
      <td>7.265544</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>1176669</td>
      <td>7.199233</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>1951292</td>
      <td>7.156923</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>1921033</td>
      <td>7.154412</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>2385473</td>
      <td>7.190187</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>3579850</td>
      <td>7.186560</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>5668165</td>
      <td>7.257260</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>3976373</td>
      <td>7.275052</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>2634034</td>
      <td>7.249119</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>4426584</td>
      <td>7.325758</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>4286770</td>
      <td>7.223540</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>4459724</td>
      <td>7.171151</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>4317784</td>
      <td>7.112413</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>7027000</td>
      <td>7.070647</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>5894532</td>
      <td>7.045620</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>7480686</td>
      <td>7.114410</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>9590716</td>
      <td>6.999909</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>8209986</td>
      <td>7.011864</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>11022718</td>
      <td>6.901546</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>10693163</td>
      <td>6.859703</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>13250500</td>
      <td>6.892164</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>15972119</td>
      <td>6.732825</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>18044725</td>
      <td>6.752793</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>16347306</td>
      <td>6.560325</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>14460484</td>
      <td>6.444896</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>15211249</td>
      <td>6.375974</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>14570038</td>
      <td>6.249384</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>12481394</td>
      <td>6.121925</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1850030</td>
      <td>5.834524</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>123456</td>
      <td>6.935704</td>
    </tr>
    <tr>
      <th>All</th>
      <td>239235500</td>
      <td>6.921952</td>
    </tr>
  </tbody>
</table>


### 3.9.8 透视表过滤

```python
table = pd.pivot_table(df, index = ["年代"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0)
```


```python
type(table)
```
pandas.core.frame.DataFrame


```python
table[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>776</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>338</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>7679</td>
      <td>7.575000</td>
    </tr>
  </tbody>
</table>


**1994年被誉为电影史上最伟大的一年，但是通过数据我们可以发现，1994年的平均得分其实并不是很高。1924年的电影均分最高。**


```python
table[table.index == 1994]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994</th>
      <td>5668165</td>
      <td>7.25726</td>
    </tr>
  </tbody>
</table>

```python
table.sort_values("评分", ascending = False)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1924</th>
      <td>10382</td>
      <td>8.059259</td>
    </tr>
    <tr>
      <th>1928</th>
      <td>16133</td>
      <td>7.964103</td>
    </tr>
    <tr>
      <th>1888</th>
      <td>776</td>
      <td>7.950000</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>724</td>
      <td>7.925000</td>
    </tr>
    <tr>
      <th>1923</th>
      <td>7611</td>
      <td>7.811765</td>
    </tr>
    <tr>
      <th>1922</th>
      <td>16876</td>
      <td>7.804545</td>
    </tr>
    <tr>
      <th>1925</th>
      <td>27607</td>
      <td>7.788000</td>
    </tr>
    <tr>
      <th>1952</th>
      <td>128121</td>
      <td>7.775294</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>10203</td>
      <td>7.773913</td>
    </tr>
    <tr>
      <th>1927</th>
      <td>20163</td>
      <td>7.751724</td>
    </tr>
    <tr>
      <th>1921</th>
      <td>16056</td>
      <td>7.750000</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>324975</td>
      <td>7.741000</td>
    </tr>
    <tr>
      <th>1938</th>
      <td>20976</td>
      <td>7.736842</td>
    </tr>
    <tr>
      <th>1954</th>
      <td>222337</td>
      <td>7.714851</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>281964</td>
      <td>7.707042</td>
    </tr>
    <tr>
      <th>1916</th>
      <td>7333</td>
      <td>7.690909</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>11548</td>
      <td>7.680000</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>48462</td>
      <td>7.673469</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>169862</td>
      <td>7.655263</td>
    </tr>
    <tr>
      <th>1909</th>
      <td>248</td>
      <td>7.650000</td>
    </tr>
    <tr>
      <th>1949</th>
      <td>57110</td>
      <td>7.646067</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>103136</td>
      <td>7.645349</td>
    </tr>
    <tr>
      <th>1951</th>
      <td>75495</td>
      <td>7.637179</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>317989</td>
      <td>7.622115</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>1495</td>
      <td>7.616667</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>87864</td>
      <td>7.615254</td>
    </tr>
    <tr>
      <th>1946</th>
      <td>61033</td>
      <td>7.606780</td>
    </tr>
    <tr>
      <th>1943</th>
      <td>42300</td>
      <td>7.605357</td>
    </tr>
    <tr>
      <th>1956</th>
      <td>153943</td>
      <td>7.591724</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>323972</td>
      <td>7.591304</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>189424</td>
      <td>7.580769</td>
    </tr>
    <tr>
      <th>1953</th>
      <td>421134</td>
      <td>7.579310</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>7679</td>
      <td>7.575000</td>
    </tr>
    <tr>
      <th>1959</th>
      <td>220809</td>
      <td>7.571875</td>
    </tr>
    <tr>
      <th>1937</th>
      <td>77381</td>
      <td>7.568750</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>228056</td>
      <td>7.567544</td>
    </tr>
    <tr>
      <th>1942</th>
      <td>138211</td>
      <td>7.554717</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>216213</td>
      <td>7.544737</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>201454</td>
      <td>7.540782</td>
    </tr>
    <tr>
      <th>1944</th>
      <td>54398</td>
      <td>7.538889</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>108116</td>
      <td>7.536885</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>169316</td>
      <td>7.536364</td>
    </tr>
    <tr>
      <th>1934</th>
      <td>39573</td>
      <td>7.536000</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>236142</td>
      <td>7.521910</td>
    </tr>
    <tr>
      <th>1939</th>
      <td>289681</td>
      <td>7.520408</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>40834</td>
      <td>7.515217</td>
    </tr>
    <tr>
      <th>1947</th>
      <td>32834</td>
      <td>7.502703</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>176</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>924</td>
      <td>7.500000</td>
    </tr>
    <tr>
      <th>1919</th>
      <td>3506</td>
      <td>7.494444</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>8911</td>
      <td>7.492857</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>61684</td>
      <td>7.483333</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>217681</td>
      <td>7.477143</td>
    </tr>
    <tr>
      <th>1898</th>
      <td>1157</td>
      <td>7.450000</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>25919</td>
      <td>7.440000</td>
    </tr>
    <tr>
      <th>1941</th>
      <td>72982</td>
      <td>7.427660</td>
    </tr>
    <tr>
      <th>1908</th>
      <td>1475</td>
      <td>7.425000</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>36289</td>
      <td>7.420833</td>
    </tr>
    <tr>
      <th>1984</th>
      <td>784106</td>
      <td>7.380887</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>122661</td>
      <td>7.367647</td>
    </tr>
    <tr>
      <th>1930</th>
      <td>13877</td>
      <td>7.360000</td>
    </tr>
    <tr>
      <th>1906</th>
      <td>1329</td>
      <td>7.342857</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>4426584</td>
      <td>7.325758</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>188333</td>
      <td>7.324706</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>426771</td>
      <td>7.296727</td>
    </tr>
    <tr>
      <th>1932</th>
      <td>22203</td>
      <td>7.294595</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>128492</td>
      <td>7.294149</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>426270</td>
      <td>7.292857</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>569800</td>
      <td>7.286260</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>1073070</td>
      <td>7.282111</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>3976373</td>
      <td>7.275052</td>
    </tr>
    <tr>
      <th>1985</th>
      <td>509528</td>
      <td>7.267722</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>1875710</td>
      <td>7.265544</td>
    </tr>
    <tr>
      <th>1915</th>
      <td>6298</td>
      <td>7.260000</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>5668165</td>
      <td>7.257260</td>
    </tr>
    <tr>
      <th>1972</th>
      <td>475755</td>
      <td>7.253431</td>
    </tr>
    <tr>
      <th>1901</th>
      <td>658</td>
      <td>7.250000</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>916279</td>
      <td>7.249536</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>2634034</td>
      <td>7.249119</td>
    </tr>
    <tr>
      <th>1973</th>
      <td>170270</td>
      <td>7.238095</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>1227</td>
      <td>7.228571</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>4286770</td>
      <td>7.223540</td>
    </tr>
    <tr>
      <th>1918</th>
      <td>4285</td>
      <td>7.200000</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>1176669</td>
      <td>7.199233</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>2385473</td>
      <td>7.190187</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>3579850</td>
      <td>7.186560</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>539000</td>
      <td>7.182063</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>4459724</td>
      <td>7.171151</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>1951292</td>
      <td>7.156923</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>1921033</td>
      <td>7.154412</td>
    </tr>
    <tr>
      <th>1981</th>
      <td>440794</td>
      <td>7.154044</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>210467</td>
      <td>7.151376</td>
    </tr>
    <tr>
      <th>1971</th>
      <td>299585</td>
      <td>7.149309</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>7480686</td>
      <td>7.114410</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>4317784</td>
      <td>7.112413</td>
    </tr>
    <tr>
      <th>1976</th>
      <td>312364</td>
      <td>7.107725</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>201366</td>
      <td>7.096517</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>1493</td>
      <td>7.075000</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>7027000</td>
      <td>7.070647</td>
    </tr>
    <tr>
      <th>1974</th>
      <td>375884</td>
      <td>7.063964</td>
    </tr>
    <tr>
      <th>1975</th>
      <td>435544</td>
      <td>7.056995</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>5894532</td>
      <td>7.045620</td>
    </tr>
    <tr>
      <th>1896</th>
      <td>7874</td>
      <td>7.037500</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>1068</td>
      <td>7.020000</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>4548</td>
      <td>7.015385</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>8209986</td>
      <td>7.011864</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>9590716</td>
      <td>6.999909</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>526</td>
      <td>6.940000</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>123456</td>
      <td>6.935704</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>11022718</td>
      <td>6.901546</td>
    </tr>
    <tr>
      <th>1899</th>
      <td>142</td>
      <td>6.900000</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>13250500</td>
      <td>6.892164</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>10693163</td>
      <td>6.859703</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>1663</td>
      <td>6.820000</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>18044725</td>
      <td>6.752793</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>15972119</td>
      <td>6.732825</td>
    </tr>
    <tr>
      <th>1913</th>
      <td>436</td>
      <td>6.671429</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>338</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>1897</th>
      <td>201</td>
      <td>6.633333</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>16347306</td>
      <td>6.560325</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>2728</td>
      <td>6.473077</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>14460484</td>
      <td>6.444896</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>15211249</td>
      <td>6.375974</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>14570038</td>
      <td>6.249384</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>12481394</td>
      <td>6.121925</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1850030</td>
      <td>5.834524</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>51</td>
      <td>4.800000</td>
    </tr>
  </tbody>
</table>


**同样的，我们也可以按照多个索引来进行汇总。**


```python
pd.pivot_table(df, index = ["产地", "年代"], values = ["投票人数", "评分"], aggfunc = {"投票人数":np.sum, "评分":np.mean}, fill_value = 0)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th>年代</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">中国台湾</th>
      <th>1963</th>
      <td>121</td>
      <td>6.400000</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>461</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>51</td>
      <td>7.900000</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>4444</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>178</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">韩国</th>
      <th>2012</th>
      <td>610317</td>
      <td>6.035238</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>1130800</td>
      <td>6.062037</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>453152</td>
      <td>5.650833</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>349808</td>
      <td>5.423853</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>34797</td>
      <td>5.730000</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 2 columns</p>




## 3.10 数据重塑和轴向旋转

### 3.10.1 层次化索引

层次化索引是pandas的一项重要功能，它能使我们在一个轴上拥有多个索引。

#### 3.10.1.1 Series的层次化索引：

```python
s = pd.Series(np.arange(1,10), index = [['a','a','a','b','b','c','c','d','d'], [1,2,3,1,2,3,1,2,3]])
s #类似于合并单元格
```


    a  1    1
       2    2
       3    3
    b  1    4
       2    5
    c  3    6
       1    7
    d  2    8
       3    9
    dtype: int32


```python
s.index
```


    MultiIndex([('a', 1),
                ('a', 2),
                ('a', 3),
                ('b', 1),
                ('b', 2),
                ('c', 3),
                ('c', 1),
                ('d', 2),
                ('d', 3)],
               )




```python
s['a'] #外层索引
```


    1    1
    2    2
    3    3
    dtype: int32




```python
s['a':'c'] #切片
```


    a  1    1
       2    2
       3    3
    b  1    4
       2    5
    c  3    6
       1    7
    dtype: int32




```python
s[:,1] #内层索引
```


    a    1
    b    4
    c    7
    dtype: int32




```python
s['c',3] #提取具体的值
```


    6

#### 3.10.1.2 通过unstack方法将Series变成DataFrame

```python
s.unstack()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>



```python
s.unstack().stack() #形式上的相互转换 具体的应用？还是说整花活
```


    a  1    1.0
       2    2.0
       3    3.0
    b  1    4.0
       2    5.0
    c  1    7.0
       3    6.0
    d  2    8.0
       3    9.0
    dtype: float64



#### 3.10.1.3 Dataframe的层次化索引： 

对于DataFrame来说，行和列都能进行层次化索引。


```python
data = pd.DataFrame(np.arange(12).reshape(4,3), index = [['a','a','b','b'],[1,2,1,2]], columns = [['A','A','B'],['Z','X','C']])
data
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">A</th>
      <th>B</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>Z</th>
      <th>X</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>



```python
data['A']
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Z</th>
      <th>X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
    </tr>
  </tbody>
</table>



```python
data.index.names = ["row1","row2"]
data.columns.names = ["col1", "col2"]
data
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>col1</th>
      <th colspan="2" halign="left">A</th>
      <th>B</th>
    </tr>
    <tr>
      <th></th>
      <th>col2</th>
      <th>Z</th>
      <th>X</th>
      <th>C</th>
    </tr>
    <tr>
      <th>row1</th>
      <th>row2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
这样位置就不对了，需要调整




```python
data.swaplevel("row1","row2") #位置调整，交换
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>col1</th>
      <th colspan="2" halign="left">A</th>
      <th>B</th>
    </tr>
    <tr>
      <th></th>
      <th>col2</th>
      <th>Z</th>
      <th>X</th>
      <th>C</th>
    </tr>
    <tr>
      <th>row2</th>
      <th>row1</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <th>a</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <th>a</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <th>b</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <th>b</th>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
  </tbody>
</table>




#### 3.10.1.4 将电影数据处理成多层索引的结构

```python
df.index #默认索引
```


    Int64Index([    0,     1,     2,     3,     4,     5,     6,     7,     8,
                    9,
                ...
                38153, 38154, 38155, 38156, 38157, 38158, 38159, 38160, 38161,
                38162],
               dtype='int64', length=38163)

**把产地和年代同时设成索引，产地是外层索引，年代为内层索引。**

**set_index可以把列变成索引**

**reset_index是把索引变成列** 

```python
df = df.set_index(["产地", "年代"])
df
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>产地</th>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">美国</th>
      <th>1994</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.600000</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>意大利</th>
      <th>1997</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>美国</th>
      <th>1994</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.400000</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <th>1993</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.400000</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>美国</th>
      <th>1935</th>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">中国大陆</th>
      <th>1986</th>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.000000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <th>1977</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32</td>
      <td>剧情/战争</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>6.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>美国</th>
      <th>2018</th>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>38163 rows × 7 columns</p>


**每一个索引都是一个元组：**

```python
df.index[0]
```


    ('美国', 1994)



**获取所有的美国电影，由于产地信息已经变成了索引，因此要是用.loc方法。**

```python
df.loc["美国"] #行标签索引行数据，注意索引多行时两边都是闭区间
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.600000</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1957</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.400000</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.400000</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>辛德勒的名单</td>
      <td>306904</td>
      <td>剧情/历史/战争</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>9.400000</td>
      <td>华盛顿首映</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1987</th>
      <td>零下的激情</td>
      <td>199</td>
      <td>剧情/爱情/犯罪</td>
      <td>1987-11-06 00:00:00</td>
      <td>98</td>
      <td>7.400000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>离别秋波</td>
      <td>240</td>
      <td>剧情/爱情/音乐</td>
      <td>1986-02-19 00:00:00</td>
      <td>90</td>
      <td>8.200000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>极乐森林</td>
      <td>45</td>
      <td>纪录片</td>
      <td>1986-09-14 00:00:00</td>
      <td>90</td>
      <td>8.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>11714 rows × 7 columns</p>


```python
df.loc["中国大陆"]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1993</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>大闹天宫</td>
      <td>74881</td>
      <td>动画/奇幻</td>
      <td>1905-05-14 00:00:00</td>
      <td>114</td>
      <td>9.2</td>
      <td>上集</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>穹顶之下</td>
      <td>51113</td>
      <td>纪录片</td>
      <td>2015-02-28 00:00:00</td>
      <td>104</td>
      <td>9.2</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>茶馆</td>
      <td>10678</td>
      <td>剧情/历史</td>
      <td>1905-06-04 00:00:00</td>
      <td>118</td>
      <td>9.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1988</th>
      <td>山水情</td>
      <td>10781</td>
      <td>动画/短片</td>
      <td>1905-06-10 00:00:00</td>
      <td>19</td>
      <td>9.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>T省的八四、八五</td>
      <td>380</td>
      <td>剧情</td>
      <td>1905-06-08 00:00:00</td>
      <td>94</td>
      <td>8.7</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>失踪的女中学生</td>
      <td>101</td>
      <td>儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>102</td>
      <td>7.4</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>血战台儿庄</td>
      <td>2908</td>
      <td>战争</td>
      <td>1905-06-08 00:00:00</td>
      <td>120</td>
      <td>8.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.1</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.0</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>3791 rows × 7 columns</p>


**这样做的最大好处是我们可以简化很多的筛选环节**

**每一个索引是一个元组：** 

```python
df = df.swaplevel("产地", "年代") #调换标签顺序
df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>年代</th>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1994</th>
      <th>美国</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.600000</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1957</th>
      <th>美国</th>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1997</th>
      <th>意大利</th>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>1994</th>
      <th>美国</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.400000</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>1993</th>
      <th>中国大陆</th>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.400000</td>
      <td>香港</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1935</th>
      <th>美国</th>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1986</th>
      <th>中国大陆</th>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.100000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.000000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>1977</th>
      <th>俄罗斯</th>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32</td>
      <td>剧情/战争</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>6.600000</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2018</th>
      <th>美国</th>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>38163 rows × 7 columns</p>



```python
df.loc[1994]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>美国</th>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>9.4</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>34街的</td>
      <td>768</td>
      <td>剧情/家庭/奇幻</td>
      <td>1994-12-23 00:00:00</td>
      <td>114</td>
      <td>7.9</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>活着</td>
      <td>202794</td>
      <td>剧情/家庭</td>
      <td>1994-05-18 00:00:00</td>
      <td>132</td>
      <td>9.0</td>
      <td>法国</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>鬼精灵2： 恐怖</td>
      <td>60</td>
      <td>喜剧/恐怖/奇幻</td>
      <td>1994-04-08 00:00:00</td>
      <td>85</td>
      <td>5.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>黑色第16</td>
      <td>44</td>
      <td>剧情/惊悚</td>
      <td>1996-02-01 00:00:00</td>
      <td>106</td>
      <td>6.8</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>蜡笔小新之布里布里王国的秘密宝藏 クレヨンしんちゃん ブリブリ王国の</td>
      <td>2142</td>
      <td>动画</td>
      <td>1994-04-23 00:00:00</td>
      <td>94</td>
      <td>7.7</td>
      <td>日本</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>龙珠Z剧场版10：两人面临危机! 超战士难以成眠 ドラゴンボール Z 劇場版：危険なふたり！</td>
      <td>579</td>
      <td>动画</td>
      <td>1994-03-12 00:00:00</td>
      <td>53</td>
      <td>7.2</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>重案实录之惊天械劫案 重案實錄之驚天械劫</td>
      <td>90</td>
      <td>动作/犯罪</td>
      <td>1905-06-16 00:00:00</td>
      <td>114</td>
      <td>7.3</td>
      <td>美国</td>
    </tr>
  </tbody>
</table>
<p>489 rows × 7 columns</p>


#### 3.10.1.5 取消层次化索引

```python
df = df.reset_index()
df[:5]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


### 3.10.2 数据旋转 

行列转化：以前5部电影为例


```python
data = df[:5]
data
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


.T可以直接让数据的行列进行交换


```python
data.T #似曾相识,Numpy的矩阵转置
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>年代</th>
      <td>1994</td>
      <td>1957</td>
      <td>1997</td>
      <td>1994</td>
      <td>1993</td>
    </tr>
    <tr>
      <th>产地</th>
      <td>美国</td>
      <td>美国</td>
      <td>意大利</td>
      <td>美国</td>
      <td>中国大陆</td>
    </tr>
    <tr>
      <th>名字</th>
      <td>肖申克的救赎</td>
      <td>控方证人</td>
      <td>美丽人生</td>
      <td>阿甘正传</td>
      <td>霸王别姬</td>
    </tr>
    <tr>
      <th>投票人数</th>
      <td>692795</td>
      <td>42995</td>
      <td>327855</td>
      <td>580897</td>
      <td>478523</td>
    </tr>
    <tr>
      <th>类型</th>
      <td>剧情/犯罪</td>
      <td>剧情/悬疑/犯罪</td>
      <td>剧情/喜剧/爱情</td>
      <td>剧情/爱情</td>
      <td>剧情/爱情/同性</td>
    </tr>
    <tr>
      <th>上映时间</th>
      <td>1994-09-10 00:00:00</td>
      <td>1957-12-17 00:00:00</td>
      <td>1997-12-20 00:00:00</td>
      <td>1994-06-23 00:00:00</td>
      <td>1993-01-01 00:00:00</td>
    </tr>
    <tr>
      <th>时长</th>
      <td>142</td>
      <td>116</td>
      <td>116</td>
      <td>142</td>
      <td>171</td>
    </tr>
    <tr>
      <th>评分</th>
      <td>9.6</td>
      <td>9.5</td>
      <td>9.5</td>
      <td>9.4</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>首映地点</th>
      <td>多伦多电影节</td>
      <td>美国</td>
      <td>意大利</td>
      <td>洛杉矶首映</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


**dataframe也可以使用stack和unstack，转化为层次化索引的Series** 

```python
data.stack()
```


    0  年代                     1994
       产地                       美国
       名字                   肖申克的救赎
       投票人数                 692795
       类型                    剧情/犯罪
       上映时间    1994-09-10 00:00:00
       时长                      142
       评分                      9.6
       首映地点                 多伦多电影节
    1  年代                     1957
       产地                       美国
       名字                     控方证人
       投票人数                  42995
       类型                 剧情/悬疑/犯罪
       上映时间    1957-12-17 00:00:00
       时长                      116
       评分                      9.5
       首映地点                     美国
    2  年代                     1997
       产地                      意大利
       名字                    美丽人生 
       投票人数                 327855
       类型                 剧情/喜剧/爱情
       上映时间    1997-12-20 00:00:00
       时长                      116
       评分                      9.5
       首映地点                    意大利
    3  年代                     1994
       产地                       美国
       名字                     阿甘正传
       投票人数                 580897
       类型                    剧情/爱情
       上映时间    1994-06-23 00:00:00
       时长                      142
       评分                      9.4
       首映地点                  洛杉矶首映
    4  年代                     1993
       产地                     中国大陆
       名字                     霸王别姬
       投票人数                 478523
       类型                 剧情/爱情/同性
       上映时间    1993-01-01 00:00:00
       时长                      171
       评分                      9.4
       首映地点                     香港
    dtype: object




```python
data.stack().unstack()  #转回来
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
    </tr>
  </tbody>
</table>


## 3.11 数据分组，分组运算

### 3.11.1 GroupBy技术：实现数据的分组，和分组运算，作用类似于数据透视表

按照电影的产地进行分组 


```python
group = df.groupby(df["产地"])
```

### 3.11.2 先定义一个分组变量group 


```python
type(group)
```


    pandas.core.groupby.generic.DataFrameGroupBy



### 3.11.3 计算分组后各个的统计量 


```python
group.mean() 
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1999.009709</td>
      <td>8474.864078</td>
      <td>87.257282</td>
      <td>7.066667</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>2004.582432</td>
      <td>10915.587708</td>
      <td>81.517014</td>
      <td>6.062991</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>1991.088865</td>
      <td>8141.709870</td>
      <td>88.553214</td>
      <td>6.473551</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>1999.091371</td>
      <td>2003.781726</td>
      <td>88.507614</td>
      <td>7.246701</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>1984.892857</td>
      <td>1021.180672</td>
      <td>96.100840</td>
      <td>7.557143</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>1998.721721</td>
      <td>1619.144450</td>
      <td>87.656399</td>
      <td>7.226713</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>2002.520451</td>
      <td>1921.834979</td>
      <td>80.592384</td>
      <td>6.727221</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>2006.039326</td>
      <td>3219.587079</td>
      <td>120.949438</td>
      <td>6.864888</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>1992.786325</td>
      <td>1191.982906</td>
      <td>92.641026</td>
      <td>7.085470</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>1999.888889</td>
      <td>3606.333333</td>
      <td>88.373737</td>
      <td>7.232323</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>1996.053869</td>
      <td>2624.736533</td>
      <td>92.258570</td>
      <td>7.187365</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1985.599190</td>
      <td>3374.955466</td>
      <td>104.333333</td>
      <td>7.183131</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>1999.886536</td>
      <td>3592.015781</td>
      <td>85.010587</td>
      <td>7.192569</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>1999.503650</td>
      <td>1244.153285</td>
      <td>83.065693</td>
      <td>7.197080</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>1991.794044</td>
      <td>3663.066380</td>
      <td>90.249013</td>
      <td>7.243093</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>1987.027624</td>
      <td>881.640884</td>
      <td>80.734807</td>
      <td>7.441989</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>2009.129252</td>
      <td>5322.724490</td>
      <td>88.442177</td>
      <td>6.109184</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>2002.966102</td>
      <td>4798.111864</td>
      <td>85.593220</td>
      <td>6.953559</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>1987.106952</td>
      <td>1549.700535</td>
      <td>94.625668</td>
      <td>7.425668</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>1994.519891</td>
      <td>8677.294861</td>
      <td>89.976097</td>
      <td>6.923351</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>1996.630926</td>
      <td>4979.837848</td>
      <td>89.213318</td>
      <td>7.498420</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>2001.198675</td>
      <td>957.589404</td>
      <td>75.887417</td>
      <td>7.160265</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>2001.546275</td>
      <td>3355.266366</td>
      <td>90.905192</td>
      <td>7.025056</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>2004.212389</td>
      <td>2283.938053</td>
      <td>92.548673</td>
      <td>7.248673</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2008.100596</td>
      <td>6527.518629</td>
      <td>100.018629</td>
      <td>6.351118</td>
    </tr>
  </tbody>
</table>



```python
group.sum()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>1235388</td>
      <td>5237466</td>
      <td>53925</td>
      <td>4367.200000</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>7599372</td>
      <td>41380993</td>
      <td>309031</td>
      <td>22984.800000</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>5668630</td>
      <td>23179448</td>
      <td>252111</td>
      <td>18430.200000</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>393821</td>
      <td>394745</td>
      <td>17436</td>
      <td>1427.600000</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>944809</td>
      <td>486082</td>
      <td>45744</td>
      <td>3597.200000</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>3763593</td>
      <td>3048849</td>
      <td>165057</td>
      <td>13607.900000</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>1419787</td>
      <td>1362581</td>
      <td>57140</td>
      <td>4769.600000</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>714150</td>
      <td>1146173</td>
      <td>43058</td>
      <td>2443.900000</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>233156</td>
      <td>139462</td>
      <td>10839</td>
      <td>829.000000</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>197989</td>
      <td>357027</td>
      <td>8749</td>
      <td>716.000000</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>2037971</td>
      <td>2679856</td>
      <td>94196</td>
      <td>7338.300000</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>1471329</td>
      <td>2500842</td>
      <td>77311</td>
      <td>5322.700000</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>10011432</td>
      <td>17981631</td>
      <td>425563</td>
      <td>36006.000000</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>273932</td>
      <td>170449</td>
      <td>11380</td>
      <td>986.000000</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>5551130</td>
      <td>10208966</td>
      <td>251524</td>
      <td>20186.500000</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>359652</td>
      <td>159577</td>
      <td>14613</td>
      <td>1347.000000</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>590684</td>
      <td>1564881</td>
      <td>26002</td>
      <td>1796.100000</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>590875</td>
      <td>1415443</td>
      <td>25250</td>
      <td>2051.300000</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>371589</td>
      <td>289794</td>
      <td>17695</td>
      <td>1388.600000</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>23363806</td>
      <td>101645832</td>
      <td>1053980</td>
      <td>81100.135704</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>5307045</td>
      <td>13236409</td>
      <td>237129</td>
      <td>19930.800000</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>302181</td>
      <td>144596</td>
      <td>11459</td>
      <td>1081.200000</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>886685</td>
      <td>1486383</td>
      <td>40271</td>
      <td>3112.100000</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>226476</td>
      <td>258085</td>
      <td>10458</td>
      <td>819.100000</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>2694871</td>
      <td>8759930</td>
      <td>134225</td>
      <td>8523.200000</td>
    </tr>
  </tbody>
</table>



### 3.11.4 计算每年的平均评分 


```python
df["评分"].groupby(df["年代"]).mean()
```


    年代
    1888    7.950000
    1890    4.800000
    1892    7.500000
    1894    6.633333
    1895    7.575000
              ...   
    2013    6.375974
    2014    6.249384
    2015    6.121925
    2016    5.834524
    2018    6.935704
    Name: 评分, Length: 127, dtype: float64



### 3.11.5 只对数值变量进行分组运算 


```python
df["年代"] = df["年代"].astype("str")
df.groupby(df["产地"]).median() #不会再对年代进行求取
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>中国台湾</th>
      <td>487.0</td>
      <td>92.0</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>中国大陆</th>
      <td>502.0</td>
      <td>90.0</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>中国香港</th>
      <td>637.0</td>
      <td>92.0</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>丹麦</th>
      <td>182.0</td>
      <td>94.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>俄罗斯</th>
      <td>132.5</td>
      <td>93.0</td>
      <td>7.7</td>
    </tr>
    <tr>
      <th>其他</th>
      <td>158.0</td>
      <td>90.0</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>加拿大</th>
      <td>258.0</td>
      <td>89.0</td>
      <td>6.8</td>
    </tr>
    <tr>
      <th>印度</th>
      <td>139.0</td>
      <td>131.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>墨西哥</th>
      <td>183.0</td>
      <td>94.0</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>巴西</th>
      <td>131.0</td>
      <td>96.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>德国</th>
      <td>212.0</td>
      <td>94.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>意大利</th>
      <td>187.0</td>
      <td>101.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>日本</th>
      <td>359.0</td>
      <td>89.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>比利时</th>
      <td>226.0</td>
      <td>90.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>法国</th>
      <td>244.0</td>
      <td>95.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>波兰</th>
      <td>174.0</td>
      <td>87.0</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>泰国</th>
      <td>542.5</td>
      <td>92.5</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>澳大利亚</th>
      <td>323.0</td>
      <td>95.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>瑞典</th>
      <td>191.0</td>
      <td>96.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>美国</th>
      <td>415.0</td>
      <td>93.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>英国</th>
      <td>345.0</td>
      <td>92.0</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>荷兰</th>
      <td>180.0</td>
      <td>85.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>西班牙</th>
      <td>267.0</td>
      <td>97.0</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>阿根廷</th>
      <td>146.0</td>
      <td>97.0</td>
      <td>7.3</td>
    </tr>
    <tr>
      <th>韩国</th>
      <td>1007.0</td>
      <td>104.0</td>
      <td>6.5</td>
    </tr>
  </tbody>
</table>




### 3.11.6 传入多个分组变量 


```python
df.groupby([df["产地"],df["年代"]]).mean() #根据两个变量进行分组
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>投票人数</th>
      <th>时长</th>
      <th>评分</th>
    </tr>
    <tr>
      <th>产地</th>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">中国台湾</th>
      <th>1963</th>
      <td>121.000000</td>
      <td>113.000000</td>
      <td>6.400000</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>153.666667</td>
      <td>105.000000</td>
      <td>6.800000</td>
    </tr>
    <tr>
      <th>1966</th>
      <td>51.000000</td>
      <td>60.000000</td>
      <td>7.900000</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>4444.000000</td>
      <td>112.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>1968</th>
      <td>89.000000</td>
      <td>83.000000</td>
      <td>7.400000</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">韩国</th>
      <th>2012</th>
      <td>5812.542857</td>
      <td>100.771429</td>
      <td>6.035238</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>10470.370370</td>
      <td>97.731481</td>
      <td>6.062037</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>3776.266667</td>
      <td>98.666667</td>
      <td>5.650833</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>3209.247706</td>
      <td>100.266055</td>
      <td>5.423853</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>1739.850000</td>
      <td>106.100000</td>
      <td>5.730000</td>
    </tr>
  </tbody>
</table>
<p>1578 rows × 3 columns</p>




### 3.11.7 获得每个地区，每一年的电影的评分的均值 


```python
group = df["评分"].groupby([df["产地"], df["年代"]])
means = group.mean()
means
```


    产地    年代  
    中国台湾  1963    6.400000
          1965    6.800000
          1966    7.900000
          1967    8.000000
          1968    7.400000
                    ...   
    韩国    2012    6.035238
          2013    6.062037
          2014    5.650833
          2015    5.423853
          2016    5.730000
    Name: 评分, Length: 1578, dtype: float64




```python
means = group = df["评分"].groupby([df["产地"], df["年代"]]).mean()
means
```


    产地    年代  
    中国台湾  1963    6.400000
          1965    6.800000
          1966    7.900000
          1967    8.000000
          1968    7.400000
                    ...   
    韩国    2012    6.035238
          2013    6.062037
          2014    5.650833
          2015    5.423853
          2016    5.730000
    Name: 评分, Length: 1578, dtype: float64



### 3.11.8 Series通过unstack方法转化为dataframe

**会产生缺失值**


```python
means.unstack().T
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>产地</th>
      <th>中国台湾</th>
      <th>中国大陆</th>
      <th>中国香港</th>
      <th>丹麦</th>
      <th>俄罗斯</th>
      <th>其他</th>
      <th>加拿大</th>
      <th>印度</th>
      <th>墨西哥</th>
      <th>巴西</th>
      <th>...</th>
      <th>波兰</th>
      <th>泰国</th>
      <th>澳大利亚</th>
      <th>瑞典</th>
      <th>美国</th>
      <th>英国</th>
      <th>荷兰</th>
      <th>西班牙</th>
      <th>阿根廷</th>
      <th>韩国</th>
    </tr>
    <tr>
      <th>年代</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1888</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.950000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.800000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1894</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.450000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>7.076471</td>
      <td>5.306500</td>
      <td>6.105714</td>
      <td>6.555556</td>
      <td>6.875000</td>
      <td>6.853571</td>
      <td>6.018182</td>
      <td>6.400000</td>
      <td>6.983333</td>
      <td>8.00</td>
      <td>...</td>
      <td>6.966667</td>
      <td>5.568000</td>
      <td>6.76000</td>
      <td>7.100</td>
      <td>6.308255</td>
      <td>7.460140</td>
      <td>6.33</td>
      <td>6.358333</td>
      <td>6.616667</td>
      <td>6.062037</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>6.522222</td>
      <td>4.963830</td>
      <td>5.616667</td>
      <td>7.120000</td>
      <td>7.175000</td>
      <td>6.596250</td>
      <td>5.921739</td>
      <td>6.374194</td>
      <td>7.250000</td>
      <td>6.86</td>
      <td>...</td>
      <td>7.060000</td>
      <td>5.653571</td>
      <td>6.56875</td>
      <td>6.960</td>
      <td>6.393056</td>
      <td>7.253398</td>
      <td>7.30</td>
      <td>6.868750</td>
      <td>7.150000</td>
      <td>5.650833</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>6.576000</td>
      <td>4.969189</td>
      <td>5.589189</td>
      <td>7.166667</td>
      <td>7.342857</td>
      <td>6.732727</td>
      <td>6.018750</td>
      <td>6.736364</td>
      <td>6.500000</td>
      <td>6.76</td>
      <td>...</td>
      <td>6.300000</td>
      <td>5.846667</td>
      <td>6.88000</td>
      <td>7.625</td>
      <td>6.231486</td>
      <td>7.123256</td>
      <td>6.70</td>
      <td>6.514286</td>
      <td>7.233333</td>
      <td>5.423853</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>NaN</td>
      <td>4.712000</td>
      <td>5.390909</td>
      <td>7.000000</td>
      <td>NaN</td>
      <td>6.833333</td>
      <td>6.200000</td>
      <td>6.900000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.522581</td>
      <td>7.200000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.730000</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.935704</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>127 rows × 25 columns</p>





## 3.12 离散化处理 

**在实际的数据分析项目中，对有的数据属性，我们往往并不关注数据的绝对取值，只关心它所处的区间或者等级**

**比如，我们可以把评分9分及以上的电影定义为A，7到9分定义为B，5到7分定义为C，3到5分定义为D，小于3分定义为E。**

**离散化也可称为分组、区间化。**

Pandas为我们提供了方便的函数**cut()**:

**pd.cut(x,bins,right = True,labels = None, retbins = False,precision = 3,include_lowest = False)** 

参数解释：

**x：需要离散化的数组、Series、DataFrame对象**

**bins：分组的依据，right = True，include_lowest = False，默认左开右闭，可以自己调整。**

**labels：是否要用标记来替换返回出来的数组，retbins：返回x当中每一个值对应的bins的列表，precision精度。**


```python
df["评分等级"] = pd.cut(df["评分"], [0,3,5,7,9,10], labels = ['E','D','C','B','A']) #labels要和区间划分一一对应
df
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.600000</td>
      <td>多伦多电影节</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>美国</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.500000</td>
      <td>意大利</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.400000</td>
      <td>洛杉矶首映</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.400000</td>
      <td>香港</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38158</th>
      <td>1935</td>
      <td>美国</td>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
      <td>B</td>
    </tr>
    <tr>
      <th>38159</th>
      <td>1986</td>
      <td>中国大陆</td>
      <td>血溅画屏</td>
      <td>95</td>
      <td>剧情/悬疑/犯罪/武侠/古装</td>
      <td>1905-06-08 00:00:00</td>
      <td>91</td>
      <td>7.100000</td>
      <td>美国</td>
      <td>B</td>
    </tr>
    <tr>
      <th>38160</th>
      <td>1986</td>
      <td>中国大陆</td>
      <td>魔窟中的幻想</td>
      <td>51</td>
      <td>惊悚/恐怖/儿童</td>
      <td>1905-06-08 00:00:00</td>
      <td>78</td>
      <td>8.000000</td>
      <td>美国</td>
      <td>B</td>
    </tr>
    <tr>
      <th>38161</th>
      <td>1977</td>
      <td>俄罗斯</td>
      <td>列宁格勒围困之星火战役 Блокада: Фильм 2: Ленинградский ме...</td>
      <td>32</td>
      <td>剧情/战争</td>
      <td>1905-05-30 00:00:00</td>
      <td>97</td>
      <td>6.600000</td>
      <td>美国</td>
      <td>C</td>
    </tr>
    <tr>
      <th>38162</th>
      <td>2018</td>
      <td>美国</td>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>38163 rows × 10 columns</p>


**同样的，我们可以根据投票人数来刻画电影的热门**

**投票越多的热门程度越高**


```python
bins = np.percentile(df["投票人数"], [0,20,40,60,80,100]) #获取分位数
df["热门程度"] = pd.cut(df["投票人数"],bins,labels = ['E','D','C','B','A'])
df[:5]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
    </tr>
  </tbody>
</table>




**大烂片集合：投票人数很多，评分很低**

**遗憾的是，我们可以发现，烂片几乎都是中国大陆的**


```python
df[(df.热门程度 == 'A') & (df.评分等级 == 'E')]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>623</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>B区</td>
      <td>5187</td>
      <td>剧情/惊悚/恐怖</td>
      <td>2011-06-03 00:00:00</td>
      <td>89</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4167</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>怖偶</td>
      <td>4867</td>
      <td>悬疑/惊悚</td>
      <td>2014-05-07 00:00:00</td>
      <td>88</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5200</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>床下有人</td>
      <td>4309</td>
      <td>悬疑/惊悚</td>
      <td>2011-10-14 00:00:00</td>
      <td>100</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>6585</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>帝国秘符</td>
      <td>4351</td>
      <td>动作/冒险</td>
      <td>2013-09-18 00:00:00</td>
      <td>93</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8009</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>飞天</td>
      <td>4764</td>
      <td>剧情</td>
      <td>2011-07-01 00:00:00</td>
      <td>115</td>
      <td>2.9</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8181</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>分手达人</td>
      <td>3937</td>
      <td>喜剧/爱情</td>
      <td>2014-06-06 00:00:00</td>
      <td>90</td>
      <td>2.7</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>9372</th>
      <td>2012</td>
      <td>中国大陆</td>
      <td>孤岛惊魂</td>
      <td>2982</td>
      <td>悬疑/惊悚/恐怖</td>
      <td>2013-01-26 00:00:00</td>
      <td>93</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>10275</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>海天盛宴·韦口</td>
      <td>3788</td>
      <td>情色</td>
      <td>2013-10-12 00:00:00</td>
      <td>88</td>
      <td>2.9</td>
      <td>网络</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>16512</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>孪生密码</td>
      <td>6390</td>
      <td>动作/悬疑</td>
      <td>2013-11-08 00:00:00</td>
      <td>96</td>
      <td>2.9</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>21189</th>
      <td>2010</td>
      <td>日本</td>
      <td>拳皇</td>
      <td>6329</td>
      <td>动作/科幻/冒险</td>
      <td>2012-10-12 00:00:00</td>
      <td>93</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>22348</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>闪魂</td>
      <td>3119</td>
      <td>惊悚/犯罪</td>
      <td>2014-02-21 00:00:00</td>
      <td>94</td>
      <td>2.6</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>22524</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>少年毛泽东</td>
      <td>3058</td>
      <td>动画/儿童/冒险</td>
      <td>2015-04-30 00:00:00</td>
      <td>76</td>
      <td>2.4</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>23754</th>
      <td>2013</td>
      <td>英国</td>
      <td>史前怪兽</td>
      <td>3543</td>
      <td>动作/惊悚/冒险</td>
      <td>2014-01-01 00:00:00</td>
      <td>89</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>27832</th>
      <td>2011</td>
      <td>中国大陆</td>
      <td>无极限之危情速递</td>
      <td>6319</td>
      <td>喜剧/动作/爱情/冒险</td>
      <td>2011-08-12 00:00:00</td>
      <td>94</td>
      <td>2.8</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>31622</th>
      <td>2010</td>
      <td>中国大陆</td>
      <td>异度公寓</td>
      <td>3639</td>
      <td>惊悚</td>
      <td>2010-06-04 00:00:00</td>
      <td>93</td>
      <td>2.7</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>32007</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>英雄之战</td>
      <td>8359</td>
      <td>动作/爱情</td>
      <td>2014-03-21 00:00:00</td>
      <td>90</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>32180</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>咏春小龙</td>
      <td>8861</td>
      <td>剧情/动作</td>
      <td>2013-07-20 00:00:00</td>
      <td>90</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>32990</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>再爱一次好不好</td>
      <td>6999</td>
      <td>喜剧/爱情</td>
      <td>2014-04-11 00:00:00</td>
      <td>94</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38090</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>大话天仙</td>
      <td>21629</td>
      <td>喜剧/奇幻/古装</td>
      <td>2014-02-02 00:00:00</td>
      <td>91</td>
      <td>3.0</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38092</th>
      <td>2013</td>
      <td>中国大陆</td>
      <td>天机·富春山居图</td>
      <td>74709</td>
      <td>动作/冒险</td>
      <td>2013-06-09 00:00:00</td>
      <td>122</td>
      <td>2.9</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38093</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>特工艾米拉</td>
      <td>10852</td>
      <td>动作/悬疑</td>
      <td>2014-04-11 00:00:00</td>
      <td>96</td>
      <td>2.7</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38097</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>汽车人总动员</td>
      <td>12892</td>
      <td>喜剧/动画/冒险</td>
      <td>2015-07-03 00:00:00</td>
      <td>85</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38102</th>
      <td>2016</td>
      <td>中国大陆</td>
      <td>2016年中央电视台春节</td>
      <td>17328</td>
      <td>歌舞/真人秀</td>
      <td>2016-02-07 00:00:00</td>
      <td>280</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38108</th>
      <td>2014</td>
      <td>中国大陆</td>
      <td>放手爱</td>
      <td>29254</td>
      <td>喜剧/爱情</td>
      <td>2014-04-30 00:00:00</td>
      <td>93</td>
      <td>2.3</td>
      <td>中国大陆</td>
      <td>E</td>
      <td>A</td>
    </tr>
  </tbody>
</table>




**冷门高分电影**


```python
df[(df.热门程度 == 'E') & (df.评分等级 == 'A')]
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>563</th>
      <td>2011</td>
      <td>英国</td>
      <td>BBC喜剧音</td>
      <td>38</td>
      <td>喜剧/音乐/歌舞</td>
      <td>2011-08-13 00:00:00</td>
      <td>95</td>
      <td>9.3</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>895</th>
      <td>2014</td>
      <td>日本</td>
      <td>JOJO的奇妙冒险 特别见面会 Walk Like Crusade</td>
      <td>36</td>
      <td>纪录片</td>
      <td>2014-10-26 00:00:00</td>
      <td>137</td>
      <td>9.3</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>2012</td>
      <td>英国</td>
      <td>Pond一家最</td>
      <td>45</td>
      <td>纪录片</td>
      <td>2012-09-29 00:00:00</td>
      <td>12</td>
      <td>9.2</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1540</th>
      <td>2007</td>
      <td>英国</td>
      <td>阿森纳：温格的十一人</td>
      <td>74</td>
      <td>运动</td>
      <td>2007-10-22 00:00:00</td>
      <td>78</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1547</th>
      <td>2009</td>
      <td>英国</td>
      <td>阿斯加德远征</td>
      <td>59</td>
      <td>纪录片</td>
      <td>2011-09-17 00:00:00</td>
      <td>85</td>
      <td>9.3</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>36846</th>
      <td>2012</td>
      <td>中国大陆</td>
      <td>末了，未了</td>
      <td>34</td>
      <td>剧情/喜剧/爱情</td>
      <td>2012-12-16 00:00:00</td>
      <td>90</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37000</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>身经百战</td>
      <td>74</td>
      <td>纪录片</td>
      <td>2015-03-24 00:00:00</td>
      <td>91</td>
      <td>9.1</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37033</th>
      <td>1986</td>
      <td>英国</td>
      <td>歌唱神探</td>
      <td>36</td>
      <td>剧情/悬疑/歌舞</td>
      <td>1986-11-16 00:00:00</td>
      <td>415</td>
      <td>9.1</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37557</th>
      <td>1975</td>
      <td>美国</td>
      <td>山那边</td>
      <td>70</td>
      <td>剧情</td>
      <td>1975-11-14 00:00:00</td>
      <td>103</td>
      <td>9.1</td>
      <td>美国</td>
      <td>A</td>
      <td>E</td>
    </tr>
    <tr>
      <th>37883</th>
      <td>2015</td>
      <td>美国</td>
      <td>奎</td>
      <td>62</td>
      <td>纪录片/短片</td>
      <td>2015-08-19 00:00:00</td>
      <td>9</td>
      <td>9.1</td>
      <td>纽约电影论坛</td>
      <td>A</td>
      <td>E</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 11 columns</p>


**将处理后的数据进行保存 **


```python
df.to_excel("movie_data3.xlsx")
```



## 3.12 合并数据集

### 3.12.1 append

先把数据集拆分为多个，再进行合并

```python
df_usa = df[df.产地 == "美国"]
df_china = df[df.产地 == "中国大陆"]
```


```python
df_china.append(df_usa) #直接追加到后面，最好是变量相同的
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.400000</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1961</td>
      <td>中国大陆</td>
      <td>大闹天宫</td>
      <td>74881</td>
      <td>动画/奇幻</td>
      <td>1905-05-14 00:00:00</td>
      <td>114</td>
      <td>9.200000</td>
      <td>上集</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2015</td>
      <td>中国大陆</td>
      <td>穹顶之下</td>
      <td>51113</td>
      <td>纪录片</td>
      <td>2015-02-28 00:00:00</td>
      <td>104</td>
      <td>9.200000</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1982</td>
      <td>中国大陆</td>
      <td>茶馆</td>
      <td>10678</td>
      <td>剧情/历史</td>
      <td>1905-06-04 00:00:00</td>
      <td>118</td>
      <td>9.200000</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1988</td>
      <td>中国大陆</td>
      <td>山水情</td>
      <td>10781</td>
      <td>动画/短片</td>
      <td>1905-06-10 00:00:00</td>
      <td>19</td>
      <td>9.200000</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38151</th>
      <td>1987</td>
      <td>美国</td>
      <td>零下的激情</td>
      <td>199</td>
      <td>剧情/爱情/犯罪</td>
      <td>1987-11-06 00:00:00</td>
      <td>98</td>
      <td>7.400000</td>
      <td>美国</td>
      <td>B</td>
      <td>D</td>
    </tr>
    <tr>
      <th>38153</th>
      <td>1986</td>
      <td>美国</td>
      <td>离别秋波</td>
      <td>240</td>
      <td>剧情/爱情/音乐</td>
      <td>1986-02-19 00:00:00</td>
      <td>90</td>
      <td>8.200000</td>
      <td>美国</td>
      <td>B</td>
      <td>C</td>
    </tr>
    <tr>
      <th>38156</th>
      <td>1986</td>
      <td>美国</td>
      <td>极乐森林</td>
      <td>45</td>
      <td>纪录片</td>
      <td>1986-09-14 00:00:00</td>
      <td>90</td>
      <td>8.100000</td>
      <td>美国</td>
      <td>B</td>
      <td>E</td>
    </tr>
    <tr>
      <th>38158</th>
      <td>1935</td>
      <td>美国</td>
      <td>1935年</td>
      <td>57</td>
      <td>喜剧/歌舞</td>
      <td>1935-03-15 00:00:00</td>
      <td>98</td>
      <td>7.600000</td>
      <td>美国</td>
      <td>B</td>
      <td>E</td>
    </tr>
    <tr>
      <th>38162</th>
      <td>2018</td>
      <td>美国</td>
      <td>复仇者联盟3</td>
      <td>123456</td>
      <td>剧情/科幻</td>
      <td>2018-05-04 00:00:00</td>
      <td>142</td>
      <td>6.935704</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
<p>15505 rows × 11 columns</p>
将这两个数据集进行合并



### 3.12.2 merge？

```python
pd.merge(left, right, how = 'inner', on = None, left_on = None, right_on = None,
    left_index = False, right_index = False, sort = True,
    suffixes = ('_x', '_y'), copy = True, indicator = False, validate=None) 
```

left : DataFrame

right : DataFrame or named Series
    Object to merge with.

how : {'left', 'right', 'outer', 'inner'}, default 'inner'
    Type of merge to be performed.



以六部热门电影为例：


```python
df1 = df.loc[:5]
df1
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.4</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
  </tbody>
</table>



```python
df2 = df.loc[:5][["名字","产地"]]
df2["票房"] = [123344,23454,55556,333,6666,444]
```


```python
df2
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>产地</th>
      <th>票房</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>肖申克的救赎</td>
      <td>美国</td>
      <td>123344</td>
    </tr>
    <tr>
      <th>1</th>
      <td>控方证人</td>
      <td>美国</td>
      <td>23454</td>
    </tr>
    <tr>
      <th>2</th>
      <td>美丽人生</td>
      <td>意大利</td>
      <td>55556</td>
    </tr>
    <tr>
      <th>3</th>
      <td>阿甘正传</td>
      <td>美国</td>
      <td>333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>中国大陆</td>
      <td>6666</td>
    </tr>
    <tr>
      <th>5</th>
      <td>泰坦尼克号</td>
      <td>美国</td>
      <td>444</td>
    </tr>
  </tbody>
</table>





```python
df2 = df2.sample(frac = 1) #打乱数据
```


```python
df2.index = range(len(df2))
df2
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>名字</th>
      <th>产地</th>
      <th>票房</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>泰坦尼克号</td>
      <td>美国</td>
      <td>444</td>
    </tr>
    <tr>
      <th>1</th>
      <td>阿甘正传</td>
      <td>美国</td>
      <td>333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>控方证人</td>
      <td>美国</td>
      <td>23454</td>
    </tr>
    <tr>
      <th>3</th>
      <td>美丽人生</td>
      <td>意大利</td>
      <td>55556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>霸王别姬</td>
      <td>中国大陆</td>
      <td>6666</td>
    </tr>
    <tr>
      <th>5</th>
      <td>肖申克的救赎</td>
      <td>美国</td>
      <td>123344</td>
    </tr>
  </tbody>
</table>




现在，我们需要把df1和df2合并

我们发现，df2有票房数据，df1有评分等其他信息  
由于样本的顺序不一致，因此不能直接采取直接复制的方法


```python
pd.merge(df1, df2, how = "inner", on = "名字")
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地_x</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
      <th>产地_y</th>
      <th>票房</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>A</td>
      <td>A</td>
      <td>美国</td>
      <td>123344</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
      <td>美国</td>
      <td>23454</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>A</td>
      <td>A</td>
      <td>意大利</td>
      <td>55556</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>A</td>
      <td>A</td>
      <td>美国</td>
      <td>333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
      <td>中国大陆</td>
      <td>6666</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.4</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
      <td>美国</td>
      <td>444</td>
    </tr>
  </tbody>
</table>




由于两个数据集都存在产地，因此合并后会有两个产地信息



### 3.11.3 concat

将多个数据集进行批量合并

```python
df1 = df[:10]
df2 = df[100:110]
df3 = df[200:210]
dff = pd.concat([df1,df2,df3],axis = 0) #默认axis = 0，列拼接需要修改为1
dff
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>157074</td>
      <td>剧情/爱情/灾难</td>
      <td>2012-04-10 00:00:00</td>
      <td>194</td>
      <td>9.4</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1993</td>
      <td>美国</td>
      <td>辛德勒的名单</td>
      <td>306904</td>
      <td>剧情/历史/战争</td>
      <td>1993-11-30 00:00:00</td>
      <td>195</td>
      <td>9.4</td>
      <td>华盛顿首映</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1997</td>
      <td>日本</td>
      <td>新世纪福音战士剧场版：Air/真心为你 新世紀エヴァンゲリオン劇場版 Ai</td>
      <td>24355</td>
      <td>剧情/动作/科幻/动画/奇幻</td>
      <td>1997-07-19 00:00:00</td>
      <td>87</td>
      <td>9.4</td>
      <td>日本</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2013</td>
      <td>日本</td>
      <td>银魂完结篇：直到永远的万事屋 劇場版 銀魂 完結篇 万事屋よ</td>
      <td>21513</td>
      <td>剧情/动画</td>
      <td>2013-07-06 00:00:00</td>
      <td>110</td>
      <td>9.4</td>
      <td>日本</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1994</td>
      <td>法国</td>
      <td>这个杀手不太冷</td>
      <td>662552</td>
      <td>剧情/动作/犯罪</td>
      <td>1994-09-14 00:00:00</td>
      <td>133</td>
      <td>9.4</td>
      <td>法国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1993</td>
      <td>韩国</td>
      <td>101</td>
      <td>146</td>
      <td>喜剧/爱情</td>
      <td>1993-06-19 00:00:00</td>
      <td>112</td>
      <td>7.4</td>
      <td>韩国</td>
      <td>B</td>
      <td>D</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1995</td>
      <td>英国</td>
      <td>10</td>
      <td>186</td>
      <td>喜剧</td>
      <td>1995-01-25 00:00:00</td>
      <td>101</td>
      <td>7.4</td>
      <td>美国</td>
      <td>B</td>
      <td>D</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2013</td>
      <td>韩国</td>
      <td>素媛</td>
      <td>114819</td>
      <td>剧情/家庭</td>
      <td>2013-10-02 00:00:00</td>
      <td>123</td>
      <td>9.1</td>
      <td>韩国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2003</td>
      <td>美国</td>
      <td>101忠狗续集：伦敦</td>
      <td>924</td>
      <td>喜剧/动画/家庭</td>
      <td>2003-01-21 00:00:00</td>
      <td>70</td>
      <td>7.5</td>
      <td>美国</td>
      <td>B</td>
      <td>B</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2000</td>
      <td>美国</td>
      <td>10</td>
      <td>9514</td>
      <td>喜剧/家庭</td>
      <td>2000-09-22 00:00:00</td>
      <td>100</td>
      <td>7.0</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2013</td>
      <td>韩国</td>
      <td>10</td>
      <td>601</td>
      <td>剧情</td>
      <td>2014-04-24 00:00:00</td>
      <td>93</td>
      <td>7.2</td>
      <td>美国</td>
      <td>B</td>
      <td>C</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2006</td>
      <td>美国</td>
      <td>10件或</td>
      <td>1770</td>
      <td>剧情/喜剧/爱情</td>
      <td>2006-12-01 00:00:00</td>
      <td>82</td>
      <td>7.7</td>
      <td>美国</td>
      <td>B</td>
      <td>B</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2014</td>
      <td>美国</td>
      <td>10年</td>
      <td>1531</td>
      <td>喜剧/同性</td>
      <td>2015-06-02 00:00:00</td>
      <td>90</td>
      <td>6.9</td>
      <td>美国</td>
      <td>C</td>
      <td>B</td>
    </tr>
    <tr>
      <th>108</th>
      <td>2012</td>
      <td>日本</td>
      <td>11·25自决之日 三岛由纪夫与年轻人们 11・25自決の</td>
      <td>149</td>
      <td>剧情</td>
      <td>2012-06-02 00:00:00</td>
      <td>119</td>
      <td>5.6</td>
      <td>日本</td>
      <td>C</td>
      <td>D</td>
    </tr>
    <tr>
      <th>109</th>
      <td>1997</td>
      <td>美国</td>
      <td>泰坦尼克号</td>
      <td>535491</td>
      <td>剧情/爱情/灾难</td>
      <td>1998-04-03 00:00:00</td>
      <td>194</td>
      <td>9.1</td>
      <td>中国大陆</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>200</th>
      <td>2014</td>
      <td>日本</td>
      <td>最完美的离婚 2014特别篇</td>
      <td>18478</td>
      <td>剧情/喜剧/爱情</td>
      <td>2014-02-08 00:00:00</td>
      <td>120</td>
      <td>9.1</td>
      <td>日本</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2009</td>
      <td>日本</td>
      <td>2001夜物</td>
      <td>84</td>
      <td>剧情/动画</td>
      <td>2009-10-02 00:00:00</td>
      <td>80</td>
      <td>6.6</td>
      <td>美国</td>
      <td>C</td>
      <td>D</td>
    </tr>
    <tr>
      <th>202</th>
      <td>2009</td>
      <td>中国香港</td>
      <td>头七 頭</td>
      <td>7039</td>
      <td>恐怖</td>
      <td>2009-05-21 00:00:00</td>
      <td>60</td>
      <td>6.2</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
    <tr>
      <th>203</th>
      <td>1896</td>
      <td>法国</td>
      <td>火车进站 L</td>
      <td>7001</td>
      <td>纪录片/短片</td>
      <td>1896-01-06</td>
      <td>60</td>
      <td>8.8</td>
      <td>法国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>204</th>
      <td>2009</td>
      <td>美国</td>
      <td>银行舞蹈</td>
      <td>6944</td>
      <td>短片</td>
      <td>1905-07-01 00:00:00</td>
      <td>60</td>
      <td>7.8</td>
      <td>美国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>205</th>
      <td>2003</td>
      <td>荷兰</td>
      <td>2003提雅</td>
      <td>48</td>
      <td>音乐</td>
      <td>2003-10-07 00:00:00</td>
      <td>200</td>
      <td>8.9</td>
      <td>美国</td>
      <td>B</td>
      <td>E</td>
    </tr>
    <tr>
      <th>206</th>
      <td>2012</td>
      <td>美国</td>
      <td>死亡飞车3：地狱烈</td>
      <td>6937</td>
      <td>动作</td>
      <td>2012-12-12 00:00:00</td>
      <td>60</td>
      <td>5.8</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
    <tr>
      <th>207</th>
      <td>2012</td>
      <td>日本</td>
      <td>时光钟摆 振り</td>
      <td>6876</td>
      <td>剧情/动画/短片</td>
      <td>2012-03-20 00:00:00</td>
      <td>60</td>
      <td>8.7</td>
      <td>美国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>208</th>
      <td>2011</td>
      <td>中国香港</td>
      <td>你还可爱么 你還可愛</td>
      <td>6805</td>
      <td>短片</td>
      <td>2011-04-22 00:00:00</td>
      <td>60</td>
      <td>8.3</td>
      <td>美国</td>
      <td>B</td>
      <td>A</td>
    </tr>
    <tr>
      <th>209</th>
      <td>2002</td>
      <td>中国香港</td>
      <td>一碌蔗</td>
      <td>6799</td>
      <td>剧情/喜剧/爱情</td>
      <td>2002-09-19 00:00:00</td>
      <td>60</td>
      <td>6.7</td>
      <td>美国</td>
      <td>C</td>
      <td>A</td>
    </tr>
  </tbody>
</table>


# 4 Python数据分析之Matplotlib

## 4.1 Matplotlib基础



**matplotlib**是一个**Python**的 2D 图形包。pyplot封装了很多画图的函数

导入相关的包：


```python
import matplotlib.pyplot as plt
import numpy as np
```

``matplotlib.pyplot``包含一系列类似**MATLAB**中绘图函数的相关函数。每个``matplotlib.pyplot``中的函数对当前的图像进行一些修改，例如：产生新的图像，在图像中产生新的绘图区域，在绘图区域中画线，给绘图加上标记，等等......``matplotlib.pyplot``会自动记住当前的图像和绘图区域，因此这些函数会直接作用在当前的图像上。

在实际的使用过程中，常常以``plt``作为``matplotlib.pyplot``的省略。

### 4.1.1 plt.show()函数 

默认情况下，``matplotlib.pyplot``不会直接显示图像，只有调用``plt.show()``函数时，图像才会显示出来。

**plt.show()``默认是在新窗口打开一幅图像，并且提供了对图像进行操作的按钮。**

不过在``ipython``命令中，我们可以将它插入``notebook``中，并且不需要调用``plt.show()``也可以显示：

* ``%matplotlib notebook``
* ``%matplotlib inline``

不过在实际写程序中，我们还是习惯调用``plt.show()``函数将图像显示出来。


```python
%matplotlib inline #魔术命令
```

### 4.1.2 plt.plot()函数 

#### 4.1.2.1 例子

``plt.plot()``函数可以用来绘线型图：


```python
plt.plot([1,2,3,4]) #默认以列表的索引作为x，输入的是y
plt.ylabel('y')
plt.xlabel("x轴") #设定标签，使用中文的话后面需要再设定
```


    Text(0.5, 0, 'x轴')


![image-20240123154546719](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123154546719.png)
    

#### 4.1.2.2 基本用法

``plot``函数基本的用法：

指定x和y

* ``plt.plot(x,y)``

默认参数，x为0~N-1

* ``plt.plot(y)``

因此，在上面的例子中，我们没有给定``x``的值，所以其默认值为``[0,1,2,3]``

传入``x``和``y``：


```python
plt.plot([1,2,3,4],[1,4,9,16])
plt.show() #相当于打印的功能，下面不会再出现内存地址
```

![image-20240123155235072](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155235072.png)
    




#### 4.1.2.3 字符参数

和**MATLAB**中类似，我们还可以用字符来指定绘图的格式：

表示颜色的字符参数有：

|字符|颜色|
|-:|-:|
|``'b'``|蓝色，blue|
|``'g'``|绿色，green|
|``'r'``|红色，red|
|``'c'``|青色，cyan|
|``'m'``|品红，magenta|
|``'y'``|黄色，yellow|
|``'k'``|黑色，black|
|``'w'``|白色，white|

表示类型的字符参数有：

|字符|类型|字符|类型|
|-:|-:|-:|-:|
|``'-'``|实线|``'--'``|虚线|
|``'-'.``|虚点线|``':'``|点线|
|``'.'``|点|``','``|像素点|
|``'o'``|圆点|``'v'``|下三角点|
|``'^'``|上三角点|``'<'``|左三角点|
|``'>'``|右三角点|``'1'``|下三叉点|
|``'2'``|上三叉点|``'3'``|左三叉点|
|``'4'``|右三叉点|``'s'``|正方点|
|``'p'``|五角点|``'*'``|星形点|
|``'h'``|六边形点1|``'H'``|六边形点2|
|``'+'``|加号点|``'x'``|乘号点|
|``'D'``|实心菱形点|``'d'``|瘦菱形点|
|``'_'``|横线点|||

例如我们要画出红色圆点：


```python
plt.plot([1,2,3,4],[1,4,9,16],"ro") #也可以是or，没顺序要求
plt.show()
```


![image-20240123155542708](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155542708.png)
    

可以看出，有两个点在图像的边缘，因此，我们需要改变轴的显示范围。



#### 4.1.2.4 显示范围

与**MATLAB**类似，这里可以使用``axis``函数指定坐标轴显示的范围：
```python
plt.axis([xmin, xmax, ymin, ymax])
```


```python
plt.plot([1,2,3,4],[1,4,9,16],"g*")
plt.axis([0,6,0,20])
plt.show()
```

![image-20240123155631341](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155631341.png)
    




#### 4.1.2.5 传入``Numpy``数组 

之前我们传给``plot``的参数都是列表，事实上，向``plot``中传入``numpy``数组是更常用的做法。事实上，如果传入的是列表，``matplotlib``会在内部将它转化成数组再进行处理：

在一个图里面画多条线：


```python
t = np.arange(0.,5.,0.2) #左闭右开从0到5间隔0.2
plt.plot(t,t,"r--",
        t,t**2,"bs",
        t,t**3,"g^")
plt.show()
```

![image-20240123155802223](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123155802223.png)
    




#### 4.1.2.6 传入多组数据 

事实上，在上面的例子中，我们不仅仅向``plot``函数传入了数组，还传入了多组``(x,y,format_str)``参数，它们在同一张图上显示。

这意味着我们不需要使用多个``plot``函数来画多组数组，只需要可以将这些组合放到一个``plot``函数中去即可。



#### 4.1.2.7 线条属性 

之前提到，我们可以用**字符串**来控制线条的属性，事实上还可以用**关键词**来改变线条的性质，例如``linewidth``可以改变线条的宽度，``color``可以改变线条的颜色：


```python
x = np.linspace(-np.pi,np.pi)
y = np.sin(x)
plt.plot(x,y,linewidth = 4.0,color = 'r') #细节调整的两个方式
plt.show()
```

![image-20240123160023088](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123160023088.png)
    




#### 4.1.2.8 使用plt.plot()的返回值来设置线条属性

``plot``函数返回一个``Line2D``对象组成的列表，每个对象代表输入的一对组合，例如：

* line1,line2 为两个 Line2D 对象
```python
line1, line2 = plt.plot(x1, y1, x2, y2)
```
* 返回3个 Line2D 对象组成的列表
```python
lines = plt.plot(x1, y1, x2, y2, x3, y3)
```

我们可以使用这个返回值来对线条属性进行设置：


```python
line1,line2 = plt.plot(x,y,"r-",x,y+1,"g-")
line1.set_antialiased(False)  #抗锯齿
plt.show()
```


![image-20240123161045947](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161045947.png)
    



```python
line = plt.plot(x,y,"r-",x,y+1,"g-")
line[1].set_antialiased(False) #列表
plt.show()
```

![image-20240123161150448](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161150448.png)
    




#### 4.1.2.9 plt.setp() 修改线条性质

更方便的做法是使用``plt``的``setp``函数：

```python
line = plt.plot(x,y)
#plt.setp(line, color = 'g',linewidth = 4)
plt.setp(line,"color",'r',"linewidth",4) #matlab风格
```

![image-20240123161521568](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161521568.png)



#### 4.1.2.10 子图 

``figure()``函数会产生一个指定编号为``num``的图：

```python
plt.figure(num)
```
这里，``figure(1)``其实是可以省略的，因为默认情况下``plt``会自动产生一幅图像。

使用``subplot``可以在一幅图中生成多个子图，其参数为：
```python
plt.subplot(numrows, numcols, fignum)
```
当``numrows * numncols < 10``时，中间的逗号可以省略，因此``plt.subplot(211)``就相当于``plt.subplot(2,1,1)``。


```python
def f(t):
    return np.exp(-t)*np.cos(2*np.pi*t)

t1 = np.arange(0.0,5.0,0.1)
t2 = np.arange(0.0,4.0,0.02)

plt.figure(figsize = (10,6))
plt.subplot(211)
plt.plot(t1,f(t1),"bo",t2,f(t2),'k') #子图1上有两条线

plt.subplot(212)
plt.plot(t2,np.cos(2*np.pi*t2),"r--")
plt.show()
```

![image-20240123161658051](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123161658051.png)
    

## 4.2 电影数据绘图

**在了解绘图的基础知识之后，我们可以对电影数据进行可视化分析。**


```python
import warnings
warnings.filterwarnings("ignore") #关闭一些可能出现但对数据分析并无影响的警告
```


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
plt.rcParams["font.sans-serif"] = ["SimHei"] #解决中文字符乱码的问题
plt.rcParams["axes.unicode_minus"] = False #正常显示负号
```


```python
df = pd.read_excel(r"C:\Users\Lovetianyi\Desktop\python\作业5\movie_data3.xlsx", index_col = 0)
```


```python
df[:5]
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>年代</th>
      <th>产地</th>
      <th>名字</th>
      <th>投票人数</th>
      <th>类型</th>
      <th>上映时间</th>
      <th>时长</th>
      <th>评分</th>
      <th>首映地点</th>
      <th>评分等级</th>
      <th>热门程度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1994</td>
      <td>美国</td>
      <td>肖申克的救赎</td>
      <td>692795</td>
      <td>剧情/犯罪</td>
      <td>1994-09-10 00:00:00</td>
      <td>142</td>
      <td>9.6</td>
      <td>多伦多电影节</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>美国</td>
      <td>控方证人</td>
      <td>42995</td>
      <td>剧情/悬疑/犯罪</td>
      <td>1957-12-17 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>美国</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1997</td>
      <td>意大利</td>
      <td>美丽人生</td>
      <td>327855</td>
      <td>剧情/喜剧/爱情</td>
      <td>1997-12-20 00:00:00</td>
      <td>116</td>
      <td>9.5</td>
      <td>意大利</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1994</td>
      <td>美国</td>
      <td>阿甘正传</td>
      <td>580897</td>
      <td>剧情/爱情</td>
      <td>1994-06-23 00:00:00</td>
      <td>142</td>
      <td>9.4</td>
      <td>洛杉矶首映</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1993</td>
      <td>中国大陆</td>
      <td>霸王别姬</td>
      <td>478523</td>
      <td>剧情/爱情/同性</td>
      <td>1993-01-01 00:00:00</td>
      <td>171</td>
      <td>9.4</td>
      <td>香港</td>
      <td>A</td>
      <td>A</td>
    </tr>
  </tbody>
</table>


## 4.2.1 柱状图

绘制每个国家或地区的电影数量的柱状图：

**柱状图**(bar chart)，是一种以长方形的长度为变量的表达图形的统计报告图，由一系列高度不等的纵向条纹表示数据分布的情况，用来比较两个或以上的价值（不同时间或者不同条件），只有一个变量，通常利用较小的数据集分析。柱状图亦可横向排列，或用多维方式表达。


```python
data = df["产地"].value_counts()
data
```


    美国      11714
    日本       5006
    中国大陆     3791
    中国香港     2847
    法国       2787
    英国       2658
    其他       1883
    韩国       1342
    德国       1021
    意大利       741
    加拿大       709
    中国台湾      618
    俄罗斯       476
    西班牙       443
    印度        356
    澳大利亚      295
    泰国        294
    丹麦        197
    瑞典        187
    波兰        181
    荷兰        151
    比利时       137
    墨西哥       117
    阿根廷       113
    巴西         99
    Name: 产地, dtype: int64




```python
x = data.index
y = data.values

plt.figure(figsize = (10,6)) #设置图片大小
plt.bar(x,y,color = "g") #绘制柱状图，表格给的数据是怎样就怎样，不会自动排序

plt.title("各国家或地区电影数量", fontsize = 20) #设置标题
plt.xlabel("国家或地区",fontsize = 18) 
plt.ylabel("电影数量") #对横纵轴进行说明
plt.tick_params(labelsize = 14) #设置标签字体大小
plt.xticks(rotation = 90) #标签转90度

for a,b in zip(x,y): #数字直接显示在柱子上（添加文本）
    #a:x的位置，b:y的位置，加上10是为了展示位置高一点点不重合，
    #第二个b:显示的文本的内容,ha,va:格式设定,center居中,top&bottom在上或者在下,fontsize:字体指定
    plt.text(a,b+10,b,ha = "center",va = "bottom",fontsize = 10) 

#plt.grid() #画网格线，有失美观因而注释点

plt.show()
```

![image-20240123162855896](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123162855896.png)



### 4.2.2 曲线图

绘制每年上映的电影数量的曲线图：

**曲线图**又称折线图，是利用曲线的升，降变化来表示被研究现象发展趋势的一种图形。它在分析研究社会经济现象的发展变化、依存关系等方面具有重要作用。

绘制曲线图时，如果是某一现象的时间指标，应将时间绘在坐标的横轴上，指标绘在坐标的纵轴上。如果是两个现象依存关系的显示，可以将表示原因的指标绘在横轴上，表示结果的指标绘在纵轴上，同时还应注意整个图形的长宽比例。

### 1888-2015年 


```python
data = df["年代"].value_counts()
data = data.sort_index()[:-2] #排除掉2016年以后的数据，共两条
data
```


    1888       2
    1890       1
    1892       1
    1894       3
    1895       8
            ... 
    2011    1845
    2012    2018
    2013    1977
    2014    1867
    2015    1569
    Name: 年代, Length: 125, dtype: int64




```python
x = data.index
y = data.values

plt.plot(x,y,color = 'b')
plt.title("每年电影数量",fontsize = 20)
plt.ylabel("电影数量",fontsize = 18)
plt.xlabel("年份",fontsize = 18)

for (a,b) in zip(x[::10],y[::10]): #每隔10年进行数量标记，防止过于密集
    plt.text(a,b+10,b,ha = "center", va = "bottom", fontsize = 10)
    
#标记特殊点如极值点，xy设置箭头尖的坐标，xytext注释内容起始位置，arrowprops对箭头设置，传字典，facecolor填充颜色，edgecolor边框颜色
plt.annotate("2012年达到最大值", xy = (2012,data[2012]), xytext = (2025,2100), arrowprops = dict(facecolor = "black",edgecolor = "red"))

#纯文本注释内容，例如注释增长最快的地方
plt.text(1980,1000,"电影数量开始快速增长")
plt.show()
```

![image-20240123163449116](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123163449116.png)
    利用matplotlib绘图已经较为成熟了


对于这幅图形，我们使用``xlabel, ylabel, title, text``方法设置了文字，其中：

* ``xlabel``: x轴标注
* ``ylabel``: y轴标注
* ``title``: 图形标题
* ``text``: 在指定位置(坐标)放入文字

输入特殊符号支持使用``Tex``语法，用``$<some Text code>$``隔开。

除了使用``text``在指定位置标上文字之外，还可以使用``annotate``进行注释，``annotate``主要有两个参数：

* ``xy``: 注释位置
* ``xytext``: 注释文字位置

从上图可以看出，电影数量是逐年增加的，增长的趋势在2000年都变得飞快 



### 4.2.3 饼图 

根据电影的长度绘制饼图：

**饼图**英文学名为Sector Graph，又名Pie Graph。常用于统计学模块。2D饼图为圆形，手画时，常用圆规作图。

仅排列在工作表的一列或一行中的数据可以绘制到饼图中。饼图显示一个数据系列（数据系列：在图表中绘制的相关数据点，这些数据源自数据表的行或列。图表中的每个数据系列具有唯一的颜色或团并且在图表中的图例中表示。可以在图表中绘制一个或多个数据系列。饼图只有一个数据系列。）中各项的大小与各项总和的比例。饼图中的数据点（数据点：在图表中绘制的单个值，这些值由条形，柱形，折线，饼图或圆环图的扇面、圆点和其他被称为数据标记的图形表示。相同颜色的数据标记组成一个数据系列。）显示为整个饼图的百分比。

**函数原型：**

```python
pie(x, explode = None, labels = None, colors = None, autopct = None, pctdistance = 0.6,
    shadow = False, labeldistance = 1.1, startangle = None, radius = None)
```

**参数：**  
**x：** (每一块）的比例，如果sum(x)>1会使用sum(x)归一化  
**labels：** （每一块）饼图外侧显示的说明文字  
**explode：** （每一块）离开中心距离  
**startangle：** 起始绘制角度，默认图是从x轴正方向逆时针画起，如设定=90则从y轴正方向画起   
**shadow：** 是否阴影  
**labeldistance：** label绘制位置，相对于半径的比例，如<1则绘制在饼图内侧  
**autopct：** 控制饼图内百分比设置，可以使用format字符串或者format function  
**'%1.1f'：** 指小数点前后位数（没有用空格补齐）  
**pctdistance：** 类似于labeldistance，指定autopct的位置刻度  
**radius：** 控制饼图半径

**返回值：**  
如果没有设置autopct，返回(patches,texts)  
如果设置autopct，返回(patches,texts,autotexts)


```python
data = pd.cut(df["时长"], [0,60,90,110,1000]).value_counts() #数据离散化
data
```


    (90, 110]      13201
    (0, 60]         9884
    (60, 90]        7661
    (110, 1000]     7417
    Name: 时长, dtype: int64




```python
y = data.values
y = y/sum(y) #归一化，不进行的话系统会自动进行

plt.figure(figsize = (7,7))
plt.title("电影时长占比",fontsize = 15)
patches,l_text,p_text = plt.pie(y, labels = data.index, autopct = "%.1f %%", colors = "bygr", startangle = 90)

for i in p_text: #通过返回值设置饼图内部字体
    i.set_size(15)
    i.set_color('w')

for i in l_text: #通过返回值设置饼图外部字体
    i.set_size(15)
    i.set_color('r')
    
plt.legend() #图例
plt.show()
```

![image-20240123163810642](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123163810642.png)
    

### 4.2.4 频率直方图

根据电影的评分绘制频率直方图：

**直方图**(Histogram)又称质量分布图。是一种统计报告图。由一系列高度不等的纵向条纹或线段表示数据分布的情况。一般用横轴表示数据类型，纵轴表示分布情况。

直方图是数值数据分布的精确图形表示。这是一个连续变量（定量变量）的概率分布的估计，并且被卡尔·皮尔逊(Karl Pearson)首先引入。它是一种条形图。为了构建直方图，第一步是将值的范围分段，即将整个值的范围分成一系列间隔，然后计算每个间隔中有多少值。这些值通常被指定为连续的，不重叠的变量间隔。间隔必须相邻，并且通常是（但不是必须的）相等的大小。

直方图也可以被归一化以显示“相对频率”。然后，它显示了属于几个类别中每个案例的比例，其高度等于1。


```python
plt.figure(figsize = (10,6))
plt.hist(df["评分"], bins = 20, edgecolor = 'k',alpha = 0.5)
plt.show()
```


![image-20240123164305903](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123164305903.png)
    


hist的参数非常多，但常用的就这六个，只有第一个是必须的，后面可选

**arr**: 需要计算直方图的一维数组

**bins**: 直方图的柱数，可选项，默认为10

**normed**: 是否将得到的直方图向量归一化。默认为0

**facecolor**: 直方图颜色

**edgecolor**: 直方图边框颜色

**alpha**: 透明度

**histtype**: 直方图类型，"bar", "barstacked", "step", "stepfilled"

返回值：

**n**: 直方图向量，是否归一化由参数normed设定

**bins**: 返回各个bin的区间范围

**patches**: 返回每一个bin里面包含的数据，是一个list

从上图我们可以发现，电影的评分是服从一个右偏的正态分布的。 





### 4.2.5 双轴图

```python
from scipy.stats import norm #获取正态分布密度函数
```


```python
fig = plt.figure(figsize = (10,8))
ax1 = fig.add_subplot(111) #确认子图
n,bins,patches = ax1.hist(df["评分"],bins = 100, color = 'm') #bins默认是10

ax1.set_ylabel("电影数量",fontsize = 15)
ax1.set_xlabel("评分",fontsize = 15)
ax1.set_title("频率分布图",fontsize = 20)

#准备拟合
y = norm.pdf(bins,df["评分"].mean(),df["评分"].std()) #bins,mu,sigma
ax2 = ax1.twinx() #双轴
ax2.plot(bins,y,"b--")
ax2.set_ylabel("概率分布",fontsize = 15)
plt.show()
```

![image-20240123165422619](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123165422619.png)



### 4.2.6 散点图

根据电影时长和电影评分绘制散点图：

用两组数据构成多个坐标点，考察坐标点的分布，判断两变量之间是否存在某种关联或总结坐标点的分布模式。散点图将序列显示为一组点。值由点在图表中的位置表示。类别由图表中的不同标记表示。散点图通常用于比较跨类别的聚合数据。


```python
x = df["时长"][::100]
y = df["评分"][::100] #解决数据冗杂的问题

plt.figure(figsize = (10,6))
plt.scatter(x,y,color = 'c',marker = 'p',label = "评分")
plt.legend() #图例
plt.title("电影时长与评分散点图",fontsize = 20)
plt.xlabel("时长",fontsize = 18)
plt.ylabel("评分",fontsize = 18)
plt.show()
```

![image-20240123165612585](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123165612585.png)
    




由于我们的数据量过大，所以画出来的图非常冗杂

可以发现，大部分的电影时长还是集中在100附近，评分大多在7分左右



#### 4.2.6.1 marker属性

设置散点的形状

| **marker** | **description** | **描述**  |
| ---------- | --------------- | --------- |
| "."        | point           | 点        |
| ","        | pixel           | 像素      |
| "o"        | circle          | 圈        |
| "v"        | triangle_down   | 倒三角形  |
| "^"        | triangle_up     | 正三角形  |
| "<"        | triangle_left   | 左三角形  |
| ">"        | triangle_right  | 右三角形  |
| "1"        | tri_down        | tri_down  |
| "2"        | tri_up          | tri_up    |
| "3"        | tri_left        | tri_left  |
| "4"        | tri_right       | tri_right |
| "8"        | octagon         | 八角形    |
| "s"        | square          | 正方形    |
| "p"        | pentagon        | 五角      |
| "\*"       | star            | 星星      |
| "h"        | hexagon1        | 六角1     |
| "H"        | hexagon2        | 六角2     |
| "+"        | plus            | 加号      |
| "x"        | x               | x号       |
| "D"        | diamond         | 钻石      |
| "d"        | thin_diamon     | 细钻      |
| "\|"       | vline           | v线       |
| "\_"       | hline           | H线       |



### 4.2.7 箱型图

绘制各个地区的评分箱型图

**箱型图**（Box-plot）又称为盒须图，盒式图或箱型图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。在各种领域中也经常被使用，常见于品质管理。它主要用于反映原始数据分布的特征，还可以进行多组数据分布特征的比较。箱线图的绘制方法是：先找出一组数据的中位数，两个四分位数，上下边缘线；然后，连接两个四分位数画出箱子；再将上下边缘线与箱子相连接，中位数在箱子中间。

箱型图（Box-plot）又称为盒须图，盒式图或箱型图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。在各种领域中也经常被使用，常见于品质管理。它主要用于反映原始数据分布的特征，还可以进行多组数据分布特征的比较。箱线图的绘制方法是：先找出一组数据的中位数，两个四分位数，上下边缘线；然后，连接两个四分位数画出箱子；再将上下边缘线与箱子相连接，中位数在箱子中间。

箱型图（Box-plot）又称为盒须图，盒式图或箱型图，是一种用作显示一组数据分散情况资料的统计图。因形状如箱子而得名。在各种领域中也经常被使用，常见于品质管理。它主要用于反映原始数据分布的特征，还可以进行多组数据分布特征的比较。箱线图的绘制方法是：先找出一组数据的中位数，两个四分位数，上下边缘线；然后，连接两个四分位数画出箱子；再将上下边缘线与箱子相连接，中位数在箱子中间。

![375d3e018a4a35cde9d3cb178ce43ce](https://aquazone.oss-cn-guangzhou.aliyuncs.com/375d3e018a4a35cde9d3cb178ce43ce.jpg)



**一般计算过程**

（ 1 ）计算上四分位数（ Q3 ），中位数，下四分位数（ Q1 ）

（ 2 ）计算上四分位数和下四分位数之间的差值，即四分位数差（IQR, interquartile range）Q3-Q1

（ 3 ）绘制箱线图的上下范围，上限为上四分位数，下限为下四分位数。在箱子内部中位数的位置绘制横线

（ 4 ）大于上四分位数1.5倍四分位数差的值，或者小于下四分位数1.5倍四分位数差的值，划为异常值（outliers）

（ 5 ）异常值之外，最靠近上边缘和下边缘的两个值处，画横线，作为箱线图的触须

（ 6 ）极端异常值，即超出四分位数差3倍距离的异常值，用实心点表示；较为温和的异常值，即处于1.5倍-3倍四分位数差之间的异常值，用空心点表示

（ 7 ）为箱线图添加名称，数轴等

**参数详解**

```python
plt.boxplot(x,notch=None,sym=None,vert=None,
    whis=None,positions=None,widths=None,
    patch_artist=None,meanline=None,showmeans=None,
    showcaps=None,showbox=None,showfliers=None,
    boxprops=None,labels=None,flierprops=None,
    medianprops=None,meanprops=None,
    capprops=None,whiskerprops=None,)
```
**x: 指定要绘制箱线图的数据；**

**notch: 是否是凹口的形式展现箱线图，默认非凹口；**

**sym: 指定异常点的形状，默认为+号显示；**

**vert: 是否需要将箱线图垂直摆放，默认垂直摆放；**

**whis: 指定上下须与上下四分位的距离，默认为为1.5倍的四分位差；**

**positions: 指定箱线图的位置，默认为[0,1,2...]；**

**widths: 指定箱线图的宽度，默认为0.5；**

**patch_artist: 是否填充箱体的颜色；**

**meanline:是否用线的形式表示均值，默认用点来表示；**

**showmeans: 是否显示均值，默认不显示；**

**showcaps: 是否显示箱线图顶端和末端的两条线，默认显示；**

**showbox: 是否显示箱线图的箱体，默认显示；**

**showfliers: 是否显示异常值，默认显示；**

**boxprops: 设置箱体的属性，如边框色，填充色等；**

**labels: 为箱线图添加标签，类似于图例的作用；**

**filerprops: 设置异常值的属性，如异常点的形状、大小、填充色等；**

**medainprops: 设置中位数的属性，如线的类型、粗细等**

**meanprops: 设置均值的属性，如点的大小，颜色等；**

**capprops: 设置箱线图顶端和末端线条的属性，如颜色、粗细等；**

**whiskerprops: 设置须的属性，如颜色、粗细、线的类型等**



美国电影评分的箱线图:


```python
data = df[df.产地 == "美国"]["评分"]

plt.figure(figsize = (10,6))
plt.boxplot(data,whis = 2,flierprops = {"marker":'o',"markerfacecolor":"r","color":'k'}
           ,patch_artist = True, boxprops = {"color":'k',"facecolor":"#66ccff"})
plt.title("美国电影评分",fontsize = 20)
plt.show()
```

![image-20240123165958467](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123165958467.png)
    



多组数据箱线图 


```python
data1 = df[df.产地 == "中国大陆"]["评分"]
data2 = df[df.产地 == "日本"]["评分"]
data3 = df[df.产地 == "中国香港"]["评分"]
data4 = df[df.产地 == "英国"]["评分"]
data5 = df[df.产地 == "法国"]["评分"]

plt.figure(figsize = (12,8))
plt.boxplot([data1,data2,data3,data4,data5],labels = ["中国大陆","日本","中国香港","英国","法国"],
           whis = 2,flierprops = {"marker":'o',"markerfacecolor":"r","color":'k'}
           ,patch_artist = True, boxprops = {"color":'k',"facecolor":"#66ccff"},
           vert = False)

ax = plt.gca() #获取当时的坐标系
ax.patch.set_facecolor("gray") #设置坐标系背景颜色
ax.patch.set_alpha(0.3) #设置背景透明度

plt.title("电影评分箱线图",fontsize = 20)
plt.show()
```

![image-20240123170040437](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123170040437.png)

​    

**通过vert属性可以把图旋转过来**



### 4.2.9 热力图


```python
data = df[["投票人数","评分","时长"]]
data[:5]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>投票人数</th>
      <th>评分</th>
      <th>时长</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>692795</td>
      <td>9.6</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42995</td>
      <td>9.5</td>
      <td>116</td>
    </tr>
    <tr>
      <th>2</th>
      <td>327855</td>
      <td>9.5</td>
      <td>116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>580897</td>
      <td>9.4</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>478523</td>
      <td>9.4</td>
      <td>171</td>
    </tr>
  </tbody>
</table>


pandas本身也封装了画图函数 

我们可以画出各个属性之间的散点图，对角线是分布图 


```python
%pylab inline 
#魔术命令，让图像直接展示在notebook里面
result = pd.plotting.scatter_matrix(data[::100],diagonal = "kde",color = 'k',alpha = 0.3,figsize = (10,10)) 
#diagonal = hist:对角线上显示的是数据集各个特征的直方图/kde:数据集各个特征的核密度估计
```

![image-20240123170255598](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123170255598.png)
    



### 4.2.10 相关系数矩阵图

现在我们来画电影时长，投票人数，评分的一个相关系数矩阵图:

**seaborn**是一个精简的python库，可以创建具有统计意义的图表，能理解pandas的DataFrame类型


```python
import seaborn as sns

corr = data.corr() #获取相关系数
corr = abs(corr) #取绝对值

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111)

ax = sns.heatmap(corr,vmax = 1,vmin = 0,annot = True,annot_kws = {"size":13,"weight":"bold"},linewidths = 0.05)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.show()
```


![image-20240123170413299](https://aquazone.oss-cn-guangzhou.aliyuncs.com/image-20240123170413299.png)
   


#### 4.2.10.1 参数详解 

```python
sns.heatmap(data,vmin=None,vmax=None,cmap=None,center=None,robust=False,annot=None,fmt='.2g',annot_kws=None,linewidths=0,linecolor='white',cbar=True,cbar_kws=None,cbar_ax=None,square=False,xticklabels='auto',yticklabels='auto',mask=None,ax=None,**kwargs,)
```

（ 1 ）热力图输入数据参数：

data:矩阵数据集，可以是numpy的数组（array），也可以是pandas的DataFrame。如果是DataFrame，则df的index/column信息会分别对应到heatmap的columns和rows，即pt.index是热力图的行标，pt.columns是热力图的列标。

（ 2 ）热力图矩阵块颜色参数：

vmax,vmin:分别是热力图的颜色取值最大和最小范围，默认是根据data数据表里的取值确定。cmap:从数字到色彩空间的映射，取值是matplotlib包里的colormap名称或颜色对象，或者表示颜色的列表；改参数默认值：根据center参数设定。center:数据表取值有差异时，设置热力图的色彩中心对齐值；通过设置center值，可以调整生成的图像颜色的整体深浅；设置center数据时，如果有数据溢出，则手动设置的vmax、vmin会自动改变。robust:默认取值False，如果是False，且没设定vmin和vmax的值。

（ 3 ）热力图矩阵块注释参数：

annot(annotate的缩写):默认取值False；如果是True，在热力图每个方格写入数据；如果是矩阵，在热力图每个方格写入该矩阵对应位置数据。fmt:字符串格式代码，矩阵上标识数字的数据格式，比如保留小数点后几位数字。annot_kws:默认取值False；如果是True，设置热力图矩阵上数字的大小颜色字体，matplotlib包text类下的字体设置；

（ 4 ）热力图矩阵块之间间隔及间隔线参数：

linewidth:定义热力图里“表示两两特征关系的矩阵小块”之间的间隔大小。linecolor:切分热力图上每个矩阵小块的线的颜色，默认值是"white"。

（ 5 ）热力图颜色刻度条参数：

cbar:是否在热力图侧边绘制颜色进度条，默认值是True。cbar_kws:热力图侧边绘制颜色刻度条时，相关字体设置，默认值是None。cbar_ax：热力图侧边绘制颜色刻度条时，刻度条位置设置，默认值是None

（ 6 ）

square:设置热力图矩阵小块形状，默认值是False。xticklabels,yticklabels:xticklabels控制每列标签名的输出；yticklabels控制每行标签名的输出。默认值是auto。如果是True，则以DataFrame的列名作为标签名。如果是False，则不添加行标签名。如果是列表，则标签名改为列表中给的内容。如果是整数K，则在图上每隔K个标签进行一次标注。
