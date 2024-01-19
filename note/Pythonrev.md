# PYTHON基础语法快速复习

## 目录

[TOC]



# 1 起步

## 1.3.0 Helloworld
```python
print("Helloworld")
```

# 2 变量和简单数据类型
## 2.2 变量

```python
word = "hi"
print(word)
```

每个变量都指向一个值————与该变量相关联的信息
if修改变量的值 则始终记录最新值
```python
word = "hi"
print(word)
word = "helloworld"
print(word)
```

变量命名：只能包含*字母* *数字* *下划线*，且不可以数字打头，不可包含空格，慎用l O
避免犯低级命名错误 尽管有时难以避免
变量定义：变量是可以赋给值的标签，也可以说变量指向特定的值

## 2.3 字符串String
### 2.3.1 def 就是一系列字符 py中 用引号(单双等价)括起的都是字符串

字符串大小变换法则：在print（）中加入下列方法（可对数据执行的操作）：
title（）首字母大写
upper（）全部大写
lower（）全部小写

另外的，casefold（）效果也一样，区别就是lower（）只对A~Z有效
```python
name="chicken"
print(name.title())
```

### 2.3.2 在字符串中使用变量（新增f字符串）
f即format，py通过把花括号内的变量替换为它的值来设置字符串格式
```python
x='mth'
y='fker'
z=f'{x}{y}'
print(z)
```

```python
x='mth'
y='fker'
z=f'hello，{x.title()}{y}!'
print(z)
```

变量中引用变量需要{}
函数中调用变量（）

### 2.3.3 制表符和换行符添加空白
制表符：\t tab后移，不一定四位 
换行符：\n
叠加：\n\t
注意：\t\n 则无tab

```python
print("LTC:\n\tDZ\n\tMF")
print("LTC:\t\nDZ\n\tMF")
```

### 2.3.4 删除多余空白
lstrip() left去除开头空白
rstrip() right去除末尾空白
strip() 去除两边空白

```python
x="lsgo "
print(x.rstrip())
```

注意，这里仅仅只是删除了输出的空白，想要彻底永久删除，必须关联到变量

```python
x="lsgo "
x=x.rstrip()
print(x)
```

### 2.3.5 一些特性补充
#### 字符串加法
```python
s="Hello"+"World"
s
```
'hello worrld'

#### 三引号
三引号用来输入包含多行文字的字符串（可以是单引号也可以是双引号）
```python
s='''hello
my
world'''
print(s)
```
hello
my
world

#### 字符串索引(索引号从0开始)
```python
s='hello world'
s[1]
```
'e'

#### 字符串分割（spilt方法，也可按自己指定的方法分割，默认空格）
```python
s="hello world"
s.split()
```
['hello', 'world']

```python
s="hello,world"
s.split(",")
```
['hello', 'world']

#### 查看字符串长度
```python
len(s)
```
11

## 2.4 数

赋值 =
加 +
减 -
乘 *
除 /
乘方 **
不等于 ！=
等于 ==
逻辑运算：&与  |或 not取反

同时，还支持多次运算

### 2.4.2/3浮点数
一切带小数点的数都叫浮点数：小数点可以出现在数的任何位置
但是计算机的浮点运算是近似的，不要轻易写两个浮点数相等的判断语句，但是可以判断二者之差是否小于极小值
一切语言都有这个问题，本质上是因为有些十进制小数无法以二进制小数表示（整数可以）（ieee754目前主流语言浮点数规范所导致的）（计组见）
任意两个数相除，结果总是浮点数，即便这两个数都是整数且能整除
只要有操作数是浮点数，结果总是浮点数

### 2.4.4/5 数中的下划线以及同时多个变量赋值
书写很大的数时，可以用_划分，使更清晰易读，打印的时候，py不会管（3.6以上版本）

```python
x,y,z=1,2,3
print(x,y,z)
```

### 2.4.6 常量
一般地，用全大写来指出应当将某个变量视为常量

### 2.5 布尔类型
True or False

### 2.6 空值
```python
a=''
len(a)
```
0

## 2.7 注释
用井号表示

## 2.8 ZEN
> The Zen of Python, by Tim Peters
> Beautiful is better than ugly.
> Explicit is better than implicit.
> Simple is better than complex.
> Complex is better than complicated.
> Flat is better than nested.
> Sparse is better than dense.
> Readability counts.
> Special cases aren't special enough to break the rules.
> Although practicality beats purity.
> Errors should never pass silently.
> Unless explicitly silenced.
> In the face of ambiguity, refuse the temptation to guess.
> There should be one-- and preferably only one --obvious way to do it.
> Although that way may not be obvious at first unless you're Dutch.
> Now is better than never.
> Although never is often better than *right* now.
> If the implementation is hard to explain, it's a bad idea.
> If the implementation is easy to explain, it may be a good idea.
> Namespaces are one honking great idea -- let's do more of those!
>
> 不要企图编写完美无缺的代码，而是先编写有效的代码，再决定是做进一步改进，还是转而去编写新代码。


```python
import this
```

# 3 列表list
由一系列按特定顺序排列的元素组成，让你在一个地方存储成组的信息，有序的集合
命名多指定一个表示复数的名称

```python
X_JAPAN=['X','J','A','P','A','N']
print(X_JAPAN)
```

['X','J','A','P','A','N']

### 3.1.1/2/3 访问及使用列表元素

索引：表示元素的位置
索引从0开始
如果想要返回最后一个元素，可以将索引指定为-1（未知列表长度下非常有用）


```python
X_JAPAN=['X','J','A','P','A','N']
print(X_JAPAN[0].title())
```

X_JAPAN=['X','J','A','P','A','N']
word=f'My favourite band is {X_JAPAN[0]}.'
print(word)

## 修改添加和删除元素


```python
#修改
X_JAPAN=['X','J','A','P','A','N']
X_JAPAN[0]='xxx'
print(X_JAPAN)
```


```python
#添加
#附加到末尾(添加新的对象) append()
X_JAPAN=['X','J','A','P','A','N']
X_JAPAN.append('wow')
print(X_JAPAN)
```


```python
#在末尾追加(多个)值 extend()
X_JAPAN=['X','J','A','P','A','N']
X_JAPAN.extend('wow')
print(X_JAPAN)
```


```python
#插入元素 insert()
X_JAPAN=['X','J','A','P','A','N']
X_JAPAN.insert(0,'wow')
print(X_JAPAN)
```


```python
#删除元素
#del语句
X_JAPAN=['X','J','A','P','A','N']
del X_JAPAN[0]
print(X_JAPAN)
```

删除的方法（不是语句）：

+ pop() 删除末尾元素，弹出栈顶元素
+ pop(2) 按索引删除任意元素
+ remove('X') 根据值删除元素，但只删除第一个，若有多个，则需用到循环


## 3.3 组织列表
+ 变量.sort()永久排序，按字母顺序排列，大写字母和数字都排在前面
+ 逆序：变量.sort(reverse=True) reverse默认为False
+ sorted函数临时排序：print(sorted(变量))
+ 确定长度：函数len()

# 4 操作列表

## 4.1 遍历列表(Traversal)
遍历是二叉树上进行其他运算的基础
for：需要对列表中的每个元素都执行相同操作
循环：让计算机自动完成重复工作的常见方式


```python
magicians=["David","Tom","Jack"]
for magician in magicians:
    print(magician)
# 值得注意的是 for后的语句应为缩进块，还有冒号也很重要
# 为什么要缩进？因为这也是for循环的一部分
```

## 4.3 创建数值列表
**函数range()** 让你生成一系列数。


```python
for value in range(1,5):
    print(value)
#这里只会打印1，2，3，4
#差一行为所造成的后果
#range(6)则返回0~5
```


```python
#指定步长
musician=list(range(2,11,2))
#从2开始，不断加2，直到11
```


```python
#将前十个整数的平方加入列表中
squares=[]
for value in range(1,11):
    square=value ** 2
    squares.append(square)
print(squares)
#为了简洁，也可以：
for value in range(1,11):
    squares.append(value ** 2)
print(squares)
```

### 4.3.3 数字列表的简单统计运算
min() 最小值
max() 最大值
sum() 总和

### 4.3.4 列表解析
将for循环和创建新元素的代码合并成一行，并自动附加新元素
可以让行数更少，变得更简洁


```python
squares=[value**2 for value in range(1,11)]
print(squares)
```

## 4.3 使用列表的一部分
切片：

```print(musicians[0:3]) #注意：仅包含前三位```

 [:4] 则默认从头开始
 [2:] 终止于列表末尾

```python
a[0:3] #数据切片，左闭右开区间
a[-1] #-1表示倒数第一个数
a[-3:-1] #负索引切片
a[:3]
a[-3:] #不写的话默认从左边第一个开始或者取到右边最后一个
a[2:9:3] #间隔取值，前面区间仍然是左闭右开，后面为步长，同样也适用于负数的情况
```

 遍历切片：
```python
for musician in musicians[1:4]:
    print(player.title())
```

复制：创造一个副本而非直接改变原有列表
```A=B[:] ```

## 4.5 元组(tuple)
**不可变的列表**
使用**()**而非[]
不可修改！不可使用pop等方法！
元组由逗号标识，圆括号是为了更整洁清晰，if只含一个元素，则必须加上逗号：
musician=(me,)
遍历方法与列表相同

### 4.5.3 修改元组变量
虽然不能修改元素，但是可以给存储元组的变量赋值，可以重定义整个元组，这是合法的

## 设置代码格式
PEP8
缩进：默认四个空格
行长：最多不超过80字符，注释不超过72字符
空行：将代码不同部分分开

# 5 if语句与循环
== 相等
！=不相等

musicians.py


```python
musicians=['me','yoshiki','hyde']
for musician in musicians:
    if musician == 'me':
        print(musician.upper())
    else:
        print(musician.title())
#同样地，不要漏掉冒号和缩进

```

## 5.2 条件测试
car='BMW'
True
False
也可数值比较

检查多个条件：and or
类似与或门
检查特定值是否在列表中：in/not in
布尔表达式：A=True

## 5.3 if语句：


```python
#简单if语句
if 1 is True:
    print("go")
#if-else语句
#if-elif-else结构
#多elif代码块
```

if处理列表：
检查特殊元素
确定列表非空
使用多个列表

PEP8：
== <= >=等比较运算符两边各添加一个空格



## 5.4 循环结构
### 5.4.1 for-in循环
如果明确知道循环执行的次数，我们推荐使用**for-in**循环，例如上面说的那个重复3600次的场景，我们可以用下面的代码来实现。 注意，被for-in循环控制的代码块也是通过缩进的方式来构造，这一点跟分支结构中构造代码块的做法是一样的。我们被for-in循环控制的代码块称为循环体，通常循环体中的语句会根据循环的设定被重复执行。
sleep_hw.py
```python
# 每隔1秒输出一次“hello, world”，持续1小时
import time

for i in range(3600):
    print('hello, world')
    time.sleep(1)
```
Ctrl+C或者终止按钮来结束

sum_for_in.py
```python
# 从1到100的整数求和

total = 0
for i in range(1, 101):
    total += i
print(total)

# 内置sum函数，可省略循环结构
print(sum(range(2, 101, 2)))
```

### 5.4.2 while循环
sum_while.py
```python
# 从1到100的整数求和

total = 0
i = 1
while i <= 100:
    total += i
    i += 1
print(total)
```

### 5.4.3 break和continue
与C的用法类似

### 5.4.4 嵌套的循环结构
9x9_list.py
```python
# 打印九九乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f'{i}×{j}={i * j}', end='\t')
    print()
```

### 5.4.5 循环结构用法举例
要求：输入一个大于1的正整数，判断它是不是素数。
prime_number_judge.py
```python
# 判断素数
num = int(input('请输入一个正整数: '))
end = int(num ** 0.5)
is_prime = True
for i in range(2, end + 1):
    if num % i == 0:
        is_prime = False
        break
if is_prime:
    print(f'{num}是素数')
else:
    print(f'{num}不是素数')
```

要求：输入两个大于0的正整数，求两个数的最大公约数。
GCD.py
```python
x = int(input('x = '))
y = int(input('y = '))
for i in range(x, 0, -1):
    if x % i == 0 and y % i == 0:
        print(f'最大公约数: {i}')
        break
```

GCD1.py
欧几里得算法（辗转相除法）
```python
x = int(input('x = '))
y = int(input('y = '))
while y % x != 0:
    x, y = y % x, x
print(f'最大公约数: {x}')
```

要求：计算机出一个1到100之间的随机数，玩家输入自己猜的数字，计算机给出对应的提示信息“大一点”、“小一点”或“猜对了”，如果玩家猜中了数字，计算机提示用户一共猜了多少次，游戏结束，否则游戏继续。

guess_num.py
```python
import random

answer = random.randrange(1, 101)
counter = 0
while True:
    counter += 1
    num = int(input('请输入: '))
    if num < answer:
        print('大一点.')
    elif num > answer:
        print('小一点.')
    else:
        print('猜对了.')
        break
print(f'你一共猜了{counter}次.')
```

### 5.4.6 小结
**如果事先知道循环结构重复的次数，我们通常使用for循环；如果循环结构的重复次数不能确定，可以用while循环。**此外，我们可以在循环结构中使用break终止循环，也可以在循环结构中使用continue关键字让循环结构直接进入下一轮次。

# 6 字典（dict）和集合（set）
这次应该使用大括号
字典是一系列键值对，每个键都与一个值相关联，可以用键来访问相关联的值


```python
Aqua={'sex':'female',"age":"19"}
print(Aqua['sex'])
print(Aqua['age'])
Aqua['skill']='hack'
#修改键值对
del Aqua['age']
#删除键值对
aquasex=Aqua.get('height','no height')?
```


```python
#遍历字典
#遍历键值对：
for a,b in Aqua.items():
    print(a,b)
```


```python
#仅遍历所有键
for item in Aqua:
for item in Aqua.keys():
#keys()更为明显，但也可不写
```


```python
#按特定顺序遍历字典中的所有键：
for items in sorted(Aqua.keys()):
```


```python
#遍历字典中的所有值：
for name in Aqua.values():
```

剔除重复项再遍历列表：
使用集合set(),利用集合的互异性：


```python
for name in set(Aqua.values()):
#也可使用一对花括号直接创建集合，用逗号隔开
```

## 6.4 嵌套
不宜太多
### 字典列表：


```python
alien_0 = {'color': 'green', 'points': 5}
alien_1 = {'color': 'yellow', 'points': 10}
alien_2 = {'color': 'red', 'points': 15}
aliens = [alien_0, alien_1, alien_2]
for alien in aliens:
    print(alien)
```

### 在字典中储存列表


```python
pizza = {
'crust': 'thick',
'toppings': ['mushrooms', 'extra cheese'],
}
# 概述所点的比萨。
print(f"You ordered a {pizza['crust']}-crust pizza "
"with the following toppings:")
for topping in pizza['toppings']:
    print("topping")
```

### 典中典


```python
users = {
'aeinstein': {
'first': 'albert',
'last': 'einstein',
'location': 'princeton',
},
'mcurie': {
'first': 'marie',
'last': 'curie',
'location': 'paris',
},
}
for username, user_info in users.items():
    print(f"\nUsername: {username}")
full_name = f"{user_info['first']} {user_info['last']}"
location = user_info['location']
print(f"\tFull name: {full_name.title()}")
print(f"\tLocation: {location.title()}")
```
## 6.5 集合set

Python用**{}**来生成集合，集合**不含有相同元素**

```python
s = {2,3,4,2}#自动删除重复数据
s
```
{2,3,4}

```python
s.add(1) #如果加入集合中已有元素没有任何影响
s
```
{1, 2, 3, 4}

```python
s1 = {2,3,5,6}

s & s1 #取交集 {2, 3}
s | s1 #取并集 {1, 2, 3, 4, 5, 6}
s - s1 #差集 {1, 4}
```
```python
s.remove(5) #删除集合s中一个元素，注意只能删除一次
s.pop() #随机删除任意一个元素，删完后报错
s.clear() #一键清空
s
```

## 6.6 可变对象和不可变对象
可变对象可以对其进行插入，删除等操作，不可变对象不可以对其进行有改变的操作。Python中列表，字典，集合等都是可变的，元组，字符串，整型等都是不可变的。

### 6.6.1 类型转换
```python
int()
float()
type()#显示类型
list()#字符串分成字符列表
```
## 6.7 推导式与数据结构

在Python中，推导式是一种简洁而强大的语法特性，可以用来创建列表、集合、字典等数据结构。

### 6.7.1 列表推导式

列表推导式是Python中最常见的推导式，它允许我们用一行代码创建新的列表。
一些实例：

list_comprehension.py

```python
# 生成1到10的平方列表
squares = [x**2 for x in range(1, 11)]

# 过滤偶数，生成仅包含奇数的列表
odd_numbers = [x for x in range(1, 11) if x % 2 != 0]

# 使用条件表达式生成包含奇偶性判断的列表
evenodd_label = ["Even" if x % 2 == 0 else "Odd" for x in range(1, 11)]
```

### 6.7.2 集合推导式

集合推导式类似于列表推导式，但生成的是集合。
一些实例：

set_comprehension.py

```python
# 生成1到10的平方集合
squares_set = {x**2 for x in range(1, 11)}

# 过滤偶数，生成仅包含奇数的集合
odd_numbers_set = {x for x in range(1, 11) if x % 2 != 0}
```

### 6.7.3 字典推导式

字典推导式可以用一行代码生成字典，非常适用于从其他数据结构构建字典的场景。
一些实例：

dict_comprehension.py

```python
# 生成1到10的数字与其平方的字典
squares_dict = {x: x**2 for x in range(1, 11)}

# 过滤偶数，生成仅包含奇数的字典
odd_numbers_dict = {x: "Odd" for x in range(1, 11) if x % 2 != 0}
```

### 6.7.4 推导式的性能

在Python中，推导式是一种优雅而简洁的语法特性，但在使用时需要注意其性能，特别是在处理大规模数据时。我们将对比推导式与传统循环在性能上的差异，并讨论何时选择使用推导式以及何时选择传统循环。

#### 6.7.4.1 推导式 vs. 传统循环

考虑以下示例，使用列表推导式和传统循环分别生成包含1到1000的平方的列表：
列表推导式：
```squares = [x**2 for x in range(1, 1001)]```

传统循环：

```python
squares = []
for x in range(1, 1001):
    squares.append(x**2)
```

#### 6.7.4.2 性能比较

为了比较性能，我们可以使用Python内置的`timeit`模块来测量代码的执行时间。以下是一个简单的性能测试：

```python
import timeit
# 列表推导式性能测试
time_list_comprehension = timeit.timeit('[x**2 for x in range(1, 1001)]', number=1000)

# 传统循环性能测试
time_for_loop = timeit.timeit("""squares = []
for x in range(1, 1001):
    squares.append(x**2)
""", number=1000)

print(f"列表推导式执行时间: {time_list_comprehension} 秒")
print(f"传统循环执行时间: {time_for_loop} 秒")  
```

#### 6.7.4.3 何时选择推导式？

推导式在简化代码和提高可读性方面具有优势，但在性能上可能不总是最佳选择。推导式通常在处理简单任务和小型数据集时表现良好。当任务变得复杂或数据规模庞大时，传统循环可能更具有灵活性和控制性。

选择使用推导式的情况：

- 任务简单，代码清晰易读。
- 处理小型数据集，性能影响可接受。

#### 6.7.4.3 何时选择传统循环？

传统循环在处理复杂任务和大规模数据集时可能更为适用。它提供更多的灵活性和可控性，有时可以通过手动优化提高性能。

**选择使用传统循环的情况：**

- 任务复杂，需要更多控制结构。
- 处理大型数据集，性能至关重要。

### 6.7.5 嵌套推导式

嵌套推导式是Python中强大的特性之一，它允许在一个推导式内部包含另一个推导式，用于处理多维数据结构，例如二维列表或字典。本节将深入讨论嵌套推导式的用法以及在处理多维数据时的应用场景。

#### 6.7.5.1 二维列表的创建

考虑一个场景，想要创建一个5x5的九九乘法表。嵌套推导式能够以紧凑而清晰的方式实现这个任务：

2_dim_list.py

```python
multiplication_table = [[i * j for j in range(1, 6)] for i in range(1, 6)]
print(multiplication_table)
```

这个嵌套推导式创建了一个包含九九乘法表的二维列表，其中外层推导式负责生成每一行，内层推导式负责生成每一列的元素。这种结构使得代码易于理解且高度可读。

#### 6.7.5.2 过滤嵌套列表

嵌套推导式也可以用于过滤多维数据结构。例如，想要获取九九乘法表中所有元素值为偶数的项：

filt.py

```python
coordinate_dict = {(x, y): x * y for x in range(1, 4) for y in range(1, 4)}
print(coordinate_dict)
```

这个例子中，通过在嵌套推导式中添加条件语句，可以轻松地筛选出符合条件的元素。

#### 6.7.5.3 字典的嵌套推导式

嵌套推导式同样适用于字典的创建和过滤。考虑一个示例，想要创建一个包含坐标点的字典，其中x和y的取值范围在1到3之间：

nest_dict.py

```python
coordinate_dict = {(x, y): x * y for x in range(1, 4) for y in range(1, 4)}
print(coordinate_dict)
```

这个嵌套推导式创建了一个字典，其中包含了所有可能的坐标点及其对应的乘积。

#### 6.7.5.4 多重条件的嵌套推导式

在嵌套推导式中，可以使用多重条件来进一步筛选数据。例如，如果只想要九九乘法表中那些乘积大于10的元素：

```python
filtered_values = [value for row in multiplication_table for value in row if value > 10]
print(filtered_values)
```

通过添加条件语句，我们可以方便地实现对多维数据结构的复杂过滤操作。

### 6.7.6 推导式的应用场景

推导式是Python中一项强大而灵活的语法特性，适用于各种数据处理、过滤和转换场景。以下是几个实际案例，展示推导式在不同应用场景中的优雅应用。

#### 6.7.6.1 数据初始化

在某些情况下，需要初始化一个包含特定值的列表。使用列表推导式可以使这个过程非常简洁：

data_init.py

```python
# 初始化一个包含5个零的列表
zero_list = [0 for _ in range(5)]
print(zero_list)
```

#### 6.7.6.2 过滤与转换

假设有一个包含数字的列表，想要生成一个新列表，其中包含原列表中每个元素的平方值，但仅保留偶数的平方值：

filt_tran.py

```python
original_list = [1, 2, 3, 4, 5]

# 使用推导式过滤与转换
squares_of_evens = [x**2 for x in original_list if x % 2 == 0]
print(squares_of_evens)
```

#### 6.7.6.3 字典构建

字典推导式可以用于从其他数据结构构建字典。考虑一个场景，有两个列表，一个表示城市，另一个表示对应的人口数量：

create_dict.py

```python
cities = ['New York', 'Tokyo', 'London']
populations = [8537673, 37977073, 9304016]

# 使用字典推导式构建城市与人口的字典
city_population_dict = {city: population for city, population in zip(cities, populations)}
print(city_population_dict)
```

#### 6.7.6.4 文件读取与处理

在处理文件时，推导式可以快速生成某些数据的列表。例如，读取一个文件中的每一行，提取所有大写单词：

```python
with open('sample.txt', 'r') as file:
    uppercase_words = [word for line in file for word in line.split() if word.isupper()]
print(uppercase_words)
```

#### 6.7.6.5 多维数据处理

使用嵌套推导式可以轻松处理多维数据结构，例如创建一个包含矩阵转置的列表：

```python
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 使用嵌套推导式进行矩阵转置
transposed_matrix = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print(transposed_matrix)
```

### 6.7.7 总结

Python推导式是一项强大的语法特性，以其简洁而灵活的语法，广泛应用于数据处理、过滤、转换等多个场景。通过本文的详细讲解，深入了解了列表推导式、集合推导式、字典推导式以及嵌套推导式的使用方法和优势。

在实际应用中，列表推导式在数据初始化和快速过滤转换上表现出色，集合推导式适用于生成独一无二的元素集合，而字典推导式则为从不同数据结构构建字典提供了简洁的语法。嵌套推导式则在处理多维数据结构时展现出其独特优势，使得代码更为清晰和可读。

此外，还对推导式的性能进行了比较，提供了选择使用推导式或传统循环的指导。推导式在简化代码和提高可读性方面表现优越，但在处理复杂任务和大规模数据时，开发者应该谨慎选择以平衡性能和代码结构。

通过实际案例的展示，能够更好地理解何时选择使用推导式，并学会灵活运用不同类型的推导式来提高代码的简洁性和效率。在实际应用中，合理运用推导式将为Python开发者带来更高的开发效率和代码质量。

# 7 用户输入和while循环

## 7.1 input()工作原理
### 7.1.1 input()函数：让用户输入内容(**默认字符串**)

```python
message=input("Give me something.")
print(message)
```

### 7.1.2 int()函数:字符串转数值

```python
height = input("How tall are you, in inches? ")
height = int(height)
if height >= 48:
    print("\nYou're tall enough to ride!")
else:
    print("\nYou'll be able to ride when you're a little older.")
```
### 7.1.3 求模运算符(%)

## 7.2 while循环
### 7.1.1 使用while循环
从1数到5：
```python
current_number=1
while current_number <= 5:
	print(current_number)
	current_number += 1
	#缩进便代表被缩进的内容在循环之中
```

### 7.2.2 让用户选择何时退出
```python
prom = "\nrepeat:"
prom += "\nquit"
message = ""
while message != 'quit'
	message = input(prom)
	#避免再次打印quit
	if message != 'quit':
	print(message)
```

### 7.2.3 使用标志
在要求很多条件都满足才继续运行的程序中，可定义一个变量，用于判断整个程序是否处于活动状态。这个变量称为**标志 (flag)**，充当程序的交通信号灯。可以程序在标志为True 时继续运行，并在**任何**事件导致标志的值为False 时让程序停止运行。这样，在while 语句中就只需检查一个条件:标志的当前值是否为True。然后将所有其他测试(是否发生了应将标志设置为False 的事件) 都放在其他地方，从而让程序更整洁。
```python
prom = "\nrepeat:"
prom += "\nquit"
flag = True
while flag:
	message = input(prom)
	
	#避免再次打印quit
	if message == 'quit':
	flag = False
	else:
	print(message)
```

### 7.2.4/5 使用break，continue
与C中break，continue类同
**避免无限循环！** 确认至少有一个地方让条件为False或者有break语句

## 7.3 while循环与列表和字典
便于记录大量的用户和信息
for适合遍历，但并不适合修改，此时我们使用while循环

### 7.3.1 列表间移动元素
```python
#初始化列表
users0 = ['A','B','C']
users1 = []
#验证每个用户，并将已验证用户放进对应列表里
while users0:
    user_c = users0.pop()
    
    print(f"Verifying:{user_c.title()}")	
    users1.append(user_c)
   
#显示所有已验证用户
for user1 in users1:
    print(user1.title())
```

### 7.3.2 删除为特定值的所有列表元素
第3章中使用的remove()，前提是删除的值只在列表中出现一次，如果要删除所有，要借助while循环：
```python
pets = ['dog','cat','dog','python']
print(pets)

while 'cat' in pets:
	pets.remove('cat')
	
print(pets)
```

### 7.3.3 使用用户输入来填充字典
依旧借助while循环
```python
responses = {}

flag = True

while flag:
	name=input("\nTell me ur name")
	response=input("\nFavourite mountain?")
        
    responses[name]=response
    
    repeat=input("Anyone else?")
    if repeat == 'no':
    	flag=False
  
for name,response in responses.items():
	print(f"{name}would u like to climb {response}")
```

# 8 函数
## 8.1 函数定义
```python
def greet():
	"""问候"""#称为文档字符串的注释
	print("Hello!")
	
greet()
```

### 8.1.1 向函数传递信息
```python
def greet(name):#name是形参
	"""问候"""#称为文档字符串的注释
	print(f"Hello!{name.title()}!")
greet('Jack')#'Jack'是实参
```

## 8.2 传递实参
**位置实参**：要求实参与形参顺序相同
**关键字实参**：每个实参由变量名和值组成，可使用列表和字典

### 8.2.1 位置实参
最简单的关联方式：基于实参的顺序
如果顺序颠倒，将会出现可笑的后果
```python
def des_pet(anitype,name):
	"""显示宠物信息"""
	print(f"\nI have a {anitype}")
	print(f"My{anitype}'s name is {name.title()}")
	
des_pet('hamster','harry')
des_pet('cat','kitty')#多次调用函数
```

### 8.2.2 关键字实参
传递给函数的**名称值对**，由于是直接关联，所以不会混淆
无需考虑顺序，还清楚地指出了函数调用中各个值的用途
```python
def des_pet(anitype,name):
	"""显示宠物信息"""
	print(f"\nI have a {anitype}")
	print(f"My{anitype}'s name is {name.title()}")
	
des_pet(anitype='hamster',name='harry')
des_pet(anitype='cat',name='kitty')#多次调用函数
```

### 8.2.3 默认值
可给每个形参指定默认值，即可在调用中省略实参
例如，实际情况是宠物大多是小狗
```python
def des_pet(name,anitype='dog'):
	"""显示宠物信息"""
	print(f"\nI have a {anitype}")
	print(f"My{anitype}'s name is {name.title()}")
	
des_pet(name='harry')#或者是des_pet('harry')
des_pet(anitype='cat',name='kitty')#多次调用函数
```
***注意：使用默认值时，必须先在形参列表列出无默认值的形参，再列出有默认值的***

### 8.2.4 等效的函数调用
如上默认值以及两种实参调用方法

## 8.3 返回值
return语句

### 8.3.1 返回简单值
```python
def get_name(first_name,last_name):
	"""返回整洁的姓名"""
	full_name = f"{first_name} {last_name}"
	return full_name.title()
	
musician = get_name('jimi','hendrix')
print(musician)
```

### 8.3.2 让实参变为可选的
如果musician有中间名
```python
def get_name(first_name,middle_name,last_name):
	"""返回整洁的姓名"""
	full_name = f"{first_name} {middle_name} {last_name}"
	return full_name.title()
	
musician = get_name('jimi','lee','hendrix')
print(musician)
```
但并非所有人都有中间名，若没有，则不能正确运行，那么给middle_name指定一个空的默认值，使其变为可选：
```python
def get_name(first_name,last_name，middle_name=''):#有默认值移到最后！
	"""返回整洁的姓名"""
	if middle_name:
		full_name = f"{first_name} {middle_name} {last_name}"
	else:
		full_name = f"{first_name} {last_name}"
	return full_name.title()
	
musician = get_name('jimi','lee','hendrix')
print(musician)

musician = get_name('jimi','hendrix')
print(musician)
```

### 8.3.3 返回字典
```python
def get_name(first_name,last_name):
	"""返回一个字典，其中包含有关一个人的信息"""
	full_name = {'first':first_name,'last':last_name}
	return full_name
	
musician = get_name('jimi','hendrix')
print(musician)
```

下面的修改让其接受可选值年龄：
```python
def get_name(first_name,last_name，age=None):#None表示变量没有值(这里是占位值，在条件测试中相当于False)
	"""返回一个字典，其中包含有关一个人的信息"""
	full_name = {'first':first_name,'last':last_name}
	if age:
		person['age']=age
	return full_name
	
musician = get_name('jimi','hendrix',age=27)
print(musician)
```

### 8.3.4 函数和while循环结合
```python
def get_name(first_name,middle_name,last_name):
	"""返回整洁的姓名"""
	full_name = f"{first_name} {middle_name} {last_name}"
	return full_name.title()
while True：
	print("\nTell me ur name:")
	print("'q' to quit")
	f_name=input()
	if f_name=='q'
		break
	l_name=input()
	if l_name='q'
		break

musician = get_name(f_name,l_name)
print(musician)
```

## 8.4 传递列表
旨在提高处理列表的效率
```python
def greet_name(names):
	"""向列表中用户发出问候"""
	for name in names:
		msg=f"Hello,{name.title()}"
		print(msg)
		
usernames=['Bin','Shy','Tian']
greet_name(usernames)
```

### 8.4.1 在函数中修改列表
更容易维护：
```python
def print_model(un_designs,designs):
	"""
	模拟打印每个设计，直到没有未打印的设计
	打印每个设计后，都移到designs中
	"""
	while un_designs:
	c_design = un_designs.pop()
	print(f"Printing:{c_design}")
	designs.append(c_design)

def show(designs)
	"""显示打印好的所有模型"""
	for design in designs:
		print(design)
		
un_designs=['phone','robot','mg']
designs=[]

print_model(un_designs,designs)
show(designs)
```

### 8.4.2 禁止函数修改列表
将列表的副本传递给函数：
```python
function_name(list_name[:])
```
**切片表示法[:] 创建列表的副本**

## 8.5 传递任意数量的实参
如果预先不知道函数需要接受多少个实参，但py允许收集任意数量的实参
```python
def make pizza(*toppings):
	"""打印顾客点的所有配料。""
	print(toppings)
	
make_pizza('pepperoni')
make pizza ('mushrooms','green peppers','extra cheese')
```
形参名*toppings中的星号让Python创建一个名为toppings的空元组，并将收
到的所有值都封装到这个元组中。

**注意：Python将实参封装到一个元组中，即便函数只收到一个值**

### 8.5.1 结合使用位置实参和任意数量实参
如果要让函数接受不同类型的实参，必须在函数定义中将接纳任意数量实参的形参
放在最后。Python先匹配位置实参和关键字实参，再将余下的实参都收集到最后一
个形参中。
例如，如果前面的函数还需要一个表示比萨尺寸的形参，必须将其放在形参
*toppings的前面：
```python
def make pizza(size,*toppings):
	""概述要制作的比萨。"""
	print(f"\nMaking a {size}-inch pizza with the following toppings:"
	for topping in toppings:
		print(f"-(topping)")
		
make_pizza(16,'pepperoni')
make pizza(12,'mushrooms','green peppers','extra cheese')
```
*args 也收集任意数量的位置实参

### 8.5.2 使用任意数量的关键字实参
有时需要接受任意数量实参，但预先不知道传递给函数的是什么
可编写为能够接受任意数量的键值对
例如创建用户简介:
```python
def build(first,last,**user_info)##创建一个名为user_info的空字典，并将收到的所有名称值对都放到这个字典中
	"""创建一个包含我们知道的有关用户的一切的字典"""
	user_info['first_name']=first
	user_info['last_name']=last
	return user_info

user=build('albert','einstein',location='princeton',field='physics')

print(user)
```

**kwargs 也用于收集任意数量的关键字实参

## 8.6 将函数存储在模块中
利用**import语句**将函数存储在称为**模块**的独立文件中，再将模块导入到主程序
**向上抽象**

### 8.6.1 导入整个模块
模块是拓展名为 .py 的文件，包含要导入到程序中的代码。
有点类似C中的多文件
例如:pizza.py包含函数
make.py:
```python
import pizza

#此处就能随意调用pizza.py中的函数,用点号表明所属
pizza.make_pizza()
```

### 8.6.2 导入特定函数
借助from
```from module_name import function_0,function_1,function_2```

### 8.6.3 借助as给函数指定别名
避免名称冲突和提高效率，类似C的typedef
```python
from pizza import make_pizza as mp

mp(16,'mushrooms')
```
通用语法：
```from module_name import function_name as fn```

### 8.6.4 借助as给模块指定别名
通用语法：
```import module_name as mn```

### 8.6.5 导入模块中的所有函数
使用星号（ * ） 运算符
通用语法：
```from module_name import*```
然而，最好不要这样使用，避免不必要的麻烦

# 9 类

在面向对象编程中，你编写表示现实世界中的事物和情景的类，并基于这些类来创建对象（**实例化**）。编写类时，你定义一大类对象都有的通用行为。基于类创建对象时，每个对象都自动具备这种通用行为，然后可根据需要赋予每个对象独特的个性。

**面向对象编程**（Object-Oriented Programming，简称OOP）是一种编程范式，它以对象为中心，通过定义类和对象来组织和管理代码。

在Python中，一切皆为对象。**对象是类的实例化，类是对象的蓝图或模板。**通过定义类，我们可以创建具有相同属性和行为的多个对象。

## 9.1 创建和使用类

使用类几乎可以模拟任何东西。

**封装**(Encapsulation)

**封装**是将数据（属性）和行为（方法）捆绑在一起的过程。在Python中，封装通常是通过**创建类（class）**来实现的。类中的方法通常可以访问和修改类的内部状态，但是这些内部状态对于外部代码来说是隐藏的，这就是“封装”的含义。

好处：

1.  将变化隔离；
2.  便于使用；
3.  提高复用性；
4.  提高安全性；

封装原则：

1.  将不需要对外提供的内容都隐藏起来；
2.  把属性都隐藏，提供公共方法对其访问。



### 9.1.1 创建Dog类

根据Dog类创建的每个实例都将存储名字和年龄，我们赋予每条小狗sit()和roll()的能力

dog.py

```python
class Dog:#根据约定，在python中，首字母大写的名称指的是类
    """一次模拟小狗的简单尝试"""
    
    def __init__(self,name,age)
    """初始化属性name和age"""
    self.name=name#前缀self
    self.age=age#前缀self
    
    def sit(self)
    """模拟小狗收到命令时蹲下"""
    print(f"{self.name} is sitting")
    
    def roll(self)
    """模拟小狗收到命令时打滚"""
    print(f"{self.name} is rolling")
```

**类中的函数称为方法**，前面学到的关于函数的一切都适用于方法，目前而言，唯一重要的差别是**调用方法的方式**。

方法*__init*__()是一个特殊方法，左右的两个下划线是一种约定，为了避免默认方法与普通方法发生冲突。python中用双下划线开头的方式将属性隐藏起来（设置成私有的），这就是**私有变量**。

这种自动变形的特点：

1.类中定义的__x只能在内部使用，如self.__x，引用的就是变形的结果。

2.这种变形其实正是针对外部的变形，在外部是无法通过__x这个名字访问到的。

3.在子类定义的__x不会覆盖在父类定义的__x，因为子类中变形成了：_子类名__x,而父类中变形成了：_父类名__x，即双下滑线开头的属性在继承给子类时，子类是无法覆盖的。

这种变形需要注意的问题是：

1.这种机制也并没有真正意义上限制我们从外部直接访问属性，知道了类名和属性名就可以拼出名字：_类名__属性，然后就可以访问了，如a._A__N

2.变形的过程只在类的内部生效,在定义后的赋值操作，不会变形



这里包含三个形参：self,name,age。

+ self必不可少，而且必须位于其他形参的前面。Python在调用时会自动传入实参self，这是一个指向实例本身的引用，让实例能够访问类中的属性和方法。由于它会自动传递，所以无需传递他，仅给其他形参提供值便好。
+ 以self为前缀的变量可供类中的所有方法使用，可以通过类的任何实例来访问。像这样可通过实例访问的变量称为**属性**。**属性是对象的特性或数据，它定义了对象的状态。**每个对象都有一组属性，这些属性可以是数字、字符串、列表等不同类型的数据。例如，在Person类中，姓名和年龄可以是对象的属性。
+ 另外两个方法sit()和roll(),执行时不需要额外信息，因此只有一个形参self，我们随后将创建的实例能够访问这些方法，换句话说，它们都会蹲下和打滚。



私有方法
在继承中，父类如果不想让子类覆盖自己的方法，可以将方法定义为私有的

### 9.1.2 根据类创建实例

可将类视为有关如何创建实例的说明。下面来创建一个表示特定小狗的实例：

```python
class Dog:
    --snip--
my_dog=Dog('Willie',6)#使用这两个实参调用Dog类的方法__init__()

print(f"My dog's name is {my_dog.name}")
print(f"My dog's age is {my_dog.age}")

```

通常可认为首字母大写的名称指的是类，而小写的名称指的是根据类创建的实例

#### a.访问属性

句点表示法

```python
my_dog.name# 访问my_dog属性name的值
```

#### b.调用方法

下面来让小狗蹲下和打滚

```python
class Dog:
    --snip--
my_dog=Dog('Willie',6)#使用这两个实参调用Dog类的方法__init__()

my_dog.sit()
my_dog.roll()

```

#### c.多个实例

```python
#创建实例
my_dog=Dog('Willie',6)#使用这两个实参调用Dog类的方法__init__()
ur_dog=Dog('Lucy',3)

print(f"My dog's name is {my_dog.name}")
print(f"My dog's age is {my_dog.age}")
print(f"Ur dog's name is {ur_dog.name}")
print(f"Ur dog's age is {ur_dog.age}")

#调用方法
my_dog.sit()
my_dog.roll()
ur_dog.sit()
ur_dog.roll()
```



## 9.2 使用类和实例

### 9.2.1 Car类

car.py

```python
class Car:
    """模拟汽车"""

    def __init__(self,make,model,year):
        """初始化描述汽车属性"""
        self.make=make
        self.model=model
        self.year=year

    def get_name(self):
        """返回整洁的描述性信息"""
        name=f"{self.make} {self.model} {self.year}"
        return name.title()

my_new_car=Car('audi','a4',2019)
print(my_new_car.get_name())
```



### 9.2.2 给属性指定默认值

例如读取汽车的里程表(默认初始值为0)：

```python
class Car:
    """模拟汽车"""

    def __init__(self,make,model,year):# 默认值此处无需声明
        """初始化描述汽车属性"""
        self.make=make
        self.model=model
        self.year=year
        self.meter = 0 #指定默认值

    def get_name(self):
        """返回整洁的描述性信息"""
        name=f"{self.make} {self.model} {self.year}"
        return name.title()

    def read_meter(self):
        """指出汽车里程"""
        print(f"Has {self.meter} miles on it")

my_new_car=Car('audi','a4',2019)
print(my_new_car.get_name())
my_new_car.read_meter()
```



### 9.2.4 修改属性的值
我们有三种方法修改属性的值：
+ 直接通过实例进行修改
+ 通过方法进行设置
+ 通过方法进行递增（增加特定值）

#### a.直接修改属性的值
```python
class Car:
	--snip--
	
my_new_car=Car('audi','a4',2019)
print(my_new_car.get_name())

my_new_car.meter=23
my_new_car.read_meter()
```

#### b.通过方法修改属性的值
这样是大有裨益的，无需直接访问属性，而可将值传递给方法，由它在内部进行更新。
```python
 def update(self,mile):
        """将里程表读数设置为指定值"""
        self.meter=mile
        
my_new_car.update(24)
my_new_car.read_meter()
```

加入禁止回调功能：
```python
 def update(self,mile):
        """
        将里程表读数设置为指定值
        并禁止回调
        """
        if mile >= self.meter:
            self.meter=mile
        else:
            print("U can't roll back!")
            
my_new_car.update(1)
```

#### c.通过方法对属性的值进行递增
有时需要递增而不是设置为新的值
```python
 def increase(self,mi):
        """增加指定的量"""
        self.meter += mi

my_old_car=Car('subaru','outback',2015)
print(my_old_car.get_name())

my_old_car.update(23_500)
my_old_car.read_meter()

my_old_car.increase(100)
my_old_car.read_meter()
```

## 9.3 继承(Inheritance)，组合，抽象与派生
编写类时，并非总是从空白开始。如果要编写的类是另一个现成类的特殊版本，可使用**继承**。一个类**继承**另一个类时，将自动获得另一个类的所有属性和方法。原有的类称为**父类**，而新类称为**子类**。子类继承了父类的所有属性和方法，同时还可以定义自己的属性和方法。



另外，有别于继承，还有组合的概念。

**组合**：组合指的是，在一个类中以另外一个类的对象（也就是实例）作为数据属性，称为类的组合
也就是说：**一个类的属性是另一个类的对象，就是组合**



**先抽象后继承**：

**抽象**即抽取类似或者说比较像的部分。 抽象分成两个层次：

1. 将奥巴马和梅西这俩对象比较像的部分抽取成类；
2. 将人，猪，狗这三个类比较像的部分抽取成父类。

抽象最主要的作用是划分类别（可以隔离关注点，降低复杂度） 

**继承：** 是基于抽象的结果，通过编程语言去实现它，肯定是先经历抽象这个过程，才能通过继承的方式去表达出抽象的结构。
抽象只是分析和设计的过程中，一个动作或者说一种技巧，通过抽象可以得到类



**派生**：

子类也可以添加自己新的属性或者在自己这里重新定义这些属性（不会影响到父类），需要注意的是，一旦重新定义了自己的属性且与父类重名，那么调用新增的属性时，就以自己为准了

通过继承建立了派生类与基类之间的关系，它是一种'是'的关系，比如白马是马，人是动物。
当类之间有很多相同的功能，提取这些共同的功能做成基类，用继承比较好，比如教授是老师

### 9.3.1 子类的方法*__init*__()

在既有类的基础上编写新类时，通常要调用父类的方法*__init*__(),这将初始化父类此方法中定义的所有属性，从而让子类包含这些属性。

ele_car.py

```python
class Car:
    """模拟汽车"""

    def __init__(self,make,model,year):# 默认值此处无需声明
        """初始化描述汽车属性"""
        self.make=make
        self.model=model
        self.year=year
        self.meter = 0 #指定默认值

    def get_name(self):
        """返回整洁的描述性信息"""
        name=f"{self.make} {self.model} {self.year}"
        return name.title()

    def read_meter(self):
        """指出汽车里程"""
        print(f"Has {self.meter} miles on it")

    def update(self,mile):
        """
        将里程表读数设置为指定值
        并禁止回调
        """
        if mile >= self.meter:
            self.meter=mile
        else:
            print("U can't roll back!")

    def increase(self,mi):
        """增加指定的量"""
        self.meter += mi

# 创建子类时，父类必须包含在当前文件中，且位于子类之前。
class elecar(Car):# 在括号内指定父类的名称
    """电动汽车的独特之处"""

    def __init__(self, make, model, year):  # 接受创建Car实例所需的信息
        """初始化父类的属性"""
        super().__init__(make, model, year)# super()是一个特殊函数，让你能够调用父类的方法(父类也称超类)

tesla=elecar('tesla','model s',2019)
print(tesla.get_name())
```

elecar实例的行为与Car实例一样，现在可以开始定义特有的属性和方法了



### 9.3.2 给子类定义属性和方法

继承完毕，开始添加用于区分的新属性和新方法。

下面添加电瓶容量和描述属性的方法：

```python
class Car:
    """模拟汽车"""

    def __init__(self,make,model,year):# 默认值此处无需声明
        """初始化描述汽车属性"""
        self.make=make
        self.model=model
        self.year=year
        self.meter = 0 #指定默认值

    def get_name(self):
        """返回整洁的描述性信息"""
        name=f"{self.make} {self.model} {self.year}"
        return name.title()

    def read_meter(self):
        """指出汽车里程"""
        print(f"Has {self.meter} miles on it")

    def update(self,mile):
        """
        将里程表读数设置为指定值
        并禁止回调
        """
        if mile >= self.meter:
            self.meter=mile
        else:
            print("U can't roll back!")

    def increase(self,mi):
        """增加指定的量"""
        self.meter += mi

# 创建子类时，父类必须包含在当前文件中，且位于子类之前。
class elecar(Car):# 在括号内指定父类的名称
    """电动汽车的独特之处"""

    def __init__(self, make, model, year):  # 接受创建Car实例所需的信息
        """初始化父类的属性"""
        super().__init__(make, model, year)# super()是一个特殊函数，让你能够调用父类的方法(父类也称超类)
        # 添加特有属性
        self.battery=75

    def des_battery(self):
        """打印一条描述电瓶容量的信息"""
        print(f"Has {self.battery} -kWh")

tesla=elecar('tesla','model s',2019)
print(tesla.get_name())
tesla.des_battery()
```

如果一个特殊属性为Car所共有（并不特殊），那就应该加入Car类中，而不是elecar类



### 9.3.3 重写父类的方法

**多态**(Polymorphism)

对于父类的方法，只要它不符合子类模拟的实物的行为，都可以进行**重写**。为此，可在子类中定义一个与要重写的父类方法同名的方法。如此python不会考虑父类方法，而只关注子类中定义的相关方法。



### 9.3.4 将实例用作属性

可以将类的一部分提取出来，作为一个独立的类，可以将大型类拆分成多个协同工作的小类。不是一定要继承，也可以作为独立的一个类。

例如，如果有很多关于电瓶的属性和方法，可将它们提取出来，存入到一个Battery的类



### 9.3.5 模拟实物

新境界：从较高的逻辑层面（而不是语法层面）考虑

你要找到的是效率最高的表示法。



## 9.4 导入类

应该使文件变得更整洁。允许将类存储在模块中，然后再主程序中导入所需的模块。

### 9.4.1 导入单个类

my_car.py

```python
from car import Car

my_car = Car('audi','a4',2019)
print(my_car.get_name())

my_car.meter=23
my_car.read_meter()
```

整洁，符合Zen 的理念



### 9.4.2 在一个模块中存储多个类

把battery类和elecar类加入模块car.py中便可



### 9.4.3 从一个模块中导入多个类

已加入的前提下

my_car.py

```python
from car import Car,elecar
```



### 9.4.4 导入整个模块

导入整个模块，再用句点表示法访问需要的类

```python
import car

my_beetle=car.Car('volkswagen','beetle',2019)
print(my_beetle.get_name())
```

访问的语法是：# 与模块使用方法相同

```python
module_name.className
```



### 9.4.5 导入模块中的所有类

语法是：

```python
from module_name import *
```

不建议使用，原因有二：

+ 不利于明确看出使用了模块中的哪些类
+ 可能引发名称方面的疑惑



### 9.4.6 在一个模块中导入另一个模块

理解便可



### 9.4.7 使用别名

例如：（使用as与先前类同）

```python
from elecar import Elecar as EC
tesla=EC('tesla','roadster','2019')
```



### 9.4.8 自定义工作流程

灵活使用各种类和模块



## 9.5 Python标准库

这是其他程序员编写好的一组模块

例如random模块：

```python
from random import randint
randint(1,6)

from random import choice
players=['1','d','g']
g=choice(players)
g
```



## 9.6 类编码风格

类名应采用**驼峰命名法**，即将类名中的每个单词的首字母都大写，而不使用下划线。实例名和模块名都采用小写格式，并在单词之间加上下划线。

对于每个类，都应该紧跟在类定义后面包含一个文档字符串:简要的描述类的功能，并遵循编写函数的文档字符串时采用的格式约定，每个模块也应该包含，对其中的类可用于做什么进行描述。

空行可用来组织代码：类中一个空行来分割方法，模块中两个空行来分割类。

导入时，先导入标准库，再导入自己的模块。



## 9.7 封装 继承 多态的补充

### 封装（Encapsulation）

封装是将数据（属性）和行为（方法）捆绑在一起的过程。在Python中，封装通常是通过创建类（class）来实现的。类中的方法通常可以访问和修改类的内部状态，但是这些内部状态对于外部代码来说是隐藏的，这就是“封装”的含义。

### 继承（Inheritance）

继承是一种创建新类的方法，新创建的类（子类）继承了一个或多个类（父类）的特征（属性和方法）。子类可以添加新的属性和方法，也可以覆盖或扩展父类的行为。

### 多态（Polymorphism）

多态是指不同类的对象对于相同的方法有着不同的响应。在Python中，多态通常是通过方法重写来实现的，即子类可以提供父类方法的一个新版本。

### 适用范围

- 封装是为了保护对象的内部状态和隐藏对象的实现细节，使得代码模块化。
- 继承用于创建与已存在的类有共同特征的新类，它支持代码复用并建立类之间的关系。
- 多态允许我们使用统一的接口来操作不同的底层形式（数据类型），它使得代码更加灵活和可扩展。

在面向对象设计中，合理运用这三个概念可以让代码更加清晰、灵活和易于维护。







# 21 高阶函数淆

一个函数可以作为参数传给另外一个函数，或者一个函数的返回值为另外一个函数（若返回值为该函数本身，则为递归），满足其一则为高阶函数。
## 21.1 map函数
***根据提供的函数对指定序列做映射，并返回映射后的序列***
把函数 function 依次作用在list中的每一个元素上，得到一个新的list并返回。注意，map不改变原list，而是返回一个新list。

```map(function, iterable,...)```

参数：
+ function：函数，序列中的每个元素需要指定的操作，可以是匿名函数
+ iterable：一个或多个序列

返回值：python3中返回map类

### 21.1.1 正常使用：
map.py
```python
def add(x):
    """加1函数"""
    return x + 1

result = map(add, [1, 2, 3, 4, 5])
print（result）

# 如果是Python3
print （list(result)）

# 输出结果： [2, 3, 4, 5, 6]
```

### 21.1.2 lambda（匿名）函数：
```python
result = map(lambda x: x + 1, [1, 2, 3, 4, 5])
print （result）

# 如果使用Python3
print （list(result)）

# 输出结果：[2, 3, 4, 5, 6] 
```

### 21.1.3 多个序列
```python
result = map(lambda x, y: x + y, [1, 2, 3], [1, 2, 3])
print (result)

# 如果使用Python3
print (list(result))

# 输出结果：[2, 4, 6]

# 注意：如果俩个序列中值不等，会报错：
result = map(lambda x, y: x + y, [1, 2, 3], [1, 2, 3, 4, 5])
print result
# 报错信息如下：
Traceback (most recent call last):
  File "C:/Users/lh9/PycharmProjects/lvtest/apps/tests.py", line 2431, in <module>
    result = map(lambda x, y: x + y, [1, 2, 3], [1, 2, 3, 4, 5])
  File "C:/Users/lh9/PycharmProjects/lvtest/apps/tests.py", line 2431, in <lambda>
    result = map(lambda x, y: x + y, [1, 2, 3], [1, 2, 3, 4, 5])
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
```

### 21.1.4 Lambda函数的补充
Lambda表达式是一种在Python中快速创建匿名函数的方法。一个匿名函数就像一个一次性使用的函数，没有名字。你只需使用它一次，然后在程序中继续前进。Lambda 函数通常与 map() 和filter()结合使用。lambda函数的方法可以方便地以速记的形式快速定义一个可以与这些函数一起使用的函数。在本教程中，我们将介绍几个例子，说明如何在 Python 中定义 lambda 函数，以及如何有效地利用它们。

#### 21.1.4.1 如何得来？

一个Lambda函数只是普通Python函数的一个速记版本。所以为了更好地理解如何在Python中制作一个lambda函数，我们可以按部就班地将一个普通函数转换成一个lambda函数。首先，让我们看一个标准 Python 函数的简单例子。
```python
def plus_one(num):
    result = num + 1
    return result
    
plus_one(7)

#结果是8
```

1. 去掉了 result变量
 ```python
def plus_one(num):
    return num+1
    
plus_one(7)

#结果是8
 ```

2. 让函数成为一个单行代码
```python
def plus_one(num):return num+1

plus_one(7)

#结果是8
```

3. 删除def关键字
在这里，我们把def关键字和我们分配给我们的函数**（def plus_one()**）的名称和小括号一起删除。
```python
num:return num+1
```

4. 删除return关键字
兰姆达函数没有返回语句，因为任何兰姆达函数都会隐含返回语句。
```python
num:num+1
```

5. 添加lambda关键词
最后，我们可以在剥离后的表达式前面加上lambda关键字，瞧！这就是一个lambda函数。我们就有了一个lambda函数。
```python
lambda num:num+1
```

#### 21.1.4.2 最简单的用法：将Lambda分配给一个变量

使用lambda的第一种方式是简单地将它赋值给一个变量，然后将该变量用作一个函数。让我们看看这是怎么做的。
lambda.py
```python
plus_one = lambda num: num + 1

plus_one(7)

#输出8
```



#### 21.1.4.3 if语句与lambda

```python
result = map(lambda str: str.capitalize() if 'a' in str else str, 'abracadabra')

print(list(result))

#['A', 'b', 'r', 'A', 'c', 'A', 'd', 'A', 'b', 'r', 'A']

```



关于lambda与其他函数的配合使用，我们稍后再来看。

## 21.2 filter函数

在Python中，`filter`函数是一种内置的高阶函数，它能够接受一个函数和一个迭代器，然后返回一个新的迭代器，这个新的迭代器仅包含使给定函数返回True的原始元素。这个功能在许多情况下都非常有用，比如当你需要从一个大的数据集中筛选出满足某些条件的数据时。

### 21.2.1 基本用法

参数：
+ function – 函数名；
+ iterable – 序列或者可迭代对象；

返回值：通过 function 过滤后，将返回 True 的元素保存在迭代器对象中，最后返回这个迭代器对象（Python2.0x 版本是直接返回列表 list ）；
```python
filter(function, iterable)
```

在最基本的形式中，`filter`函数接受一个函数和一个迭代器，并返回一个新的迭代器，其中包含原始迭代器中使给定函数返回True的元素。这个函数通常被称为"谓词"，因为它应该返回一个布尔值。
filter.py
```python
def is_even(n):
    return n % 2 == 0

numbers = [1, 2, 3, 4, 5, 6]
even_numbers = filter(is_even, numbers)

print(list(even_numbers))  # 输出：[2, 4, 6]
```
在这个例子中，我们首先定义了一个函数is_even，这个函数接受一个数字并检查它是否是偶数。然后，我们创建了一个列表numbers。接着，我们使用filter函数和is_even函数来从numbers列表中筛选出偶数。最后，我们将filter对象转换为列表并打印结果。

### 21.2.2 lambda与filter
你可以使用匿名函数（也称为lambda函数）作为filter函数的第一个参数。这在你只需要在一个地方使用函数，并且函数的逻辑非常简单时非常有用。
```python
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = filter(lambda x: x % 2 == 0, numbers)

print(list(even_numbers))  # 输出：[2, 4, 6]
```
在这个例子中，我们直接在filter函数调用中定义了一个匿名函数。这个匿名函数接受一个数字并检查它是否是偶数。这与前面的例子完全相同，但是更加简洁。

### 21.2.3 filter与复杂数据结构
filter函数也可以处理更复杂的数据结构。例如，如果你有一个包含字典的列表，你可以使用filter函数来筛选出满足某些条件的字典。

下面是一个例子，我们使用filter函数筛选出年龄大于30的人：
```python
data = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}, {'name': 'Charlie', 'age':35}]
old_people = filter(lambda x: x['age'] > 30, data)

print(list(old_people))  # 输出：[{'name': 'Charlie', 'age': 35}]

```
在这个例子中，我们首先定义了一个包含字典的列表data，每个字典代表一个人，并含有他们的名字和年龄。然后我们使用filter函数和一个匿名函数来筛选出年龄大于30的人。

### 21.2.4 性能考虑
虽然filter函数可以方便地筛选数据，但如果你处理的数据集非常大，你可能需要考虑性能问题。由于filter函数返回的是一个迭代器，所以它只在需要的时候处理数据，这可以节省大量内存。
然而，如果你需要频繁地访问筛选后的数据，或者需要在多个地方使用它，你可能会发现直接使用列表推导式更加高效。这是因为filter函数每次迭代都会调用函数，而列表推导式则会立即计算结果。

### 21.2.5 总结
filter函数是Python中的一种强大的工具，可以帮助你方便地筛选数据。虽然它可能不如列表推导式在所有情况下都高效，但在处理大数据集或者复杂数据结构时，filter函数可以是一个非常有用的工具。

## 21.3 reduce函数

顾名思义，reduce() 函数将提供的函数应用于“可迭代对象”并返回单个值

### 21.3.1 基本用法

```python
reduce(function, iterables)
```

该函数指定在这种情况下应将哪个表达式应用于“可迭代对象”。必须使用功能工具模块来导入此功能。

例子：
reduce.py
```python
from functools import reduce

reduce(lambda a,b: a+b,[23,21,45,98])

#输出 187
```

示例中的 reduce 函数将列表中的每个可迭代对象一一添加并返回单个结果。













# 附录

## 1 Python错误，异常捕获和处理

错误可以分为两类：语法（解析）错误和逻辑（算法）错误。

异常是一个事件，该事件在程序执行过程中发生，影响程序的正常执行。一般情况下，Python无法正常处理程序时就会发生一个异常。

### 1.1 异常的处理
#### 1.1.1 try-except
它能够将可能出错的代码进行处理，处理后报错的红色字体将会转换成简短的、正常的字体，用法如下：
```python
try:
    有可能出现异常的代码
except  异常类型 as 变量
	处理后的代码
	
这里的try-except并不影响代码的运行，如果你的代码没有报错，就算是写了tyr-except，它也只会执行try那行代码。如果那行代码没有错误，那就不会执行except里面的代码。
```

#### 1.1.2 try-except-except
可用于判断多种可能报错的情况，类似多elif
用法如下：

```python
try:
    1 / 0
    print(a)
except NameError as s:  # 第一种写法，用as+变量
    print(s)
except ZeroDivisionError:  # 第二种写法，自定义输出内容
    print("除数不能为0")  # 自定义输出的内容
```
try-except的写法很灵活，我们同样可以用元组把可能报错的异常类型囊括进去，避免写多行except.

#### 1.1.3 try-except-else
如果没有异常，则执行else里面的代码

#### 1.1.4 try-except-finally
不管代码是否有异常，最后都会执行finally里面的代码

#### 1.1.5 顶层类Exception
except后面其实可以不加错误类型，因为系统会默认认为后面的错误是类型是Exception，这是1个顶层类，包含了所有的出错类型。

#### 1.1.6 小结

```python
try:    
    正常的操作    # 可能出现异常的代码块
except [异常类型]：    
    发生异常，执行这块代码      # 如果在try部份引发了'异常类型'异常
except [异常类型，数据]:    
    发生异常，执行这块代码      # 如果引发了'异常类型'异常，获得附加数据
except [异常类型]:    
    发生异常，执行这块代码
else:    
    如果没有异常执行这块代码
finally:    
    无论是否发生异常都将执行最后的代码
```

### 1.2 自定义异常

有没有发现，前面我们去做基本的异常捕获时，每次可能出错的地方就得写一个try-except，如果有多个地方可能会出错呢？是否我们需要写多个try-except？又或者理论上代码可以运行，但我想定一下规矩，凡是不符合我规矩的行为，我都让它出现异常，比如密码长度超出我规定的长度，我想让程序出现异常。

**自定义异常可用于引发一个异常（抛出一个异常），由关键字raise引发。**

举例：模拟用户输入密码的情景，用户输入的密码不能低于6位数，自定义一个异常，用于检测用户输入的密码是否符合规定，不符合则引发异常，提示当前输入的密码长度和最小密码长度不能低于6位数。
```python
class MyError(Exception):  # 异常捕获的类
    def __init__(self, length, min_len):  # length为用户输入的密码长度，min_len为规定的最小长度
        self.length = length
        self.min_len = min_len

    # 设置抛出异常的描述信息
    def __str__(self):
        return "你输入的长度是%s,不能少于%s" % (self.length, self.min_len)


def main():
    try:
        con = input("请输入密码：")  # 获取用户输入的密码
        l = len(con)  # 获取用户输入的密码长度
        if l < 6:
            raise MyError(l, 6)  # 长度低于设定的6位数则引发异常
    except Exception as ss:  # 有错误则提示
        print(ss)
    else:
        print("您的密码输入完毕")  # 没有错误则执行

main()
```