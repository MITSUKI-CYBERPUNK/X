# 遍历列表(借助for循环)
magicians=["David","Tom","Jack"]
for magician in magicians:
    print(magician)


# 创建数值列表(借助range()函数)
for value in range(1,5):
    print(value)
#这里只会打印1，2，3，4
#差一行为所造成的后果
#range(6)则返回0~5
    
#指定步长
musician=list(range(2,11,2))
#从2开始，不断加2，直到11

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


# 切片
a=[1,2,3,4,5,6,7,8,9]
a[0:3] #数据切片，左闭右开区间
a[-1] #-1表示倒数第一个数
a[-3:-1] #负索引切片
a[:3]
a[-3:] #不写的话默认从左边第一个开始或者取到右边最后一个
a[2:9:3] #间隔取值，前面区间仍然是左闭右开，后面为步长，同样也适用于负数的情况