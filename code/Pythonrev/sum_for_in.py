# 从1到100的整数求和

total = 0
for i in range(1, 101):
    total += i
print(total)

# 内置sum函数，可省略循环结构
print(sum(range(1, 101, 1)))