def add(x):
    """加1函数"""
    return x + 1

result = map(add, [1, 2, 3, 4, 5])
print(result)

# 如果是Python3
print(list(result))

# 输出结果： [2, 3, 4, 5, 6]

# 结合lambda
result = map(lambda x: x + 1, [1, 2, 3, 4, 5])
print(result)

# 多个序列
result = map(lambda x, y: x + y, [1, 2, 3], [1, 2, 3])
print(result)