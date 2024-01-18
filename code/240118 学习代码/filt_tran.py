original_list = [1, 2, 3, 4, 5]

# 使用推导式过滤与转换
squares_of_evens = [x**2 for x in original_list if x % 2 == 0]
print(squares_of_evens)