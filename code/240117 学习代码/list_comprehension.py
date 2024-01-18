# 生成1到10的平方列表
squares = [x**2 for x in range(1, 11)]

# 过滤偶数，生成仅包含奇数的列表
odd_numbers = [x for x in range(1, 11) if x % 2 != 0]

# 使用条件表达式生成包含奇偶性判断的列表
evenodd_label = ["Even" if x % 2 == 0 else "Odd" for x in range(1, 11)]