# 生成1到10的平方集合
squares_set = {x**2 for x in range(1, 11)}

# 过滤偶数，生成仅包含奇数的集合
odd_numbers_set = {x for x in range(1, 11) if x % 2 != 0}