# 生成1到10的数字与其平方的字典
squares_dict = {x: x**2 for x in range(1, 11)}

# 过滤偶数，生成仅包含奇数的字典
odd_numbers_dict = {x: "Odd" for x in range(1, 11) if x % 2 != 0}