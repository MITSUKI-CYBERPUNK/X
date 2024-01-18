# 过滤偶数
def is_even(n):
    return n % 2 == 0

numbers = [1, 2, 3, 4, 5, 6]
even_numbers = filter(is_even, numbers)

print(list(even_numbers))  # 输出：[2, 4, 6]

# 结合lanmbda
numbers = [1, 2, 3, 4, 5, 6]
even_numbers = filter(lambda x: x % 2 == 0, numbers)

print(list(even_numbers))  # 输出：[2, 4, 6]

# 筛选年龄大于30的人
data = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}, {'name': 'Charlie', 'age':35}]
old_people = filter(lambda x: x['age'] > 30, data)

print(list(old_people))  # 输出：[{'name': 'Charlie', 'age': 35}]