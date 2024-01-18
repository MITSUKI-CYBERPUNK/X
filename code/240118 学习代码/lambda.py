# 分配给变量
plus_one = lambda num: num + 1

plus_one(7)

# 结合if语句
result = map(lambda str: str.capitalize() if 'a' in str else str, 'abracadabra')

print(list(result))
