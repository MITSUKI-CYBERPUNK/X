cities = ['New York', 'Tokyo', 'London']
populations = [8537673, 37977073, 9304016]

# 使用字典推导式构建城市与人口的字典
city_population_dict = {city: population for city, population in zip(cities, populations)}
print(city_population_dict)