#导入单个类
from car import Car# 多个用逗号

my_car = Car('audi','a4',2019)
print(my_car.get_name())

my_car.meter=23
my_car.read_meter()

#导入整个模块
import car

my_beetle=car.Car('volkswagen','beetle',2019)
print(my_beetle.get_name())
