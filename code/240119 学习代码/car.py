class Car:
    """模拟汽车"""

    def __init__(self,make,model,year):# 默认值此处无需声明
        """初始化描述汽车属性"""
        self.make=make
        self.model=model
        self.year=year
        self.meter = 0 #指定默认值

    def get_name(self):
        """返回整洁的描述性信息"""
        name=f"{self.make} {self.model} {self.year}"
        return name.title()

    def read_meter(self):
        """指出汽车里程"""
        print(f"Has {self.meter} miles on it")

    def update(self,mile):
        """
        将里程表读数设置为指定值
        并禁止回调
        """
        if mile >= self.meter:
            self.meter=mile
        else:
            print("U can't roll back!")

    def increase(self,mi):
        """增加指定的量"""
        self.meter += mi

my_new_car=Car('audi','a4',2019)
print(my_new_car.get_name())
my_new_car.read_meter()

# a.直接修改属性的值
my_new_car.meter=23
my_new_car.read_meter()

# b.通过方法修改属性的值
# 方法见类中update()
# 我们再次加入禁止回调功能
my_new_car.update(24)
my_new_car.read_meter()
my_new_car.update(1)

# 通过方法对属性的值进行递增
# 方法见类中increase
my_old_car=Car('subaru','outback',2015)
print(my_old_car.get_name())

my_old_car.update(23_500)
my_old_car.read_meter()

my_old_car.increase(100)
my_old_car.read_meter()

