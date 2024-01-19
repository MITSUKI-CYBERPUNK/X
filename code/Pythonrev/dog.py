class Dog:  # 根据约定，在python中，首字母大写的名称指的是类
    """一次模拟小狗的简单尝试"""

    def __init__(self, name, age): # 注意缩进
        """初始化属性name和age"""

        self.name = name  # 前缀self
        self.age = age  # 前缀self

    def sit(self):
        """模拟小狗收到命令时蹲下"""

        print(f"{self.name} is sitting")

    def roll(self):
        """模拟小狗收到命令时打滚"""

        print(f"{self.name} is rolling")

#创建实例
my_dog=Dog('Willie',6)#使用这两个实参调用Dog类的方法__init__()
ur_dog=Dog('Lucy',3)

print(f"My dog's name is {my_dog.name}")
print(f"My dog's age is {my_dog.age}")
print(f"Ur dog's name is {ur_dog.name}")
print(f"Ur dog's age is {ur_dog.age}")

#调用方法
my_dog.sit()
my_dog.roll()
ur_dog.sit()
ur_dog.roll()