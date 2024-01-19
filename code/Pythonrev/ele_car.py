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

# 创建子类时，父类必须包含在当前文件中，且位于子类之前。
class elecar(Car):# 在括号内指定父类的名称
    """电动汽车的独特之处"""

    def __init__(self, make, model, year):  # 接受创建Car实例所需的信息
        """初始化父类的属性"""
        super().__init__(make, model, year)# super()是一个特殊函数，让你能够调用父类的方法(父类也称超类)
        # 添加特有属性
        self.battery=75

    def des_battery(self):
        """打印一条描述电瓶容量的信息"""
        print(f"Has {self.battery} -kWh")

tesla=elecar('tesla','model s',2019)
print(tesla.get_name())
tesla.des_battery()

