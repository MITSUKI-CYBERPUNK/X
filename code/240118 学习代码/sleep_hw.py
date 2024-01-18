# 每隔1秒输出一次“hello, world”，持续1小时
import time

for i in range(3600):
    print('hello, world')
    time.sleep(1)