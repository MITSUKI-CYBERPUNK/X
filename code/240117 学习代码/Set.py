s = {2,3,4,2}#自动删除重复数据
print(s)

s.add(1) #如果加入集合中已有元素没有任何影响
print(s)

s1 = {2,3,5,6}

print(s & s1) #取交集 {2, 3}
print(s | s1) #取并集 {1, 2, 3, 4, 5, 6}
print(s - s1) #差集 {1, 4}

s.remove(2) #删除集合s中一个元素，注意只能删除一次
print(s)

s.pop() #随机删除任意一个元素，删完后报错
print(s)

s.clear() #一键清空
print(s)