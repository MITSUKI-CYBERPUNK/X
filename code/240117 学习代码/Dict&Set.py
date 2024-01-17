# 字典
Aqua={'sex':'female',"age":"19"}
print(Aqua['sex'])
print(Aqua['age'])
Aqua['skill']='hack'
#修改键值对
del Aqua['age']
#删除键值对
aquasex=Aqua.get('height','no height')

#遍历字典
#遍历键值对：
for a,b in Aqua.items():
    print(a,b)

#仅遍历所有键
for item in Aqua:
    print(item)
#for item in Aqua.keys():
#keys()更为明显，但也可不写
    
#按特定顺序遍历字典中的所有键：
for item in sorted(Aqua.keys()):
    print(item)

#遍历字典中的所有值：
for name in Aqua.values():
    print(name)