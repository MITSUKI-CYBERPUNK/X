X_JAPAN=['X','J','A','P','A','N']
print(X_JAPAN)

print(X_JAPAN[0].title()) # 索引访问并首字母大写

X_JAPAN[0]='xxx' # 修改元素
print(X_JAPAN)

X_JAPAN.append('wow') # 添加到末尾（添加新的对象）
print(X_JAPAN)

X_JAPAN.extend('wow') # 在末尾追加值
print(X_JAPAN)

X_JAPAN.insert(0,'wow') # 插入元素
print(X_JAPAN)

del X_JAPAN[0] # 删除元素（del语句）
print(X_JAPAN)

X_JAPAN.pop() # 删除末尾元素，弹出栈顶元素
print(X_JAPAN)

X_JAPAN.pop() # 按索引删除任意元素
print(X_JAPAN)

X_JAPAN.remove('J') # 根据值删除元素，但只删除第一个，若有多个，则需用到循环
print(X_JAPAN)

X_JAPAN.sort() # 永久排序，按字母顺序排列，大写字母和数字都排在前面
print(X_JAPAN)

X_JAPAN.sort(reverse=True) # reverse默认为False
print(X_JAPAN)

len(X_JAPAN) # 确定长度