# 0 初始化
#导入三大库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#忽略警告
import warnings
warnings.filterwarnings('ignore')

#导入数据
train=pd.read_csv(r'D:\数据分析项目\Titanic\train.csv')
train.describe()#描述性统计

#每一列数据类型
train.info()

#查看前五行示例
train.head()


# 1 数据清洗
train.isnull().sum()#查看缺失值

# 下面来对年龄进行数据清洗，移除缺失值：
train = train.dropna(subset=['Age', 'Survived'])


# 2 可视化与数据分析

# 2.1 整体生还率与死亡率
s=train['Survived'].value_counts()
s

plt.rcParams['figure.figsize'] = (5,5)#设置大小
plt.pie(s,autopct='%.2f%%',labels=['Death','Survived'],explode=[0,0.1])#autopct百分比函数
plt.legend()#设置图例
plt.title('Overall Survival Rate')#设置标题
plt.show()


# 2.2 乘客性别对生还(数)率的影响
# 2.2.1 男女生存人数情况

#按照性别分组并计算每个性别的生存人数
sex=train.groupby('Sex')['Survived'].sum()
sex.plot.bar(color='blue')
plt.title('Survived Count by Sex')
plt.show()


# 2.2.2 乘客性别对生还率的影响

train.groupby(['Sex','Survived'])['Survived'].count().unstack().plot(kind='bar',stacked='True')
plt.title('Survived Count by Sex')
plt.ylabel('count')


# 2.3 乘客年龄对生还(数)率的影响

# 2.3.1 各年龄存活人数情况

#根据年龄分组
age_groups = pd.cut(train['Age'], bins=[0, 18, 35, 60, float('inf')], labels=['Child', 'Young Adult', 'Adult', 'Elderly'])

#计算每个年龄组的存活人数
survival_counts = train.groupby(age_groups)['Survived'].sum()

#条形图
survival_counts.plot(kind='bar',color='blue')
plt.xlabel('Age Group')
plt.ylabel('Survival Count')
plt.title('Survival Count by Age Group')
plt.show()


# 2.3.2 乘客年龄对乘客生还率的影响

#计算生还率
survival_rate = train.groupby(age_groups)['Survived'].mean()

#画图
survival_rate.plot(kind='bar',color='red')
plt.xlabel('Age')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Age Group')
plt.show()


# 2.4 乘客船舱等级对生还(数)率的影响

# 2.4.1 不同船舱等级乘客存活人数情况

pclass_groups = train.groupby('Pclass')['Survived'].mean()
pc=train.groupby('Pclass')['Survived'].sum()
pc.plot.bar(color='blue')
plt.title('Survived Count by Pclass')
plt.show()

# 2.4.2 不同船舱等级对乘客生还率的影响

pclass_count = train.groupby(by='Pclass')['Survived'].value_counts()

plt.rcParams['figure.figsize'] = (3*5, 5) # 设置三个饼图的大小

pie1 = plt.subplot(1, 3, 1)
pie1.pie(pclass_count.loc[1][::-1], autopct='%.2f%%', labels=['Death', 'Survived'], explode=[0, 0.1]) # autopct百分比函数
pie1.set_title('Cabin 1') # 设置标题

pie2 = plt.subplot(1, 3, 2)
pie2.pie(pclass_count.loc[2], autopct='%.2f%%', labels=['Death', 'Survived'], explode=[0, 0.1]) # autopct百分比函数
pie2.set_title('Cabin 2') # 设置标题

pie3 = plt.subplot(1, 3, 3)
pie3.pie(pclass_count.loc[3], autopct='%.2f%%', labels=['Death', 'Survived'], explode=[0, 0.1]) # autopct百分比函数
pie3.set_title('Cabin 3') # 设置标题

plt.show()


# 2.5 乘客登船港口对生还(数)率的影响
# 2.5.1 不同登船港口乘客存活人数情况

embarked_groups = train.groupby('Embarked')['Survived'].mean()
pc=train.groupby('Embarked')['Survived'].sum()
pc.plot.bar(color='blue')
plt.title('Survived Count by Embarked')
plt.show()


# 2.5.2 不同登船港口对乘客生还率的影响

embark_count = train.groupby(by='Embarked')['Survived'].value_counts()

plt.rcParams['figure.figsize'] = (3*5, 5) # 设置三个饼图的大小

pie1 = plt.subplot(1, 3, 1)
pie1.pie(embark_count.loc['C'][::-1], autopct='%.2f%%', labels=['Death', 'Survived'], explode=[0, 0.1]) # autopct百分比函数
pie1.set_title('C') # 设置标题

pie2 = plt.subplot(1, 3, 2)
pie2.pie(embark_count.loc['Q'], autopct='%.2f%%', labels=['Death', 'Survived'], explode=[0, 0.1]) # autopct百分比函数
pie2.set_title('Q') # 设置标题

pie3 = plt.subplot(1, 3, 3)
pie3.pie(embark_count.loc['S'], autopct='%.2f%%', labels=['Death', 'Survived'], explode=[0, 0.1]) # autopct百分比函数
pie3.set_title('S') # 设置标题

plt.show()


# 2.6 不同船票价格对乘客生还(数)率的影响
# 2.6.1 不同船票价格乘客存活人数情况

#将票价分成5个范围
train['FareRange'] = pd.cut(train['Fare'], bins=[0, 50, 100, 150, 200, 550])

#统计每个票价范围内的存活人数和总人数
survival_count = train.groupby('FareRange')['Survived'].value_counts().unstack().fillna(0)
total_count = survival_count.sum(axis=1)

#折线图
survival_count.plot(kind='line',grid=True)
plt.xlabel('Fare')
plt.ylabel('Survival Count')
plt.title('Survival Count by Fare')
plt.show()


# 2.6.2 不同船票价格对乘客生存率的影响

#计算生存率
survival_rates = survival_counts[1] / total_count

#折线图
survival_rates.plot(kind='line', grid=True)
plt.xlabel('Fare')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Fare')
plt.show()


# 2.7 不同家庭成员数对乘客生还(数)率的影响
# 2.7.1 不同家庭成员数乘客存活人数情况

#计算每个乘客的家庭成员数
train["Family"] = train["SibSp"] + train["Parch"] + 1

Fam=train.groupby('Family')['Survived'].sum()
Fam.plot.bar(color='blue')
plt.title('Survived Count by Family')
plt.show()

# 2.7.2 不同家庭成员数对乘客生还率的影响

#分组计算生存率
fam_group = train.groupby("Family")["Survived"].mean()

#画图
fam_group.plot(kind='bar',color='red')
plt.xlabel("Family")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Family")
plt.show()
