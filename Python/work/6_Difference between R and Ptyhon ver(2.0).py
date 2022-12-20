import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

###막대차트(가로세로)
x = np.arange(3)
years = ['2018', '2019', '2020']
values = [100, 400, 900]
plt.bar(x, values)
plt.xticks(x, years)
plt.show()

y = np.arange(3)
plt.barh(y, values)
plt.yticks(y, years)
plt.show()

#누적막대 차트
columns = ['Players','Goals','Assists']
df = pd.DataFrame(columns=columns)
data = np.array([['R. Lewandowski',41,7],
                   ['L. Messi',30,9],
                   ['H. Kane',23,15],
                   ['K. Mbappe',26,8],
                   ['E. Haaland',27,6],
                   ['S. HeungMin',38,19]],
                    dtype = 'object')
for i in range(len(columns)):
 df[columns[i]] = data[:,i]

plt.bar(df['Players'],df['Goals'])
plt.bar(df['Players'],df['Assists'],bottom=df['Goals'])
plt.legend(columns[1:])
plt.show()

#점차트
mpg=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
plt.figure(figsize=(10,5))
sns.scatterplot(x = 'displ', y = 'hwy',hue="class", data = mpg)
plt.show()

#파이차트
columns = ['Players','Goals','Assists']
df = pd.DataFrame(columns=columns)
data = np.array([['R. Lewandowski',41,7],
                   ['L. Messi',30,9],
                   ['H. Kane',23,15],
                   ['K. Mbappe',26,8],
                   ['E. Haaland',27,6],
                   ['S. HeungMin',38,19]],
                    dtype = 'object')
for i in range(len(columns)):
 df[columns[i]] = data[:,i]

plt.pie(df['Goals'], labels=df['Players'], autopct='%.1f%%')
plt.show()

###박스 플롯
df=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
plt.figure(figsize=(13,10), dpi= 80)
sns.boxplot(x='class', y='hwy', data=df, notch=False)
plt.show()

#히스토그램
mpg=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
plt.hist(mpg['hwy'])
plt.show()

#산점도
mpg=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
plt.scatter(mpg['cty'], mpg['hwy'])
plt.show()

#중첩자료 시각화 count chart
import matplotlib.pyplot as plt
import pandas as pd

mpg=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
df_counts = mpg.groupby(['hwy', 'cty']).size().reset_index(name='counts')

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)

plt.scatter(x=df_counts.cty, y=df_counts.hwy, s=df_counts.counts*10)
plt.show()

#자료간 비교 시각화
df=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mtcars.csv")
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#밀도 그래프
df=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")

plt.figure(figsize=(16,10), dpi= 80)
sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 5, "cty"], shade=True, color="deeppink", label="Cyl=5", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)

plt.title('Density Plot', fontsize=22)
plt.legend()
plt.show()