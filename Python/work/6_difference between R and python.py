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

#박스 그래프
mpg=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
category = mpg['class'].unique()  # ['compact', 'midsize', 'suv', '2seater', 'minivan', 'pickup', 'subcompact']

class1 = mpg[mpg['class']==category[0]]
class2 = mpg[mpg['class']==category[1]]
class3 = mpg[mpg['class']==category[2]]
class4 = mpg[mpg['class']==category[3]]
class5 = mpg[mpg['class']==category[4]]
class6 = mpg[mpg['class']==category[5]]
class7 = mpg[mpg['class']==category[6]]

fig, ax = plt.subplots()

ax.boxplot([class1[class1.columns[8]],class2[class2.columns[8]],class3[class3.columns[8]],class4[class4.columns[8]],class5[class5.columns[8]],class6[class6.columns[8]],class7[class7.columns[8]]])
plt.xticks([1,2,3,4,5,6,7],[category[0],category[1],category[2],category[3],category[4],category[5],category[6]])
plt.show()

###박스플롯 버전2

# Draw Plot
df=mpg
plt.figure(figsize=(13,10), dpi= 80)
sns.boxplot(x='class', y='hwy', data=df, notch=False)

# Add N Obs inside boxplot (optional)
def add_n_obs(df,group_col,y):
    medians_dict = {grp[0]:grp[1][y].median() for grp in df.groupby(group_col)}
    xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
    n_obs = df.groupby(group_col)[y].size().values
    for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
        plt.text(x, medians_dict[xticklabel]*1.01, "#obs : "+str(n_ob), horizontalalignment='center', fontdict={'size':14}, color='white')

add_n_obs(df,group_col='class',y='hwy')

# Decoration
plt.title('Box Plot of Highway Mileage by Vehicle Class', fontsize=22)
plt.ylim(10, 40)
plt.show()




#########히스토그램
mpg=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
plt.hist(mpg['hwy'])
plt.show()

######산점도
plt.scatter(mpg['cty'], mpg['hwy'])
plt.show()

######중첩자료 시각화 count chart
import matplotlib.pyplot as plt
import pandas as pd

mpg=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mpg.csv")
df_counts = mpg.groupby(['hwy', 'cty']).size().reset_index(name='counts')

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)

plt.scatter(x=df_counts.cty, y=df_counts.hwy, s=df_counts.counts*10)
plt.show()

#####자료간 비교 시각화
df=pd.read_csv("E:/GoogleDrive/A4팀 프로젝트 자료(12월2일)/6. R,Python visualization/mtcars.csv")
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

###밀도 그래프
df=mpg
# Draw Plot
plt.figure(figsize=(16,10), dpi= 80)
sns.kdeplot(df.loc[df['cyl'] == 4, "cty"], shade=True, color="g", label="Cyl=4", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 5, "cty"], shade=True, color="deeppink", label="Cyl=5", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 6, "cty"], shade=True, color="dodgerblue", label="Cyl=6", alpha=.7)
sns.kdeplot(df.loc[df['cyl'] == 8, "cty"], shade=True, color="orange", label="Cyl=8", alpha=.7)

# Decoration
plt.title('Density Plot', fontsize=22)
plt.legend()
plt.show()